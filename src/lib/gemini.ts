import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from "@google/generative-ai";

// Initialize the Gemini API with your API key
const API_KEY = import.meta.env.VITE_GEMINI_API_KEY || "";

// Create and configure the Generative AI instance
const genAI = new GoogleGenerativeAI(API_KEY);

// Define the model name - using the latest Gemini 2.0 Flash model
const MODEL_NAME = "gemini-1.5-flash";

// Get the generative model with the specified name
const model = genAI.getGenerativeModel({
  model: MODEL_NAME,
  safetySettings: [
    {
      category: HarmCategory.HARM_CATEGORY_HARASSMENT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
      category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
      category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
      category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
  ],
  generationConfig: {
    temperature: 0.7,
    topK: 40,
    topP: 0.95,
    maxOutputTokens: 2048,
  },
});

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  imageUrl?: string;
}

/**
 * Converts our application message format to Gemini's expected format
 * @param messages Array of chat messages
 * @returns Formatted messages for Gemini API
 */
function formatMessagesForGemini(messages: ChatMessage[]) {
  return messages.map(msg => ({
    role: msg.role === "user" ? "user" : "model",
    parts: [{ text: msg.content }]
  }));
}

/**
 * Generates a response using the Gemini API
 * @param messages Array of chat messages
 * @returns Promise resolving to the generated response text
 */
export async function generateResponse(messages: ChatMessage[]): Promise<string> {
  if (!API_KEY) {
    return "API key is missing. Please check your configuration.";
  }

  try {
    console.log(`[Gemini] Generating response using ${MODEL_NAME} model`);
    console.log(`[Gemini] Message count: ${messages.length}`);
    
    // Input validation
    if (messages.length === 0) {
      console.error("[Gemini] Error: No messages provided");
      return "I need a message to respond to.";
    }
    
    const lastMessage = messages[messages.length - 1];
    if (lastMessage.role !== "user") {
      console.error("[Gemini] Error: Last message is not from user:", lastMessage);
      return "I can only respond to user messages.";
    }
    
    // For Gemini API, we need to ensure the first message in history is from a user
    // If our first message is from assistant, we'll handle it differently
    let chatHistory: ChatMessage[] = [];
    let userMessageToSend = lastMessage.content;
    
    if (messages.length > 1) {
      // If the first message is from the assistant, we'll include it in the user's first message
      if (messages[0].role === "assistant") {
        // Start with the second message (which should be from a user)
        // If there are only two messages and the first is from assistant, we'll just use the last user message
        if (messages.length > 2) {
          chatHistory = messages.slice(1, -1);
        }
      } else {
        // If the first message is from a user, we can use the history as is
        chatHistory = messages.slice(0, -1);
      }
    }
    
    console.log(`[Gemini] History message count: ${chatHistory.length}`);
    console.log(`[Gemini] First message role: ${chatHistory.length > 0 ? chatHistory[0].role : 'N/A'}`);
    
    // Format the chat history for Gemini API
    const formattedHistory = formatMessagesForGemini(chatHistory);
    
    // Create chat session with history (only if we have valid history)
    const chat = formattedHistory.length > 0 ? 
      model.startChat({
        history: formattedHistory,
      }) : 
      model.startChat();
    
    console.log(`[Gemini] Sending message: "${userMessageToSend.substring(0, 50)}${userMessageToSend.length > 50 ? '...' : ''}"`);
    
    // Send the last user message and get response
    const result = await chat.sendMessage(userMessageToSend);
    
    // Validate response
    if (!result || !result.response) {
      console.error("[Gemini] Error: Empty response from API");
      return "I received an empty response. Please try again.";
    }
    
    // Extract text from response
    const text = result.response.text();
    console.log(`[Gemini] Received response (${text.length} chars): "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`);
    
    // Check for blocked response
    if (result.promptFeedback?.blockReason) {
      console.error(`[Gemini] Response blocked: ${result.promptFeedback.blockReason}`);
      return "I'm unable to provide a response to that query due to safety concerns.";
    }
    
    return text;
  } catch (error) {
    // Log detailed error information
    console.error("[Gemini] Error generating response:", error);
    
    // Check for specific error types
    if (error instanceof Error) {
      console.error("[Gemini] Error message:", error.message);
      console.error("[Gemini] Error stack:", error.stack);
      
      // Handle specific error cases
      if (error.message.includes("API key")) {
        return "There's an issue with the API key. Please check your configuration.";
      }
      
      if (error.message.includes("network") || error.message.includes("ECONNREFUSED") || error.message.includes("ETIMEDOUT")) {
        return "I'm having trouble connecting to the AI service. Please check your internet connection and try again.";
      }
      
      if (error.message.includes("quota") || error.message.includes("rate limit") || error.message.includes("429")) {
        return "We've reached the usage limit for the AI service. Please try again later.";
      }
      
      if (error.message.includes("model not found") || error.message.includes("404")) {
        return "The AI model specified is not available. Please contact the administrator.";
      }
      
      if (error.message.includes("First content should be with role 'user'")) {
        return "There was an issue with the conversation format. Let's start a new conversation.";
      }
    }
    
    return "Sorry, I encountered an error while processing your request. Please try again.";
  }
}

/**
 * Verifies the Gemini API connection
 * @returns Promise resolving to a boolean indicating if the connection is successful
 */
export async function verifyGeminiConnection(): Promise<boolean> {
  if (!API_KEY) {
    console.error("[Gemini] API key is missing. Please set VITE_GEMINI_API_KEY.");
    return false;
  }

  try {
    console.log("[Gemini] Verifying API connection...");
    
    const result = await model.generateContent("Hello, this is a test message to verify the API connection.");
    const text = result.response.text();
    
    console.log("[Gemini] Connection verified successfully");
    console.log(`[Gemini] Test response: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`);
    
    return true;
  } catch (error) {
    console.error("[Gemini] Connection verification failed:", error);
    return false;
  }
}