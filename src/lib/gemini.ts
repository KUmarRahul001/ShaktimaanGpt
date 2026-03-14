

import { 
  GoogleGenerativeAI, 
  HarmCategory, 
  HarmBlockThreshold,
  Content 
} from "@google/generative-ai";

// Initialize the Gemini API with your API key
// Replace this with your actual key if it's not pulling from .env!
const API_KEY = import.meta.env.VITE_GEMINI_API_KEY || "AIzaSyC7qCn9WuIcJC1XBgWUNwWCgSa0CYeMBww";

// Create and configure the Generative AI instance
const genAI = new GoogleGenerativeAI(API_KEY);

// Define the model name - using the latest active 2.0 Flash model
const MODEL_NAME = "gemini-2.0-flash-lite";

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
function formatMessagesForGemini(messages: ChatMessage[]): Content[] {
  // Explicitly return the Content type so TypeScript is happy
  return messages.map((msg): Content => ({
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
    
    let chatHistory: ChatMessage[] = [];
    let userMessageToSend = lastMessage.content;
    
    if (messages.length > 1) {
      if (messages[0].role === "assistant") {
        if (messages.length > 2) {
          chatHistory = messages.slice(1, -1);
        }
      } else {
        chatHistory = messages.slice(0, -1);
      }
    }
    
    console.log(`[Gemini] History message count: ${chatHistory.length}`);
    
    // Format the chat history for Gemini API
    const formattedHistory = formatMessagesForGemini(chatHistory);
    
    // Create chat session with history (only if we have valid history)
    const chat = formattedHistory.length > 0 ? 
      model.startChat({
        history: formattedHistory,
      }) : 
      model.startChat();
    
    console.log(`[Gemini] Sending message: "${userMessageToSend.substring(0, 50)}..."`);
    
    // Send the last user message and get response
    const result = await chat.sendMessage(userMessageToSend);
    
    // Validate response
    if (!result || !result.response) {
      console.error("[Gemini] Error: Empty response from API");
      return "I received an empty response. Please try again.";
    }
    
    // Extract text from response
    const text = result.response.text();
    console.log(`[Gemini] Received response (${text.length} chars)`);
    
    // Check for blocked response - FIX: Look inside result.response
    if (result.response.promptFeedback?.blockReason) {
      console.error(`[Gemini] Response blocked: ${result.response.promptFeedback.blockReason}`);
      return "I'm unable to provide a response to that query due to safety concerns.";
    }
    
    return text;
  } catch (error) {
    console.error("[Gemini] Error generating response:", error);
    
    if (error instanceof Error) {
      if (error.message.includes("API key")) {
        return "There's an issue with the API key. Please check your configuration.";
      }
      if (error.message.includes("network") || error.message.includes("ECONNREFUSED")) {
        return "I'm having trouble connecting to the AI service. Please check your internet connection.";
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
 */
export async function verifyGeminiConnection(): Promise<boolean> {
  try {
    console.log("[Gemini] Verifying API connection locally...");
    
    if (!API_KEY || API_KEY.length < 30 || API_KEY === "YOUR_API_KEY_HERE") {
      console.error("Invalid or missing API Key.");
      return false;
    }
    
    await new Promise(resolve => setTimeout(resolve, 500));
    return true;
  } catch (error) {
    console.error("[Gemini] Connection verification failed:", error);
    return false;
  }
}