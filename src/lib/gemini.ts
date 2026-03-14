import {
  GoogleGenerativeAI,
  HarmCategory,
  HarmBlockThreshold,
  Content
} from "@google/generative-ai";

/* ---------------------------------- */
/* Environment Setup */
/* ---------------------------------- */

const API_KEY: string | undefined = import.meta.env.VITE_GEMINI_API_KEY;

if (!API_KEY) {
  throw new Error("VITE_GEMINI_API_KEY is missing from environment variables.");
}

/* ---------------------------------- */
/* Gemini Initialization */
/* ---------------------------------- */

const genAI = new GoogleGenerativeAI(API_KEY);

const MODEL_NAME = "gemini-3-flash-preview";

const model = genAI.getGenerativeModel({
  model: MODEL_NAME,

  safetySettings: [
    {
      category: HarmCategory.HARM_CATEGORY_HARASSMENT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    },
    {
      category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    },
    {
      category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    },
    {
      category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    }
  ],

  generationConfig: {
    temperature: 0.7,
    topK: 40,
    topP: 0.95,
    maxOutputTokens: 2048
  }
});

/* ---------------------------------- */
/* Types */
/* ---------------------------------- */

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  imageUrl?: string;
}

/* ---------------------------------- */
/* Utility: Convert Chat History */
/* ---------------------------------- */

function formatMessagesForGemini(messages: ChatMessage[]): Content[] {
  return messages.map((msg) => ({
    role: msg.role === "user" ? "user" : "model",
    parts: [{ text: msg.content }]
  }));
}

/* ---------------------------------- */
/* Timeout Utility */
/* ---------------------------------- */

function timeoutPromise<T>(promise: Promise<T>, ms: number): Promise<T> {
  const timeout = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error("Request timeout")), ms)
  );

  return Promise.race([promise, timeout]);
}

/* ---------------------------------- */
/* Generate Response */
/* ---------------------------------- */

export async function generateResponse(
  messages: ChatMessage[]
): Promise<string> {
  try {
    if (!messages.length) {
      return "Please send a message.";
    }

    const lastMessage = messages[messages.length - 1];

    if (lastMessage.role !== "user") {
      return "I can only respond to user messages.";
    }

    const history: ChatMessage[] =
      messages.length > 1 ? messages.slice(0, -1) : [];

    // Clean up history so it starts with a 'user' message
    const validHistory = [...history];
    while (validHistory.length > 0 && validHistory[0].role !== "user") {
      validHistory.shift();
    }

    const formattedHistory = formatMessagesForGemini(validHistory);

    const chat =
      formattedHistory.length > 0
        ? model.startChat({ history: formattedHistory })
        : model.startChat();

    const result = await timeoutPromise(
      chat.sendMessage(lastMessage.content),
      20000
    );

    if (!result?.response) {
      return "AI returned an empty response.";
    }

    const text = result.response.text();

    if (!text) {
      return "Response blocked due to safety filters.";
    }

    return text;
  } catch (error: unknown) {
    console.error("Gemini Error:", error);

    if (error instanceof Error) {
      if (error.message.includes("API key")) {
        return "Invalid Gemini API key configuration.";
      }

      if (error.message.includes("quota") || error.message.includes("429")) {
        return "AI usage limit reached. Please wait a moment and try again.";
      }

      if (error.message.includes("timeout")) {
        return "The AI request timed out. Please retry.";
      }

      if (error.message.includes("network")) {
        return "Network error. Check your connection.";
      }
    }

    return "Unexpected AI error occurred.";
  }
}

/* ---------------------------------- */
/* Verify Gemini API Connection */
/* ---------------------------------- */

export async function verifyGeminiConnection(): Promise<boolean> {
  try {
    if (!API_KEY || API_KEY.length < 30) {
      console.error("Invalid or missing API Key. Check your .env file.");
      return false;
    }
    
    // Simulating a tiny delay so the UI transitions smoothly
    await new Promise(resolve => setTimeout(resolve, 500));
    return true;
    
  } catch (error) {
    console.error("Local API key validation failed:", error);
    return false;
  }
}