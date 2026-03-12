import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from "@google/generative-ai";
import { Handler } from "@netlify/functions";

export const handler: Handler = async (event, context) => {
  // Only allow POST requests
  if (event.httpMethod !== "POST") {
    return {
      statusCode: 405,
      body: JSON.stringify({ error: "Method not allowed" }),
    };
  }

  try {
    const API_KEY = process.env.GEMINI_API_KEY;

    if (!API_KEY) {
      console.error("[Netlify Function] GEMINI_API_KEY is not configured.");
      return {
        statusCode: 500,
        body: JSON.stringify({ error: "API key is missing on the server. Please configure GEMINI_API_KEY." }),
      };
    }

    const genAI = new GoogleGenerativeAI(API_KEY);
    const MODEL_NAME = "gemini-1.5-flash";
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

    const body = JSON.parse(event.body);

    // If it's a verification ping
    if (body.action === "verify") {
      const result = await model.generateContent("Hello, this is a test message to verify the API connection.");
      const text = result.response.text();
      return {
        statusCode: 200,
        body: JSON.stringify({ success: true, message: "Connection verified" }),
      };
    }

    // Otherwise, generate response for chat
    const messages = body.messages || [];

    if (messages.length === 0) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: "No messages provided" }),
      };
    }

    const lastMessage = messages[messages.length - 1];
    if (lastMessage.role !== "user") {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: "Last message is not from user" }),
      };
    }

    let chatHistory: any[] = [];
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

    const formattedHistory = chatHistory.map((msg: any) => ({
      role: msg.role === "user" ? "user" : "model",
      parts: [{ text: msg.content }]
    }));

    const chat = formattedHistory.length > 0 ?
      model.startChat({ history: formattedHistory }) :
      model.startChat();

    const result = await chat.sendMessage(userMessageToSend);

    if (!result || !result.response) {
      return {
        statusCode: 500,
        body: JSON.stringify({ error: "Empty response from API" }),
      };
    }

    const text = result.response.text();

    if (result.promptFeedback?.blockReason) {
      return {
        statusCode: 403,
        body: JSON.stringify({ error: "Response blocked due to safety concerns." }),
      };
    }

    return {
      statusCode: 200,
      body: JSON.stringify({ text }),
    };

  } catch (error: any) {
    console.error("[Netlify Function] Error:", error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message || "Internal Server Error" }),
    };
  }
}