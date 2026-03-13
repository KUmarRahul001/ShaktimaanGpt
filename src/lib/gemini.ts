export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  imageUrl?: string;
}

/**
 * Generates a response by calling the Netlify serverless function
 * @param messages Array of chat messages
 * @returns Promise resolving to the generated response text
 */
export async function generateResponse(messages: ChatMessage[]): Promise<string> {
  try {
    const response = await fetch('/.netlify/functions/gemini-chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ messages }),
    });

    const data = await response.json();

    if (!response.ok) {
      if (response.status === 403 || data.error?.includes("safety concerns")) {
        return "I'm unable to provide a response to that query due to safety concerns.";
      }
      if (response.status === 500 && data.error?.includes("API key")) {
        return "The AI service is misconfigured. The server is missing the API key.";
      }
      throw new Error(data.error || 'Server responded with an error');
    }

    if (!data.text) {
      return "I received an empty response. Please try again.";
    }

    return data.text;
  } catch (error) {
    console.error("[Gemini Client] Error generating response:", error);
    
    if (error instanceof Error) {
      if (error.message.includes("network") || error.message.includes("Failed to fetch") || error.message.includes("ECONNREFUSED") || error.message.includes("ETIMEDOUT")) {
        return "I'm having trouble connecting to the AI service. Please check your internet connection and try again.";
      }
      if (error.message.includes("quota") || error.message.includes("rate limit") || error.message.includes("429")) {
        return "We've reached the usage limit for the AI service. Please try again later.";
      }
    }
    
    return "Sorry, I encountered an error while processing your request. Please try again.";
  }
}

/**
 * Verifies the Gemini API connection
 * @returns Promise resolving to a boolean indicating if the connection is successful
 */
export async function verifyGeminiConnection(): Promise<boolean | 'usage_exceeded'> {
  try {
    const response = await fetch('/.netlify/functions/gemini-chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ action: "verify" }),
    });

    if (response.status === 503) {
      return 'usage_exceeded';
    }

    if (!response.ok) {
      console.error("[Gemini Client] Connection verification failed with status:", response.status);
      return false;
    }

    const data = await response.json();
    return !!data.success;
  } catch (error) {
    console.error("[Gemini Client] Connection verification failed:", error);
    return false;
  }
}