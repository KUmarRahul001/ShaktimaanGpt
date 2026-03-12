import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
}

serve(async (req) => {
  // Handle CORS
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders })
  }

  try {
    const CASHFREE_APP_ID = Deno.env.get("CASHFREE_APP_ID")
    const CASHFREE_SECRET_KEY = Deno.env.get("CASHFREE_SECRET_KEY")
    const CASHFREE_BASE_URL = "https://api.cashfree.com/pg"

    if (!CASHFREE_APP_ID || !CASHFREE_SECRET_KEY) {
      throw new Error("Cashfree credentials not configured")
    }

    const url = new URL(req.url)
    const orderId = url.searchParams.get("orderId")

    if (!orderId) {
      throw new Error("Missing required field: orderId")
    }

    const response = await fetch(`${CASHFREE_BASE_URL}/orders/${orderId}`, {
      method: "GET",
      headers: {
        "x-api-version": "2022-09-01",
        "x-client-id": CASHFREE_APP_ID,
        "x-client-secret": CASHFREE_SECRET_KEY,
        "Content-Type": "application/json",
      },
    })

    const responseText = await response.text()

    if (!response.ok) {
      throw new Error(`Payment verification error: ${response.status} ${responseText}`)
    }

    return new Response(responseText, {
      headers: {
        "Content-Type": "application/json",
        ...corsHeaders,
      },
    })
  } catch (error) {
    return new Response(
      JSON.stringify({
        error: error instanceof Error ? error.message : "Unknown error",
      }),
      {
        status: 500,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders,
        },
      }
    )
  }
})