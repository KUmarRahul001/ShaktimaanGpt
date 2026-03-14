// If running in Deno, ensure your editor supports Deno (e.g., install the Deno VSCode extension).
// If running in Node.js, use the native http module instead:
import { createServer } from "http";

// Replace all usage of 'serve' below with Node.js compatible code if not using Deno.

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
}

createServer(async (req, res) => {

  // Handle CORS
  if (req.method === "OPTIONS") {
    res.writeHead(200, corsHeaders);
    res.end("ok");
    return;
  }

  try {

    // Read secrets from Supabase environment
    // @ts-ignore: Deno global is available in Supabase Edge Functions
    const CASHFREE_APP_ID = Deno.env.get("CASHFREE_APP_ID")
    // @ts-ignore: Deno global is available in Supabase Edge Functions
    const CASHFREE_SECRET_KEY = Deno.env.get("CASHFREE_SECRET_KEY")
    const CASHFREE_BASE_URL = "https://api.cashfree.com/pg"

    if (!CASHFREE_APP_ID || !CASHFREE_SECRET_KEY) {
      throw new Error("Cashfree credentials not configured")
    }

    const { orderId, orderAmount, orderCurrency, customerEmail, customerPhone, customerName } = await req.json()

    if (!orderId || !orderAmount || !customerEmail) {
      throw new Error("Missing required fields: orderId, orderAmount, customerEmail")
    }

    const payload = {
      order_id: orderId,
      order_amount: parseFloat(orderAmount),
      order_currency: orderCurrency || "INR",
      customer_details: {
        customer_id: orderId,
        customer_email: customerEmail,
        customer_phone: customerPhone || "",
        customer_name: customerName || customerEmail.split("@")[0],
      },
      order_meta: {
        return_url: `${req.headers.get("origin")}/payment-status?order_id={order_id}`,
      },
    }

    const response = await fetch(`${CASHFREE_BASE_URL}/orders`, {
      method: "POST",
      headers: {
        "x-api-version": "2022-09-01",
        "x-client-id": CASHFREE_APP_ID,
        "x-client-secret": CASHFREE_SECRET_KEY,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    })

    const responseText = await response.text()

    if (!response.ok) {
      throw new Error(`Payment service error: ${response.status} ${responseText}`)
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