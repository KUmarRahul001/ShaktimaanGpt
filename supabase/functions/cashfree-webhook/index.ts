import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3"
import { createHmac } from "node:crypto";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
}

function verifySignature(payload: string, signature: string, secret: string, timestamp: string): boolean {
  try {
    const dataToHash = timestamp + payload;
    const generatedSignature = createHmac("sha256", secret)
      .update(dataToHash)
      .digest("base64");
    return generatedSignature === signature;
  } catch (error) {
    console.error("Signature verification error:", error);
    return false;
  }
}

serve(async (req) => {
  // Handle CORS
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders })
  }

  try {
    const CASHFREE_SECRET_KEY = Deno.env.get("CASHFREE_SECRET_KEY")
    if (!CASHFREE_SECRET_KEY) {
      throw new Error("Cashfree secret not configured")
    }

    const signature = req.headers.get("x-webhook-signature")
    const timestamp = req.headers.get("x-webhook-timestamp")

    if (!signature || !timestamp) {
      return new Response("Missing signature or timestamp headers", { status: 400 })
    }

    const rawBody = await req.text()

    if (!verifySignature(rawBody, signature, CASHFREE_SECRET_KEY, timestamp)) {
      console.error("Invalid webhook signature")
      return new Response("Invalid signature", { status: 401 })
    }

    const event = JSON.parse(rawBody)

    // Ensure it's a successful payment event
    if (event.type !== "PAYMENT_SUCCESS_WEBHOOK") {
      return new Response("Event ignored", { status: 200 })
    }

    const orderData = event.data.order;
    const paymentData = event.data.payment;

    // In create-payment, we stored user_id in order_tags
    const userId = orderData.order_tags?.user_id;

    if (!userId) {
      console.error("No user_id found in order_tags");
      return new Response("User ID missing from order", { status: 400 })
    }

    // Initialize Supabase admin client to bypass RLS for updating
    const supabaseUrl = Deno.env.get("SUPABASE_URL")
    const supabaseServiceRoleKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")

    if (!supabaseUrl || !supabaseServiceRoleKey) {
      throw new Error("Supabase environment variables not set")
    }

    const supabaseAdmin = createClient(supabaseUrl, supabaseServiceRoleKey)

    // Update user profile to Pro
    // We add 30 days to current date
    const expirationDate = new Date()
    expirationDate.setDate(expirationDate.getDate() + 30)

    const { error: updateError } = await supabaseAdmin
      .from("profiles")
      .update({
        is_pro: true,
        pro_expires_at: expirationDate.toISOString()
      })
      .eq("id", userId)

    if (updateError) {
      console.error("Error updating profile:", updateError);
      return new Response("Failed to update profile", { status: 500 })
    }

    console.log(`Successfully upgraded user ${userId} to Pro`);
    return new Response("Webhook processed successfully", { status: 200 })

  } catch (error) {
    console.error("Webhook processing error:", error)
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500 }
    )
  }
})