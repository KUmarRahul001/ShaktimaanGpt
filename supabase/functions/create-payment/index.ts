import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const CASHFREE_APP_ID = '9437976249bfdc5332e584df9b797349';
    const CASHFREE_SECRET_KEY = 'cfsk_ma_prod_6e43bdbab067bfa7c28ff5f112f33f96_f6b2fb87';
    const CASHFREE_BASE_URL = 'https://api.cashfree.com/pg';

    const { orderId, orderAmount, orderCurrency, customerEmail, customerPhone, customerName } = await req.json();

    // Validate required fields
    if (!orderId || !orderAmount || !customerEmail) {
      throw new Error('Missing required fields: orderId, orderAmount, and customerEmail are mandatory');
    }

    const payload = {
      order_id: orderId,
      order_amount: parseFloat(orderAmount),
      order_currency: orderCurrency || 'INR',
      customer_details: {
        customer_id: orderId,
        customer_email: customerEmail,
        customer_phone: customerPhone || '',
        customer_name: customerName || customerEmail.split('@')[0],
      },
      order_meta: {
        return_url: `${req.headers.get('origin')}/payment-status?order_id={order_id}`,
      },
    };

    console.log('Sending request to Cashfree:', JSON.stringify(payload, null, 2));

    const response = await fetch(`${CASHFREE_BASE_URL}/orders`, {
      method: 'POST',
      headers: {
        'x-api-version': '2022-09-01',
        'x-client-id': CASHFREE_APP_ID,
        'x-client-secret': CASHFREE_SECRET_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const responseText = await response.text();
    console.log('Cashfree API response:', responseText);

    if (!response.ok) {
      throw new Error(`Payment service error: ${response.status}\nResponse: ${responseText}`);
    }

    const data = JSON.parse(responseText);
    
    return new Response(
      JSON.stringify(data),
      { 
        headers: { 
          'Content-Type': 'application/json',
          ...corsHeaders
        } 
      }
    );
  } catch (error) {
    console.error('Payment processing error:', error);
    
    return new Response(
      JSON.stringify({ 
        error: error instanceof Error ? error.message : 'Unknown error occurred',
        details: error instanceof Error ? error.stack : undefined
      }),
      { 
        status: 500,
        headers: { 
          'Content-Type': 'application/json',
          ...corsHeaders
        }
      }
    );
  }
})