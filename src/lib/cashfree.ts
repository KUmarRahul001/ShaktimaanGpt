import { v4 as uuidv4 } from 'uuid';

interface PaymentDetails {
  orderId: string;
  orderAmount: number;
  orderCurrency: string;
  customerEmail: string;
  customerPhone?: string;
  customerName?: string;
}

export async function createPaymentOrder(details: PaymentDetails) {
  try {
    const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
    const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

    if (!supabaseUrl || !supabaseAnonKey) {
      throw new Error('Missing Supabase configuration');
    }

    // Validate required fields
    if (!details.orderAmount || details.orderAmount <= 0) {
      throw new Error('Invalid order amount');
    }

    if (!details.customerEmail) {
      throw new Error('Customer email is required');
    }

    const response = await fetch(`${supabaseUrl}/functions/v1/create-payment`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${supabaseAnonKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        orderId: details.orderId,
        orderAmount: details.orderAmount,
        orderCurrency: details.orderCurrency || 'INR',
        customerEmail: details.customerEmail,
        customerPhone: details.customerPhone,
        customerName: details.customerName,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      console.error('Payment service error response:', errorData);
      throw new Error(`Payment service error: ${errorData}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error creating payment order:', error);
    throw error;
  }
}

export async function verifyPaymentOrder(orderId: string, orderAmount: number) {
  try {
    const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
    const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

    if (!supabaseUrl || !supabaseAnonKey) {
      throw new Error('Missing Supabase configuration');
    }

    const response = await fetch(`${supabaseUrl}/functions/v1/verify-payment?orderId=${orderId}&amount=${orderAmount}`, {
      headers: {
        'Authorization': `Bearer ${supabaseAnonKey}`,
      },
    });

    if (!response.ok) {
      throw new Error('Failed to verify payment order');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error verifying payment:', error);
    throw error;
  }
}