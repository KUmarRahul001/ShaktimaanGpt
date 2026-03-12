import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { CheckCircle, XCircle, ArrowLeft, Loader2 } from 'lucide-react';
import { verifyPaymentOrder } from '../lib/cashfree';
import { getCurrentUser } from '../lib/supabase';

const PaymentStatus: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const orderId = searchParams.get('order_id');
  const [status, setStatus] = useState<'success' | 'error' | 'loading'>('loading');
  const [message, setMessage] = useState('Verifying payment status...');

  useEffect(() => {
    const verify = async () => {
      if (!orderId) {
        setStatus('error');
        setMessage('Invalid order details. Please contact support.');
        return;
      }

      try {
        const user = await getCurrentUser();
        if (!user) {
          throw new Error('User not found. Please log in.');
        }

        // Amount passed from return URL just for verification call, but ideally Cashfree verify doesn't need it on standard verify endpoint,
        // However, the `verifyPaymentOrder` edge function might use it, though standard Cashfree just uses orderId.
        // We'll pass 120 as a fallback.
        const amountStr = searchParams.get('amount') || '120';
        const amount = parseFloat(amountStr);

        const verification = await verifyPaymentOrder(orderId, amount);

        // Typical Cashfree order statuses: PAID, ACTIVE, EXPIRED
        if (verification && verification.order_status === 'PAID') {
          // In a real application, the pro status should only be granted by the backend via webhook.
          // The frontend should just display success. It might take a moment for the webhook to update the DB,
          // so the user might need to refresh or the app should listen to real-time updates.
          setStatus('success');
          setMessage('Payment verified. Your account will be upgraded to Pro shortly.');
        } else {
          setStatus('error');
          setMessage('Payment was not completed successfully. Please try again.');
        }

      } catch (error) {
        console.error('Error verifying payment:', error);
        setStatus('error');
        setMessage('Failed to verify payment. If amount was deducted, please contact support.');
      }
    };

    verify();
  }, [orderId, searchParams]);

  return (
    <div className="min-h-screen bg-dark-bg">
      <header className="bg-dark-surface border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 py-6 flex items-center">
          <button
            onClick={() => navigate('/')}
            className="mr-4 text-gray-400 hover:text-white"
          >
            <ArrowLeft size={24} />
          </button>
          <h1 className="text-3xl font-display font-bold neon-purple-text">
            Payment Status
          </h1>
        </div>
      </header>

      <main className="max-w-xl mx-auto px-4 py-12">
        <motion.div
          className="bg-dark-surface p-8 rounded-xl border border-gray-800 text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {status === 'loading' ? (
            <div className="space-y-4">
              <Loader2 className="w-16 h-16 text-neon-purple mx-auto animate-spin" />
              <h2 className="text-2xl font-display font-bold text-white">
                Verifying Payment...
              </h2>
              <p className="text-gray-400">{message}</p>
            </div>
          ) : status === 'success' ? (
            <div className="space-y-4">
              <CheckCircle className="w-16 h-16 text-green-500 mx-auto" />
              <h2 className="text-2xl font-display font-bold text-green-500">
                Payment Successful!
              </h2>
              <p className="text-gray-400">{message}</p>
              <button
                onClick={() => navigate('/')}
                className="mt-6 px-6 py-3 bg-neon-purple hover:bg-purple-600 text-white font-bold rounded-lg transition-all"
              >
                Return to Chat
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <XCircle className="w-16 h-16 text-red-500 mx-auto" />
              <h2 className="text-2xl font-display font-bold text-red-500">
                Payment Failed
              </h2>
              <p className="text-gray-400">{message}</p>
              <button
                onClick={() => navigate('/pro')}
                className="mt-6 px-6 py-3 bg-neon-purple hover:bg-purple-600 text-white font-bold rounded-lg transition-all"
              >
                Try Again
              </button>
            </div>
          )}
        </motion.div>
      </main>
    </div>
  );
};

export default PaymentStatus;