import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { CheckCircle, XCircle, ArrowLeft, Loader2 } from 'lucide-react';
import { verifyPaymentOrder } from '../lib/cashfree';
import { getCurrentUser, updateProStatus } from '../lib/supabase';

const PaymentStatus: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    const verifyPayment = async () => {
      try {
        const orderId = searchParams.get('order_id');

        // If order_id is missing, check if status is already passed (for fallback)
        if (!orderId) {
          const queryStatus = searchParams.get('status');
          if (queryStatus === 'success' || queryStatus === 'error') {
            setStatus(queryStatus);
          } else {
            throw new Error('Invalid payment reference');
          }
          return;
        }

        const user = await getCurrentUser();
        if (!user) {
          throw new Error('User not found. Please log in again.');
        }

        // Hardcoded amount as per current implementation
        const orderAmount = 120;

        const response = await verifyPaymentOrder(orderId, orderAmount);

        // Ensure successful verification
        if (response && response.payment_status === 'SUCCESS') {
          await updateProStatus(user.id, true, new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString());
          setStatus('success');
        } else {
          throw new Error(response.payment_message || 'Payment verification failed');
        }

      } catch (error) {
        console.error('Payment verification error:', error);
        setErrorMessage(error instanceof Error ? error.message : 'Payment verification failed');
        setStatus('error');
      }
    };

    verifyPayment();
  }, [searchParams]);

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
              <h2 className="text-2xl font-display font-bold text-neon-purple">
                Verifying Payment...
              </h2>
              <p className="text-gray-400">Please wait while we confirm your payment status.</p>
            </div>
          ) : status === 'success' ? (
            <div className="space-y-4">
              <CheckCircle className="w-16 h-16 text-green-500 mx-auto" />
              <h2 className="text-2xl font-display font-bold text-green-500">
                Payment Successful!
              </h2>
              <p className="text-gray-400">Your account has been upgraded to Pro.</p>
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
              <p className="text-gray-400">{errorMessage || 'Please try again or contact support if the issue persists.'}</p>
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