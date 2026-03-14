import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Crown, Image, MessageSquare, Zap, CheckCircle, ArrowLeft, AlertCircle, CreditCard } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { getCurrentUser, getUserProfile, UserProfile } from '../lib/supabase';
import { createPaymentOrder } from '../lib/cashfree';
import { v4 as uuidv4 } from 'uuid';

const ProFeatures: React.FC = () => {
  const navigate = useNavigate();
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);

  useEffect(() => {
    const fetchUser = async () => {
      const user = await getCurrentUser();
      if (user) {
        const profile = await getUserProfile(user.id);
        setUserProfile(profile);
      }
    };
    fetchUser();
  }, []);

  const features = [
    {
      icon: Image,
      title: 'Image Chat',
      description: 'Upload and discuss images with Shaktimaan GPT'
    },
    {
      icon: MessageSquare,
      title: 'Unlimited Messages',
      description: 'No daily message limits'
    },
    {
      icon: Zap,
      title: 'Priority Access',
      description: 'Faster response times and priority support'
    }
  ];

  const handlePayment = async () => {
    try {
      setIsProcessing(true);
      setError(null);

      const user = await getCurrentUser();
      if (!user || !userProfile) {
        throw new Error('Please sign in to continue');
      }

      const orderId = `order_${uuidv4().replace(/-/g, '')}`;

      const paymentDetails = {
        orderId: orderId,
        orderAmount: 120,
        orderCurrency: 'INR',
        customerEmail: userProfile.email || user.email || 'customer@example.com',
        customerPhone: userProfile.phone_number || '9999999999',
        customerName: userProfile.display_name || 'Customer'
      };

      const response = await createPaymentOrder(paymentDetails);

      if (response && response.payment_session_id) {
        // Redirect to Cashfree checkout using the payment link or payment session
        if (response.payment_link) {
            window.location.href = response.payment_link;
        } else {
            // Cashfree usually returns payment_session_id. We might need a frontend redirect SDK or the backend to return the exact redirect URL.
            // Often backend returns payment_link in this format.
            // If the backend returns `payment_session_id`, the standard cashfree redirect is usually required. Let's assume the backend provides a direct link in `payment_link`.
            // Wait, the instructions said:
            // "receive payment_link \n window.location.href = payment_link"
            if (response.payment_link) {
              window.location.href = response.payment_link;
            } else if (response.payment_session_id) {
              // fallback if backend only returned payment_session_id and expected SDK, but we are asked to redirect directly
              // If the user said "receive payment_link", let's assume `response.payment_link` is provided by the Supabase Edge Function
              window.location.href = response.payment_link || response.payment_url;
            } else {
              throw new Error("Invalid response from payment server");
            }
        }
      } else if (response && response.payment_link) {
        window.location.href = response.payment_link;
      } else {
        throw new Error('Failed to initiate payment session');
      }

    } catch (error) {
      console.error('Payment error:', error);
      setError(error instanceof Error ? error.message : 'An unexpected error occurred');
      setIsProcessing(false);
    }
  };

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
          <h1 className="text-3xl font-display font-bold neon-purple-text flex items-center gap-2">
            <Crown className="text-neon-purple" />
            Upgrade to Pro
          </h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-12">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-display font-bold mb-4">
            Unlock the Full <span className="neon-purple-text">Spiritual Experience</span>
          </h2>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            Enhance your spiritual journey with premium features and unlimited access to Shaktimaan GPT's wisdom.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-12">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              className="bg-dark-surface p-6 rounded-xl border border-gray-800"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <feature.icon className="text-neon-purple mb-4" size={32} />
              <h3 className="text-xl font-display font-semibold mb-2">{feature.title}</h3>
              <p className="text-gray-400">{feature.description}</p>
            </motion.div>
          ))}
        </div>

        <div className="bg-dark-surface rounded-xl border border-gray-800 p-8 max-w-2xl mx-auto">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h3 className="text-2xl font-display font-bold mb-2">Pro Membership</h3>
              <p className="text-gray-400">Unlimited access to all features</p>
            </div>
            <div className="text-right">
              <div className="text-3xl font-display font-bold text-neon-purple">₹120</div>
              <div className="text-gray-400">per month</div>
            </div>
          </div>

          <div className="space-y-4 mb-8">
            <div className="flex items-center gap-2 text-gray-300">
              <CheckCircle className="text-neon-purple" size={20} />
              <span>Upload and analyze images</span>
            </div>
            <div className="flex items-center gap-2 text-gray-300">
              <CheckCircle className="text-neon-purple" size={20} />
              <span>Unlimited messages per day</span>
            </div>
            <div className="flex items-center gap-2 text-gray-300">
              <CheckCircle className="text-neon-purple" size={20} />
              <span>Priority response time</span>
            </div>
            <div className="flex items-center gap-2 text-gray-300">
              <CheckCircle className="text-neon-purple" size={20} />
              <span>24/7 priority support</span>
            </div>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-900/30 border border-red-800 rounded-lg flex items-center gap-2 text-red-300">
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}

          <div className="text-center">
            <button
              onClick={handlePayment}
              disabled={isProcessing}
              className="w-full py-4 bg-neon-purple hover:bg-purple-600 text-white font-bold rounded-lg transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isProcessing ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Processing...
                </>
              ) : (
                <>
                  <CreditCard size={20} />
                  Upgrade to Pro
                </>
              )}
            </button>
          </div>
          
          <p className="text-center text-gray-400 mt-4 text-sm">
            30-day money-back guarantee • Cancel anytime
          </p>
        </div>
      </main>
    </div>
  );
};

export default ProFeatures;