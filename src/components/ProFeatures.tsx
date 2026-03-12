import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Crown, Image, MessageSquare, Zap, CheckCircle, ArrowLeft, AlertCircle, QrCode } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { QRCodeSVG } from 'qrcode.react';
import { getCurrentUser, getUserProfile } from '../lib/supabase';
import { createPaymentOrder } from '../lib/cashfree';
import { load } from '@cashfreepayments/cashfree-js';

const ProFeatures: React.FC = () => {
  const navigate = useNavigate();
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
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
      if (!user) {
        throw new Error('Please sign in to continue');
      }

      const userProfile = await getUserProfile(user.id);

      // Create a unique order ID
      const orderId = `order_${user.id}_${Date.now()}`;
      
      // Call createPaymentOrder to get Cashfree payment session ID
      const paymentOrder = await createPaymentOrder({
        orderId,
        orderAmount: 120, // ₹120 per month
        orderCurrency: 'INR',
        customerEmail: userProfile?.email || user.email || '',
        customerPhone: userProfile?.phone_number || '9999999999',
        customerName: userProfile?.display_name || userProfile?.email?.split('@')[0] || 'User',
        userId: user.id,
      });

      if (!paymentOrder || !paymentOrder.payment_session_id) {
        throw new Error('Failed to create payment session.');
      }

      // Initialize Cashfree SDK
      const cashfree = await load({
        mode: 'production', // Use production as specified
      });

      // Redirect to Cashfree checkout. returnUrl is configured in backend.
      cashfree.checkout({
        paymentSessionId: paymentOrder.payment_session_id,
      });

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
            <div className="bg-white p-4 rounded-lg inline-block mb-4">
              <QRCodeSVG
                value={`upi://pay?pa=your.upi@bank&pn=ShaktimaanGPT&am=120.00&cu=INR&tn=Pro Membership`}
                size={200}
                level="H"
                includeMargin={true}
              />
            </div>
            <p className="text-sm text-gray-400 mb-4">
              Scan the QR code with any UPI app to make the payment
            </p>
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
                  <QrCode size={20} />
                  Confirm Payment
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