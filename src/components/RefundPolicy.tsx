import React from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, RefreshCw } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const RefundPolicy: React.FC = () => {
  const navigate = useNavigate();

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
            <RefreshCw />
            Refund Policy
          </h1>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="prose prose-invert max-w-none"
        >
          <div className="bg-dark-surface p-8 rounded-lg border border-gray-800">
            <h2 className="text-2xl font-display font-bold mb-6">Refund and Cancellation Policy</h2>
            
            <div className="space-y-6 text-gray-300">
              <section>
                <h3 className="text-xl font-semibold mb-3">1. Refund Eligibility</h3>
                <p>
                  We offer a 30-day money-back guarantee on all Pro subscriptions. If you're
                  not satisfied with our service, you can request a full refund within 30
                  days of your initial purchase.
                </p>
              </section>

              <section>
                <h3 className="text-xl font-semibold mb-3">2. Refund Process</h3>
                <p>
                  To request a refund:
                </p>
                <ul className="list-disc pl-6 mt-2 space-y-2">
                  <li>Contact our support team at support@shaktimaangpt.com</li>
                  <li>Include your account email and reason for the refund</li>
                  <li>Refunds are typically processed within 5-7 business days</li>
                  <li>The refund will be issued to the original payment method</li>
                </ul>
              </section>

              <section>
                <h3 className="text-xl font-semibold mb-3">3. Subscription Cancellation</h3>
                <p>
                  You can cancel your subscription at any time:
                </p>
                <ul className="list-disc pl-6 mt-2 space-y-2">
                  <li>Log into your account and go to subscription settings</li>
                  <li>Click on "Cancel Subscription"</li>
                  <li>Your access will continue until the end of the current billing period</li>
                  <li>No partial refunds for unused portions of the subscription</li>
                </ul>
              </section>

              <section>
                <h3 className="text-xl font-semibold mb-3">4. Non-Refundable Items</h3>
                <p>
                  The following are not eligible for refunds:
                </p>
                <ul className="list-disc pl-6 mt-2 space-y-2">
                  <li>Subscriptions active for more than 30 days</li>
                  <li>Accounts that have violated our Terms of Service</li>
                  <li>Special promotional offers marked as non-refundable</li>
                </ul>
              </section>

              <section>
                <h3 className="text-xl font-semibold mb-3">5. Contact Information</h3>
                <p>
                  For any questions about our refund policy or to request a refund, please
                  contact us:
                </p>
                <ul className="list-disc pl-6 mt-2 space-y-2">
                  <li>Email: support@shaktimaangpt.com</li>
                  <li>Phone: +1 (555) 123-4567</li>
                  <li>Response Time: Within 24 hours during business days</li>
                </ul>
              </section>
            </div>

            <div className="mt-8 p-4 bg-dark-surface-2 rounded-lg">
              <p className="text-sm text-gray-400">
                Last updated: March 19, 2024
              </p>
            </div>
          </div>
        </motion.div>
      </main>
    </div>
  );
};

export default RefundPolicy;