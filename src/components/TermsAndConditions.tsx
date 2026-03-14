import React from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, FileText } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const TermsAndConditions: React.FC = () => {
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
            <FileText />
            Terms and Conditions
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
            <h2 className="text-2xl font-display font-bold mb-6">Terms and Conditions</h2>
            
            <div className="space-y-6 text-gray-300">
              <section>
                <h3 className="text-xl font-semibold mb-3">1. Acceptance of Terms</h3>
                <p>
                  By accessing and using Shaktimaan GPT ("the Service"), you accept and agree
                  to be bound by the terms and conditions outlined in this agreement.
                </p>
              </section>

              <section>
                <h3 className="text-xl font-semibold mb-3">2. Use of Service</h3>
                <p>
                  You agree to use the Service only for lawful purposes and in accordance
                  with these Terms. You are responsible for maintaining the confidentiality
                  of your account credentials.
                </p>
              </section>

              <section>
                <h3 className="text-xl font-semibold mb-3">3. User Content</h3>
                <p>
                  You retain ownership of any content you submit through the Service. By
                  submitting content, you grant us a worldwide, non-exclusive license to use,
                  store, and process that content.
                </p>
              </section>

              <section>
                <h3 className="text-xl font-semibold mb-3">4. Privacy</h3>
                <p>
                  Your privacy is important to us. Please review our Privacy Policy to
                  understand how we collect, use, and protect your personal information.
                </p>
              </section>

              <section>
                <h3 className="text-xl font-semibold mb-3">5. Service Modifications</h3>
                <p>
                  We reserve the right to modify, suspend, or discontinue any part of the
                  Service at any time without notice.
                </p>
              </section>

              <section>
                <h3 className="text-xl font-semibold mb-3">6. Limitation of Liability</h3>
                <p>
                  The Service is provided "as is" without warranties of any kind. We shall
                  not be liable for any damages arising from your use of the Service.
                </p>
              </section>

              <section>
                <h3 className="text-xl font-semibold mb-3">7. Governing Law</h3>
                <p>
                  These Terms shall be governed by and construed in accordance with the laws
                  of the United States, without regard to its conflict of law provisions.
                </p>
              </section>

              <section>
                <h3 className="text-xl font-semibold mb-3">8. Changes to Terms</h3>
                <p>
                  We reserve the right to update these Terms at any time. Continued use of
                  the Service after any modifications constitutes acceptance of the updated
                  Terms.
                </p>
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

export default TermsAndConditions;