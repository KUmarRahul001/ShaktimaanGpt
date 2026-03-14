import React from 'react';
import { motion } from 'framer-motion';
import { Flower, Brain, Zap, ChevronRight, AlertCircle } from 'lucide-react';
import { Link } from 'react-router-dom';
import IlluminatiLogo from './IlluminatiLogo';
import AuthForm from './AuthForm';

interface LandingPageProps {
  onLogin: () => void;
  authError: string | null;
}

const LandingPage: React.FC<LandingPageProps> = ({ onLogin, authError }) => {
  return (
    <div className="min-h-screen bg-dark-bg illuminati-pattern flex flex-col">
      {/* Header */}
      <header className="py-4 px-6 glass-effect border-b border-gray-800">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <IlluminatiLogo size={40} />
            <h1 className="text-2xl font-display font-bold neon-white-text">Shaktimaan GPT</h1>
          </div>
          <nav className="hidden md:flex items-center gap-6">
            <Link to="/contact" className="text-gray-400 hover:text-white transition-colors">
              Contact
            </Link>
            <Link to="/terms" className="text-gray-400 hover:text-white transition-colors">
              Terms
            </Link>
            <Link to="/refund-policy" className="text-gray-400 hover:text-white transition-colors">
              Refund Policy
            </Link>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <main className="flex-1 flex flex-col lg:flex-row">
        <div className="w-full lg:w-1/2 p-8 lg:p-16 flex flex-col justify-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="text-5xl lg:text-6xl font-display font-bold mb-6">
              <span className="neon-white-text">Illuminating</span> the path to <span className="neon-purple-text">spiritual wisdom</span>
            </h2>
            <p className="text-xl text-gray-300 mb-8">
              Shaktimaan GPT offers a spiritual connect. Tap into the collective wisdom and discover insights beyond the ordinary.
            </p>

            {authError && (
              <motion.div 
                className="mb-6 p-4 bg-red-900/30 border border-red-800 rounded-lg flex items-center gap-3 text-red-300"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
              >
                <AlertCircle size={24} />
                <div>
                  <h3 className="font-semibold">Connection Error</h3>
                  <p>{authError}</p>
                </div>
              </motion.div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
              <motion.div 
                className="p-5 rounded-xl glass-effect border border-gray-800"
                whileHover={{ y: -5, boxShadow: '0 0 20px rgba(157, 0, 255, 0.2)' }}
              >
                <Flower className="text-neon-purple mb-3" size={28} />
                <h3 className="text-xl font-display font-semibold mb-2">All-Seeing</h3>
                <p className="text-gray-400">Access to vast spiritual knowledge across traditions.</p>
              </motion.div>
              
              <motion.div 
                className="p-5 rounded-xl glass-effect border border-gray-800"
                whileHover={{ y: -5, boxShadow: '0 0 20px rgba(157, 0, 255, 0.2)' }}
              >
                <Brain className="text-neon-purple mb-3" size={28} />
                <h3 className="text-xl font-display font-semibold mb-2">Intelligent</h3>
                <p className="text-gray-400">Powered by advanced AI to provide insightful spiritual guidance.</p>
              </motion.div>
              
              <motion.div 
                className="p-5 rounded-xl glass-effect border border-gray-800"
                whileHover={{ y: -5, boxShadow: '0 0 20px rgba(157, 0, 255, 0.2)' }}
              >
                <Zap className="text-neon-purple mb-3" size={28} />
                <h3 className="text-xl font-display font-semibold mb-2">Lightning Fast</h3>
                <p className="text-gray-400">Get immediate answers to your spiritual questions.</p>
              </motion.div>
            </div>

            <motion.div 
              className="hidden lg:block"
              whileHover={{ x: 5 }}
            >
              <a href="#auth" className="inline-flex items-center text-neon-purple hover:text-purple-400">
                <span className="mr-2">Get started now</span>
                <ChevronRight size={20} />
              </a>
            </motion.div>
          </motion.div>
        </div>

        <div id="auth" className="w-full lg:w-1/2 p-8 lg:p-16 flex items-center justify-center">
          <AuthForm onSuccess={onLogin} />
        </div>
      </main>

      {/* Footer */}
      <footer className="py-6 px-8 text-center text-gray-500 text-sm border-t border-gray-800">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <p>Shaktimaan GPT © {new Date().getFullYear()} | Powered by Gemini 2.0 Flash</p>
          <nav className="flex items-center gap-6">
            <Link to="/contact" className="text-gray-400 hover:text-white transition-colors">
              Contact
            </Link>
            <Link to="/terms" className="text-gray-400 hover:text-white transition-colors">
              Terms
            </Link>
            <Link to="/refund-policy" className="text-gray-400 hover:text-white transition-colors">
              Refund Policy
            </Link>
          </nav>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;