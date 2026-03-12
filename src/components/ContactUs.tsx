import React from 'react';
import { motion } from 'framer-motion';
import { Mail, Phone, MapPin, ArrowLeft } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const ContactUs: React.FC = () => {
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
          <h1 className="text-3xl font-display font-bold neon-purple-text">Contact Us</h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-12">
        <div className="grid md:grid-cols-2 gap-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="text-2xl font-display font-bold mb-6">Get in Touch</h2>
            <p className="text-gray-400 mb-8">
              Have questions about Shaktimaan GPT? We're here to help. Fill out the form
              below and we'll get back to you as soon as possible.
            </p>

            <form className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Name
                </label>
                <input
                  type="text"
                  className="input-dark w-full"
                  placeholder="Your name"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Email
                </label>
                <input
                  type="email"
                  className="input-dark w-full"
                  placeholder="your@email.com"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Subject
                </label>
                <input
                  type="text"
                  className="input-dark w-full"
                  placeholder="How can we help?"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Message
                </label>
                <textarea
                  rows={5}
                  className="input-dark w-full"
                  placeholder="Tell us more about your inquiry..."
                />
              </div>

              <button
                type="submit"
                className="w-full py-3 bg-neon-purple hover:bg-purple-600 text-white font-medium rounded-lg transition-all"
              >
                Send Message
              </button>
            </form>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="space-y-8"
          >
            <div>
              <h2 className="text-2xl font-display font-bold mb-6">Contact Information</h2>
              <div className="space-y-4">
                <div className="flex items-start gap-4">
                  <Mail className="text-neon-purple mt-1" size={20} />
                  <div>
                    <h3 className="font-medium">Email</h3>
                    <p className="text-gray-400">support@shaktimaangpt.com</p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <Phone className="text-neon-purple mt-1" size={20} />
                  <div>
                    <h3 className="font-medium">Phone</h3>
                    <p className="text-gray-400">+1 (555) 123-4567</p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <MapPin className="text-neon-purple mt-1" size={20} />
                  <div>
                    <h3 className="font-medium">Address</h3>
                    <p className="text-gray-400">
                      123 Spiritual Way<br />
                      Enlightenment Valley, CA 94123<br />
                      United States
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h2 className="text-2xl font-display font-bold mb-6">Business Hours</h2>
              <div className="space-y-2 text-gray-400">
                <p>Monday - Friday: 9:00 AM - 6:00 PM (PST)</p>
                <p>Saturday: 10:00 AM - 4:00 PM (PST)</p>
                <p>Sunday: Closed</p>
              </div>
            </div>

            <div className="p-6 bg-dark-surface-2 rounded-lg border border-gray-800">
              <h3 className="font-medium mb-2">Support Response Time</h3>
              <p className="text-gray-400">
                We aim to respond to all inquiries within 24 hours during business hours.
                For urgent matters, please contact us by phone.
              </p>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
};

export default ContactUs;