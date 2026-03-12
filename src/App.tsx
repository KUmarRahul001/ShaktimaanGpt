import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import ChatInterface from './components/ChatInterface';
import ProFeatures from './components/ProFeatures';
import ContactUs from './components/ContactUs';
import TermsAndConditions from './components/TermsAndConditions';
import RefundPolicy from './components/RefundPolicy';
import PaymentStatus from './components/PaymentStatus';
import ResetPassword from './components/ResetPassword';
import { supabase, getCurrentUser } from './lib/supabase';
import { AuthChangeEvent, Session } from '@supabase/supabase-js';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const [authError, setAuthError] = useState<string | null>(null);

  useEffect(() => {
    // Check if we have a recovery token in the URL hash (from password reset email)
    // Sometimes Supabase redirects to root instead of the requested redirectTo URL
    const hash = window.location.hash;
    // We check window.location.pathname to avoid infinite redirects if we are already on /reset-password
    if (hash && hash.includes('type=recovery') && window.location.pathname !== '/reset-password') {
      // Redirect to the reset password page with the hash intact
      window.location.href = `/reset-password${hash}`;
      return;
    }

    // Check if user is already authenticated
    const checkAuth = async () => {
      try {
        const user = await getCurrentUser();
        setIsAuthenticated(!!user);
        setAuthError(null);
      } catch (error) {
        console.error('Auth check error:', error);
        setIsAuthenticated(false);
        setAuthError('Failed to connect to authentication service');
      }
    };

    checkAuth();

    // Set up auth state change listener
    const { data: authListener } = supabase.auth.onAuthStateChange(
      (event, session) => {
        if (event === 'PASSWORD_RECOVERY' && window.location.pathname !== '/reset-password') {
          // If the event fires before the hash check above, or during a session
          window.location.href = `/reset-password${window.location.hash}`;
          return;
        }
        setIsAuthenticated(!!session);
        if (session) setAuthError(null);
      }
    );

    return () => {
      authListener.subscription.unsubscribe();
    };
  }, []);

  // Show loading state while checking authentication
  if (isAuthenticated === null) {
    return (
      <div className="min-h-screen bg-dark-bg flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-neon-purple border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <Router>
      <Routes>
        <Route 
          path="/" 
          element={
            isAuthenticated ? (
              <ChatInterface onLogout={() => setIsAuthenticated(false)} />
            ) : (
              <LandingPage onLogin={() => setIsAuthenticated(true)} authError={authError} />
            )
          } 
        />
        <Route 
          path="/pro" 
          element={
            isAuthenticated ? (
              <ProFeatures />
            ) : (
              <Navigate to="/" replace />
            )
          } 
        />
        <Route path="/contact" element={<ContactUs />} />
        <Route path="/terms" element={<TermsAndConditions />} />
        <Route path="/refund-policy" element={<RefundPolicy />} />
        <Route path="/payment-status" element={<PaymentStatus />} />
        <Route path="/reset-password" element={<ResetPassword />} />
      </Routes>
    </Router>
  );
}

export default App;