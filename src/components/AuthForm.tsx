import React, { useState, useEffect } from 'react';
import { Eye, EyeOff, AlertCircle, Mail, Check, X, Phone } from 'lucide-react';
import { motion } from 'framer-motion';
import IlluminatiLogo from './IlluminatiLogo';
import { signIn, signUp, resetPassword } from '../lib/supabase';

interface AuthFormProps {
  onSuccess: () => void;
}

interface PasswordRequirement {
  label: string;
  test: (password: string) => boolean;
}

const passwordRequirements: PasswordRequirement[] = [
  {
    label: 'At least 8 characters long',
    test: (password) => password.length >= 8,
  },
  {
    label: 'Contains uppercase letter (A-Z)',
    test: (password) => /[A-Z]/.test(password),
  },
  {
    label: 'Contains lowercase letter (a-z)',
    test: (password) => /[a-z]/.test(password),
  },
  {
    label: 'Contains number (0-9)',
    test: (password) => /[0-9]/.test(password),
  },
  {
    label: 'Contains special character (!@#$%^&*()_+-=)',
    test: (password) => /[!@#$%^&*()_+\-=]/.test(password),
  },
];

const AuthForm: React.FC<AuthFormProps> = ({ onSuccess }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [isForgotPassword, setIsForgotPassword] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [passwordFocused, setPasswordFocused] = useState(false);

  const validatePassword = () => {
    return passwordRequirements.every((req) => req.test(password));
  };

  const validatePhoneNumber = (phone: string): boolean => {
    const phoneRegex = /^\+[1-9]\d{1,14}$/;
    return phoneRegex.test(phone);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (!isLogin && !isForgotPassword) {
      // Validate password requirements for signup
      if (!validatePassword()) {
        setError('Please meet all password requirements');
        return;
      }

      // Check if passwords match
      if (password !== confirmPassword) {
        setError('Passwords do not match');
        return;
      }

      // Validate phone number for signup
      if (!phoneNumber) {
        setError('Phone number is required');
        return;
      }

      if (!validatePhoneNumber(phoneNumber)) {
        setError('Please enter a valid phone number with country code (e.g., +1234567890)');
        return;
      }
    }

    setIsLoading(true);

    try {
      if (isForgotPassword) {
        await resetPassword(email);
        setSuccess('Password reset instructions have been sent to your email.');
        setIsLoading(false);
        return;
      }

      if (isLogin) {
        await signIn(email, password);
      } else {
        await signUp(email, password, phoneNumber);
        // If signup is successful, automatically sign them in
        await signIn(email, password);
      }
      onSuccess();
    } catch (err: any) {
      console.error('Authentication error:', err);
      
      // More user-friendly error messages
      if (err.message?.includes('Failed to fetch')) {
        setError('Connection to authentication service failed. Please check your internet connection and try again.');
      } else if (err.message?.includes('Invalid login credentials')) {
        setError('Invalid email or password. Please try again.');
      } else if (err.message?.includes('User already registered')) {
        setError('This email is already registered. Please sign in instead.');
        setIsLogin(true); // Switch to login form automatically
      } else if (err.message?.includes('Email not confirmed')) {
        setError('Please confirm your email before signing in.');
      } else if (err.message?.includes('user_already_exists')) {
        setError('This email is already registered. Please sign in instead.');
        setIsLogin(true); // Switch to login form automatically
      } else {
        setError(err.message || 'An error occurred during authentication');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSwitchMode = () => {
    setError('');
    setSuccess('');
    setIsForgotPassword(false);
    setIsLogin(!isLogin);
    setPassword('');
    setConfirmPassword('');
    setPhoneNumber('');
  };

  const handleForgotPassword = () => {
    setError('');
    setSuccess('');
    setIsForgotPassword(true);
  };

  return (
    <motion.div 
      className="w-full max-w-md p-8 rounded-xl glass-effect"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="flex flex-col items-center mb-8">
        <IlluminatiLogo size={80} />
        <h2 className="mt-6 text-3xl font-display font-bold neon-text">
          {isForgotPassword ? 'Reset Password' : isLogin ? 'Welcome Back' : 'Join Shaktimaan GPT'}
        </h2>
        <p className="mt-2 text-gray-400">
          {isForgotPassword 
            ? 'Enter your email to receive reset instructions' 
            : isLogin 
              ? 'Access the spiritual wisdom' 
              : 'Discover spiritual insights'}
        </p>
      </div>

      {error && (
        <motion.div 
          className="mb-4 p-3 bg-red-900/30 border border-red-800 rounded-lg flex items-center gap-2 text-red-300"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          transition={{ duration: 0.3 }}
        >
          <AlertCircle size={18} />
          <span>{error}</span>
        </motion.div>
      )}

      {success && (
        <motion.div 
          className="mb-4 p-3 bg-green-900/30 border border-green-800 rounded-lg flex items-center gap-2 text-green-300"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          transition={{ duration: 0.3 }}
        >
          <Mail size={18} />
          <span>{success}</span>
        </motion.div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-1">
            Email
          </label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="input-dark w-full"
            placeholder="your@email.com"
            required
          />
        </div>

        {!isForgotPassword && (
          <>
            <div className="mb-4">
              <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-1">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onFocus={() => setPasswordFocused(true)}
                  onBlur={() => setPasswordFocused(false)}
                  className="input-dark w-full pr-10"
                  placeholder="••••••••"
                  required
                  minLength={8}
                />
                <button
                  type="button"
                  className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-white"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            {!isLogin && (
              <>
                <div className="mb-4">
                  <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-300 mb-1">
                    Confirm Password
                  </label>
                  <div className="relative">
                    <input
                      id="confirmPassword"
                      type={showConfirmPassword ? 'text' : 'password'}
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      className="input-dark w-full pr-10"
                      placeholder="••••••••"
                      required
                      minLength={8}
                    />
                    <button
                      type="button"
                      className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-white"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    >
                      {showConfirmPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                    </button>
                  </div>
                  {confirmPassword && password !== confirmPassword && (
                    <p className="mt-1 text-sm text-red-400">Passwords do not match</p>
                  )}
                </div>

                <div className="mb-4">
                  <label htmlFor="phoneNumber" className="block text-sm font-medium text-gray-300 mb-1">
                    Phone Number *
                  </label>
                  <div className="relative">
                    <Phone className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
                    <input
                      id="phoneNumber"
                      type="tel"
                      value={phoneNumber}
                      onChange={(e) => setPhoneNumber(e.target.value)}
                      className="input-dark w-full pl-10"
                      placeholder="+1234567890"
                      required
                    />
                  </div>
                  <p className="mt-1 text-xs text-gray-400">
                    Enter your phone number with country code (e.g., +1234567890)
                  </p>
                </div>

                <motion.div
                  className="mb-6 p-4 bg-gray-900/50 rounded-lg"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ 
                    opacity: passwordFocused || password ? 1 : 0,
                    height: passwordFocused || password ? 'auto' : 0
                  }}
                  transition={{ duration: 0.3 }}
                >
                  <h3 className="text-sm font-medium text-gray-300 mb-2">Password Requirements:</h3>
                  <ul className="space-y-2">
                    {passwordRequirements.map((req, index) => (
                      <li 
                        key={index}
                        className="flex items-center text-sm"
                      >
                        {req.test(password) ? (
                          <Check size={16} className="text-green-500 mr-2" />
                        ) : (
                          <X size={16} className="text-red-500 mr-2" />
                        )}
                        <span className={req.test(password) ? 'text-green-400' : 'text-gray-400'}>
                          {req.label}
                        </span>
                      </li>
                    ))}
                  </ul>
                </motion.div>
              </>
            )}
          </>
        )}

        <button
          type="submit"
          disabled={isLoading}
          className="w-full py-3 px-4 bg-neon-purple hover:bg-purple-600 text-white font-medium rounded-lg transition-all flex items-center justify-center"
        >
          {isLoading ? (
            <span className="flex items-center">
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing...
            </span>
          ) : (
            isForgotPassword ? 'Send Reset Instructions' : isLogin ? 'Sign In' : 'Create Account'
          )}
        </button>

        <div className="mt-4 text-center space-y-2">
          {isLogin && !isForgotPassword && (
            <button
              type="button"
              className="text-neon-purple hover:text-purple-400 text-sm"
              onClick={handleForgotPassword}
            >
              Forgot your password?
            </button>
          )}
          
          <button
            type="button"
            className="text-neon-purple hover:text-purple-400 text-sm block w-full"
            onClick={handleSwitchMode}
          >
            {isForgotPassword 
              ? "Back to sign in"
              : isLogin 
                ? "Don't have an account? Sign up" 
                : "Already have an account? Sign in"}
          </button>
        </div>
      </form>
    </motion.div>
  );
};

export default AuthForm;