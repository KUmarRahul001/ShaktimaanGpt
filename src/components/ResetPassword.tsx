import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Eye, EyeOff, AlertCircle, Check, X, ShieldCheck } from 'lucide-react';
import IlluminatiLogo from './IlluminatiLogo';
import { supabase } from '../lib/supabase';

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

const ResetPassword: React.FC = () => {
  const navigate = useNavigate();
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [passwordFocused, setPasswordFocused] = useState(false);

  useEffect(() => {
    // Check if we have the recovery token in the hash URL
    const hashParams = new URLSearchParams(window.location.hash.replace('#', '?'));
    const accessToken = hashParams.get('access_token');
    const type = hashParams.get('type');
    const errorDescription = hashParams.get('error_description');

    // For supabase detectSessionInUrl, the session is usually set automatically
    // We just want to make sure the user doesn't get stuck if there's an error
    if (errorDescription) {
      setError(errorDescription.replace(/\+/g, ' '));
    }
  }, []);

  const validatePassword = () => {
    return passwordRequirements.every((req) => req.test(password));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (!validatePassword()) {
      setError('Please meet all password requirements');
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setIsLoading(true);

    try {
      const { error } = await supabase.auth.updateUser({
        password: password
      });

      if (error) throw error;

      setSuccess('Your password has been successfully updated.');

      // Redirect to login after 3 seconds
      setTimeout(() => {
        navigate('/');
      }, 3000);

    } catch (err: any) {
      console.error('Password reset error:', err);
      setError(err.message || 'An error occurred while updating your password');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-dark-bg flex items-center justify-center p-4 relative overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 z-0">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-900/20 rounded-full blur-[100px] animate-pulse-slow"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-900/20 rounded-full blur-[100px] animate-pulse-slow" style={{ animationDelay: '2s' }}></div>
      </div>

      <motion.div
        className="w-full max-w-md p-8 rounded-xl glass-effect z-10"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex flex-col items-center mb-8">
          <IlluminatiLogo size={80} />
          <h2 className="mt-6 text-3xl font-display font-bold neon-text">
            Update Password
          </h2>
          <p className="mt-2 text-gray-400 text-center">
            Please enter your new password below.
          </p>
        </div>

        {error && (
          <motion.div
            className="mb-4 p-3 bg-red-900/30 border border-red-800 rounded-lg flex items-center gap-2 text-red-300"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            transition={{ duration: 0.3 }}
          >
            <AlertCircle size={18} className="flex-shrink-0" />
            <span className="text-sm">{error}</span>
          </motion.div>
        )}

        {success ? (
          <motion.div
            className="mb-4 p-6 bg-green-900/20 border border-green-800/50 rounded-lg flex flex-col items-center justify-center text-center gap-4 text-green-400"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
          >
            <ShieldCheck size={48} className="text-green-500" />
            <div>
              <h3 className="text-lg font-bold mb-1">Password Updated</h3>
              <p className="text-sm text-green-300">{success}</p>
              <p className="text-xs text-gray-400 mt-2">Redirecting to sign in...</p>
            </div>
            <button
              onClick={() => navigate('/')}
              className="mt-2 px-6 py-2 bg-green-900/50 hover:bg-green-800/80 text-white rounded-lg transition-colors text-sm"
            >
              Go to Sign In Now
            </button>
          </motion.div>
        ) : (
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-1">
                New Password
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

            <div className="mb-4">
              <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-300 mb-1">
                Confirm New Password
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
                  Updating...
                </span>
              ) : (
                'Update Password'
              )}
            </button>
          </form>
        )}
      </motion.div>
    </div>
  );
};

export default ResetPassword;