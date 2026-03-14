import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { User, LogOut, Trash2, Settings, Camera, X, Phone, AlertCircle, FileText, Download } from 'lucide-react';
import { UserProfile, updateUserProfile, uploadProfilePicture, deleteUserAccount } from '../lib/supabase';
import { jsPDF } from 'jspdf';

interface ProfileMenuProps {
  user: UserProfile | null;
  onLogout: () => void;
  onProfileUpdate: (profile: UserProfile) => void;
}

const ProfileMenu: React.FC<ProfileMenuProps> = ({ user, onLogout, onProfileUpdate }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [displayName, setDisplayName] = useState(user?.display_name || '');
  const [phoneNumber, setPhoneNumber] = useState(user?.phone_number || '');
  const [providerType, setProviderType] = useState(user?.provider_type || '');
  const [isUploading, setIsUploading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [phoneError, setPhoneError] = useState<string | null>(null);
  const [showInvoices, setShowInvoices] = useState(false);
  
  const menuRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  useEffect(() => {
    if (user) {
      setDisplayName(user.display_name || '');
      setPhoneNumber(user.phone_number || '');
      setProviderType(user.provider_type || '');
    }
  }, [user]);
  
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const validatePhoneNumber = (phone: string): boolean => {
    const phoneRegex = /^\+[1-9]\d{1,14}$/;
    return phoneRegex.test(phone);
  };
  
  const handleSaveProfile = async () => {
    if (!user) return;
    
    try {
      setPhoneError(null);

      if (!phoneNumber) {
        setPhoneError('Phone number is required');
        return;
      }

      if (!validatePhoneNumber(phoneNumber)) {
        setPhoneError('Please enter a valid phone number with country code (e.g., +1234567890)');
        return;
      }

      const updatedProfile = await updateUserProfile(user.id, {
        display_name: displayName,
        phone_number: phoneNumber,
        provider_type: providerType
      });
      
      onProfileUpdate(updatedProfile);
      setIsEditMode(false);
    } catch (error) {
      console.error('Error updating profile:', error);
      setPhoneError('Failed to update profile. Please try again.');
    }
  };
  
  const handleProfilePictureUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || !e.target.files[0] || !user) return;
    
    const file = e.target.files[0];
    setIsUploading(true);
    
    try {
      const avatarUrl = await uploadProfilePicture(user.id, file);
      onProfileUpdate({ ...user, avatar_url: avatarUrl });
    } catch (error) {
      console.error('Error uploading profile picture:', error);
    } finally {
      setIsUploading(false);
    }
  };
  
  const handleDeleteAccount = async () => {
    if (!user) return;
    
    setIsDeleting(true);
    
    try {
      await deleteUserAccount(user.id);
      onLogout();
    } catch (error) {
      console.error('Error deleting account:', error);
      setIsDeleting(false);
      setConfirmDelete(false);
    }
  };

  const generateInvoice = (paymentId: string, amount: number) => {
    const doc = new jsPDF();
    
    // Add company logo/header
    doc.setFontSize(20);
    doc.text('Shaktimaan GPT', 105, 20, { align: 'center' });
    
    // Add invoice details
    doc.setFontSize(12);
    doc.text('INVOICE', 20, 40);
    doc.text(`Invoice No: ${paymentId}`, 20, 50);
    doc.text(`Date: ${new Date().toLocaleDateString()}`, 20, 60);
    
    // Add customer details
    doc.text('Bill To:', 20, 80);
    doc.text(user?.display_name || 'User', 20, 90);
    doc.text(user?.email || '', 20, 100);
    
    // Add item details
    doc.text('Description', 20, 120);
    doc.text('Amount', 150, 120);
    doc.line(20, 125, 190, 125);
    doc.text('Pro Membership (1 month)', 20, 135);
    doc.text(`₹${amount.toFixed(2)}`, 150, 135);
    
    // Add total
    doc.line(20, 150, 190, 150);
    doc.text('Total:', 130, 160);
    doc.text(`₹${amount.toFixed(2)}`, 150, 160);
    
    // Add footer
    doc.setFontSize(10);
    doc.text('Thank you for your business!', 105, 200, { align: 'center' });
    
    // Save the PDF
    doc.save(`invoice-${paymentId}.pdf`);
  };

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-10 h-10 rounded-full flex items-center justify-center overflow-hidden border-2 border-neon-purple hover:border-purple-400 transition-all"
      >
        {user?.avatar_url ? (
          <img 
            src={user.avatar_url} 
            alt="Profile" 
            className="w-full h-full object-cover"
          />
        ) : (
          <User className="text-white" size={20} />
        )}
      </button>
      
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="absolute right-0 mt-2 w-72 rounded-lg glass-effect shadow-lg z-50"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            <div className="p-4 border-b border-gray-800">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <div className="w-16 h-16 rounded-full overflow-hidden border-2 border-neon-purple">
                    {user?.avatar_url ? (
                      <img 
                        src={user.avatar_url} 
                        alt="Profile" 
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full bg-gray-800 flex items-center justify-center">
                        <User className="text-gray-400" size={32} />
                      </div>
                    )}
                  </div>
                  
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleProfilePictureUpload}
                    className="hidden"
                    accept="image/*"
                  />
                  
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                    className="absolute bottom-0 right-0 w-8 h-8 rounded-full bg-neon-purple flex items-center justify-center"
                  >
                    {isUploading ? (
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    ) : (
                      <Camera size={14} className="text-white" />
                    )}
                  </button>
                </div>
                
                <div className="flex-1">
                  <h3 className="font-medium text-white">
                    {user?.display_name || user?.email?.split('@')[0] || 'User'}
                  </h3>
                  <p className="text-sm text-gray-400 truncate">{user?.email}</p>
                </div>
              </div>
            </div>
            
            <div className="p-4">
              {isEditMode ? (
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-1">
                      Display Name
                    </label>
                    <input
                      type="text"
                      value={displayName}
                      onChange={(e) => setDisplayName(e.target.value)}
                      className="input-dark w-full text-sm"
                      placeholder="Your display name"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-1">
                      Phone Number *
                    </label>
                    <div className="relative">
                      <Phone className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
                      <input
                        type="tel"
                        value={phoneNumber}
                        onChange={(e) => setPhoneNumber(e.target.value)}
                        className="input-dark w-full text-sm pl-10"
                        placeholder="+1234567890"
                        required
                      />
                    </div>
                    {phoneError && (
                      <div className="mt-1 text-sm text-red-400 flex items-center gap-1">
                        <AlertCircle size={14} />
                        <span>{phoneError}</span>
                      </div>
                    )}
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-1">
                      Provider Type
                    </label>
                    <select
                      value={providerType}
                      onChange={(e) => setProviderType(e.target.value)}
                      className="input-dark w-full text-sm"
                    >
                      <option value="">Select provider</option>
                      <option value="email">Email</option>
                      <option value="google">Google</option>
                      <option value="facebook">Facebook</option>
                      <option value="twitter">Twitter</option>
                      <option value="other">Other</option>
                    </select>
                  </div>
                  
                  <div className="flex gap-2 pt-2">
                    <button
                      onClick={() => setIsEditMode(false)}
                      className="flex-1 py-2 px-3 bg-gray-800 text-white text-sm rounded-lg hover:bg-gray-700"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleSaveProfile}
                      className="flex-1 py-2 px-3 bg-neon-purple text-white text-sm rounded-lg hover:bg-purple-600"
                    >
                      Save
                    </button>
                  </div>
                </div>
              ) : showInvoices ? (
                <div className="space-y-4">
                  <h3 className="font-medium text-white mb-2">Your Invoices</h3>
                  <div className="space-y-2">
                    {/* Example invoice - replace with actual invoice data */}
                    <div className="flex items-center justify-between p-2 bg-gray-800 rounded-lg">
                      <div>
                        <p className="text-sm font-medium">Pro Membership</p>
                        <p className="text-xs text-gray-400">March 19, 2024</p>
                      </div>
                      <button
                        onClick={() => generateInvoice('INV_123', 120)}
                        className="p-2 text-neon-purple hover:text-purple-400"
                      >
                        <Download size={16} />
                      </button>
                    </div>
                  </div>
                  <button
                    onClick={() => setShowInvoices(false)}
                    className="w-full py-2 px-3 bg-gray-800 text-white text-sm rounded-lg hover:bg-gray-700"
                  >
                    Back
                  </button>
                </div>
              ) : confirmDelete ? (
                <div className="space-y-3">
                  <div className="p-3 bg-red-900/30 border border-red-800 rounded-lg">
                    <h4 className="font-medium text-red-300 mb-1">Delete Account?</h4>
                    <p className="text-sm text-gray-300">
                      This action cannot be undone. All your data will be permanently deleted.
                    </p>
                  </div>
                  
                  <div className="flex gap-2 pt-2">
                    <button
                      onClick={() => setConfirmDelete(false)}
                      className="flex-1 py-2 px-3 bg-gray-800 text-white text-sm rounded-lg hover:bg-gray-700"
                      disabled={isDeleting}
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleDeleteAccount}
                      className="flex-1 py-2 px-3 bg-red-600 text-white text-sm rounded-lg hover:bg-red-700"
                      disabled={isDeleting}
                    >
                      {isDeleting ? (
                        <span className="flex items-center justify-center">
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                          Deleting...
                        </span>
                      ) : (
                        'Confirm Delete'
                      )}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="space-y-2">
                  <button
                    onClick={() => setIsEditMode(true)}
                    className="w-full py-2 px-3 flex items-center gap-2 text-white hover:bg-gray-800 rounded-lg transition-colors"
                  >
                    <Settings size={18} />
                    <span>Edit Profile</span>
                  </button>

                  <button
                    onClick={() => setShowInvoices(true)}
                    className="w-full py-2 px-3 flex items-center gap-2 text-white hover:bg-gray-800 rounded-lg transition-colors"
                  >
                    <FileText size={18} />
                    <span>View Invoices</span>
                  </button>
                  
                  <button
                    onClick={onLogout}
                    className="w-full py-2 px-3 flex items-center gap-2 text-white hover:bg-gray-800 rounded-lg transition-colors"
                  >
                    <LogOut size={18} />
                    <span>Logout</span>
                  </button>
                  
                  <button
                    onClick={() => setConfirmDelete(true)}
                    className="w-full py-2 px-3 flex items-center gap-2 text-red-400 hover:bg-red-900/30 rounded-lg transition-colors"
                  >
                    <Trash2 size={18} />
                    <span>Delete Account</span>
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ProfileMenu;