import React, { useState, useRef } from 'react';
import { Send, AlertCircle, Image as ImageIcon, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { uploadChatImage } from '../lib/supabase';

interface ChatInputProps {
  onSendMessage: (message: string, imageUrl?: string) => void;
  isLoading: boolean;
  isDisabled?: boolean;
  isPro?: boolean;
  userId?: string;
}

const ChatInput: React.FC<ChatInputProps> = ({ 
  onSendMessage, 
  isLoading, 
  isDisabled = false,
  isPro = false,
  userId
}) => {
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if ((input.trim() || selectedImage) && !isLoading && !isDisabled && !isUploading) {
      try {
        let imageUrl: string | undefined;
        
        if (selectedImage && userId) {
          setIsUploading(true);
          imageUrl = await uploadChatImage(userId, selectedImage);
        }

        onSendMessage(input, imageUrl);
        setInput('');
        setSelectedImage(null);
        setPreviewUrl(null);
      } catch (error) {
        console.error('Error sending message:', error);
        setUploadError('Failed to upload image. Please try again.');
      } finally {
        setIsUploading(false);
      }
    }
  };

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.size > 5 * 1024 * 1024) { // 5MB limit
        setUploadError('Image size must be less than 5MB');
        return;
      }
      
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setUploadError(null);
    }
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <motion.form 
      onSubmit={handleSubmit} 
      className="border-t border-gray-800 p-4 glass-effect"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {isDisabled && (
        <div className="mb-3 p-2 bg-red-900/30 border border-red-800 rounded-md flex items-center gap-2 text-red-300 text-sm">
          <AlertCircle size={16} />
          <span>API connection error. Chat functionality is disabled.</span>
        </div>
      )}

      {uploadError && (
        <div className="mb-3 p-2 bg-red-900/30 border border-red-800 rounded-md flex items-center gap-2 text-red-300 text-sm">
          <AlertCircle size={16} />
          <span>{uploadError}</span>
        </div>
      )}

      <AnimatePresence>
        {previewUrl && (
          <motion.div 
            className="mb-3 relative"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <div className="relative inline-block">
              <img 
                src={previewUrl} 
                alt="Preview" 
                className="max-h-32 rounded-lg border border-gray-800"
              />
              <button
                type="button"
                onClick={handleRemoveImage}
                className="absolute -top-2 -right-2 p-1 bg-red-500 rounded-full text-white hover:bg-red-600 transition-colors"
              >
                <X size={14} />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex items-center gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={isDisabled ? "API connection error" : "Ask ShaktimaanGpt anything..."}
          className={`flex-1 p-3 rounded-lg ${
            isDisabled 
              ? "border-red-800 bg-red-900/30 text-red-300 placeholder-red-700" 
              : "bg-dark-surface-2 border border-gray-800 text-white focus:outline-none focus:ring-2 focus:ring-neon-purple focus:border-transparent"
          }`}
          disabled={isLoading || isDisabled || isUploading}
        />

        {isPro && (
          <>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageSelect}
              accept="image/*"
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading || isDisabled || isUploading}
              className={`p-3 rounded-lg ${
                isLoading || isDisabled || isUploading
                  ? 'bg-gray-800 text-gray-600'
                  : 'bg-dark-surface-2 border border-gray-800 text-white hover:bg-gray-800'
              } transition-colors`}
            >
              <ImageIcon size={20} />
            </button>
          </>
        )}

        <motion.button
          type="submit"
          disabled={(!input.trim() && !selectedImage) || isLoading || isDisabled || isUploading}
          className={`p-3 rounded-lg ${
            (!input.trim() && !selectedImage) || isLoading || isDisabled || isUploading
              ? 'bg-gray-800 text-gray-600'
              : 'bg-neon-purple text-white hover:bg-purple-600'
          } transition-colors`}
          whileTap={{ scale: 0.95 }}
        >
          {isUploading ? (
            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
          ) : (
            <Send size={20} />
          )}
        </motion.button>
      </div>
    </motion.form>
  );
};

export default ChatInput;