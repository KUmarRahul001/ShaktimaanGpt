import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Flower, AlertCircle, CheckCircle2, Menu, X, Crown } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import ChatHistory from './ChatHistory';
import IlluminatiLogo from './IlluminatiLogo';
import ProfileMenu from './ProfileMenu';
import { generateResponse, verifyGeminiConnection, ChatMessage as ChatMessageType } from '../lib/gemini';
import { 
  getCurrentUser, 
  signOut, 
  saveChatHistory, 
  updateChatHistory, 
  getChatHistories, 
  getChatHistory,
  deleteChatHistory,
  getUserProfile,
  checkAIUsageLimit,
  incrementAIUsage,
  ChatHistory as ChatHistoryType,
  UserProfile
} from '../lib/supabase';

interface ChatInterfaceProps {
  onLogout: () => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ onLogout }) => {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<ChatMessageType[]>([
    {
      role: 'assistant',
      content: 'I am Shaktimaan GPT. I offer a spiritual connect. What would you like to discover?'
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState<'checking' | 'connected' | 'error'>('checking');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [chatHistories, setChatHistories] = useState<ChatHistoryType[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [user, setUser] = useState<any>(null);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Get current user and chat histories on mount
  useEffect(() => {
    const initializeUser = async () => {
      try {
        const currentUser = await getCurrentUser();
        setUser(currentUser);
        
        if (currentUser) {
          const histories = await getChatHistories(currentUser.id);
          setChatHistories(histories);
          
          // Fetch user profile
          const profile = await getUserProfile(currentUser.id);
          setUserProfile(profile);
        }
      } catch (error) {
        console.error('Error initializing user:', error);
      }
    };
    
    initializeUser();
  }, []);

  // Verify API connection on component mount
  useEffect(() => {
    const checkApiConnection = async () => {
      try {
        const isConnected = await verifyGeminiConnection();
        setApiStatus(isConnected ? 'connected' : 'error');
      } catch (error) {
        console.error('API connection check failed:', error);
        setApiStatus('error');
      }
    };
    
    checkApiConnection();
  }, []);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || apiStatus === 'error') return;
    
    const userMessage: ChatMessageType = {
      role: 'user',
      content
    };

    // Add user message to chat
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      console.log("[App] Sending message to Gemini API");

      // Check AI usage limits for free users before proceeding
      if (user && userProfile && !userProfile.is_pro) {
        const isAllowed = await checkAIUsageLimit(user.id);
        if (!isAllowed) {
          setIsLoading(false);
          setMessages(prev => [
            ...prev,
            {
              role: 'assistant',
              content: 'Daily limit reached. Upgrade to Pro for unlimited AI usage.'
            }
          ]);
          return;
        }
      }
      
      // Send all messages to get context-aware responses
      const response = await generateResponse([...messages, userMessage]);
      console.log("[App] Received response from Gemini API");
      
      // Increment AI usage
      if (user && userProfile && !userProfile.is_pro) {
        await incrementAIUsage(user.id);
      }

      // Add AI response to chat
      // FIX 1: explicitly type updatedMessages so 'assistant' is typed correctly
      const updatedMessages: ChatMessageType[] = [
        ...messages,
        userMessage,
        {
          role: 'assistant',
          content: response
        }
      ];
      
      setMessages(updatedMessages);
      
      // Save or update chat history if user is logged in
      if (user) {
        try {
          if (currentChatId) {
            // Update existing chat
            await updateChatHistory(currentChatId, updatedMessages);
          } else {
            // Create new chat
            const title = content.length > 30 ? `${content.substring(0, 30)}...` : content;
            const newChat = await saveChatHistory(user.id, updatedMessages, title);
            setCurrentChatId(newChat.id);
            setChatHistories(prev => [newChat, ...prev]);
          }
        } catch (error) {
          console.error('Error saving chat history:', error);
        }
      }
    } catch (error) {
      console.error('[App] Error in handleSendMessage:', error);
      // Add error message
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.'
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectChat = async (chatId: string) => {
    try {
      const chat = await getChatHistory(chatId);
      setMessages(chat.messages as ChatMessageType[]);
      setCurrentChatId(chatId);
      setSidebarOpen(false); // Close sidebar on mobile after selection
    } catch (error) {
      console.error('Error loading chat history:', error);
    }
  };

  const handleDeleteChat = async (chatId: string) => {
    try {
      await deleteChatHistory(chatId);
      setChatHistories(prev => prev.filter(chat => chat.id !== chatId));
      
      // If the deleted chat was the current one, start a new chat
      if (currentChatId === chatId) {
        handleNewChat();
      }
    } catch (error) {
      console.error('Error deleting chat history:', error);
    }
  };

  const handleNewChat = () => {
    setMessages([
      {
        role: 'assistant',
        content: 'I am Shaktimaan GPT. I offer a spiritual connect. What would you like to discover?'
      }
    ]);
    setCurrentChatId(null);
    setSidebarOpen(false); // Close sidebar on mobile
  };

  const handleLogout = async () => {
    try {
      await signOut();
      onLogout();
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };
  
  const handleProfileUpdate = (updatedProfile: UserProfile) => {
    setUserProfile(updatedProfile);
  };

  return (
    <div className="min-h-screen bg-dark-bg flex flex-col">
      <header className="bg-dark-surface border-b border-gray-800 shadow-sm">
        <div className="max-w-7xl mx-auto p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button 
              className="lg:hidden text-gray-400 hover:text-white"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <Menu size={24} />
            </button>
            <IlluminatiLogo size={36} />
            <h1 className="text-xl font-display font-semibold neon-white-text">Shaktimaan GPT</h1>
            <span className="hidden md:inline-block text-sm text-gray-500">A Spiritual Connect</span>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/pro')}
              className="flex items-center gap-2 px-4 py-2 bg-neon-purple hover:bg-purple-600 text-white rounded-lg transition-all"
            >
              <Crown size={18} />
              <span className="hidden sm:inline">Upgrade to Pro</span>
            </button>
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-400">API:</span>
              {apiStatus === 'checking' && (
                <span className="text-yellow-500 flex items-center gap-1">
                  <AlertCircle size={16} /> Checking...
                </span>
              )}
              {apiStatus === 'connected' && (
                <span className="text-neon-purple flex items-center gap-1">
                  <CheckCircle2 size={16} /> Connected
                </span>
              )}
              {apiStatus === 'error' && (
                <span className="text-red-500 flex items-center gap-1">
                  <AlertCircle size={16} /> Error
                </span>
              )}
            </div>
            <ProfileMenu 
              user={userProfile} 
              onLogout={handleLogout}
              onProfileUpdate={handleProfileUpdate}
            />
          </div>
        </div>
      </header>

      <div className="flex-1 flex">
        {/* Sidebar - Chat History */}
        <AnimatePresence>
          {sidebarOpen && (
            <motion.div 
              className="fixed inset-0 z-40 lg:hidden bg-black bg-opacity-50"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSidebarOpen(false)}
            >
              <motion.div 
                className="absolute top-0 left-0 bottom-0 w-72 bg-dark-surface border-r border-gray-800"
                initial={{ x: -280 }}
                animate={{ x: 0 }}
                exit={{ x: -280 }}
                transition={{ type: 'spring', damping: 25 }}
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex items-center justify-between p-4 border-b border-gray-800">
                  <h2 className="font-display font-medium neon-white-text">Chat History</h2>
                  <button 
                    onClick={() => setSidebarOpen(false)}
                    className="text-gray-500 hover:text-white"
                  >
                    <X size={20} />
                  </button>
                </div>
                <ChatHistory 
                  histories={chatHistories}
                  currentChatId={currentChatId}
                  onSelectChat={handleSelectChat}
                  onDeleteChat={handleDeleteChat}
                  onNewChat={handleNewChat}
                />
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Desktop Sidebar */}
        <div className="hidden lg:block w-72 border-r border-gray-800 bg-dark-surface">
          <div className="h-full">
            <div className="p-4 border-b border-gray-800">
              <h2 className="font-display font-medium neon-white-text">Chat History</h2>
            </div>
            <ChatHistory 
              histories={chatHistories}
              currentChatId={currentChatId}
              onSelectChat={handleSelectChat}
              onDeleteChat={handleDeleteChat}
              onNewChat={handleNewChat}
            />
          </div>
        </div>

        {/* Main Chat Area */}
        <main className="flex-1 flex flex-col">
          {apiStatus === 'error' ? (
            <div className="flex-1 flex items-center justify-center p-8">
              <div className="text-center bg-red-900/30 p-6 rounded-lg max-w-md border border-red-800">
                <AlertCircle size={48} className="text-red-500 mx-auto mb-4" />
                <h2 className="text-xl font-semibold text-red-400 mb-2">API Connection Error</h2>
                <p className="text-gray-300 mb-4">
                  Unable to connect to the Gemini API. Please check your API key and internet connection.
                </p>
                <button 
                  onClick={() => window.location.reload()}
                  className="px-4 py-2 bg-red-700 text-white rounded-md hover:bg-red-600 transition-colors"
                >
                  Retry Connection
                </button>
              </div>
            </div>
          ) : (
            <>
              <div className="flex-1 overflow-y-auto p-4 space-y-4 illuminati-pattern">
                <div className="space-y-px">
                  {messages.map((message, index) => (
                    <ChatMessage key={index} message={message} index={index} />
                  ))}
                  {isLoading && (
                    <motion.div 
                      className="flex gap-3 p-4"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.3 }}
                    >
                      <div className="flex-shrink-0">
                        <div className="w-8 h-8 rounded-full bg-purple-900/50 flex items-center justify-center">
                          <Flower size={18} className="text-neon-purple" />
                        </div>
                      </div>
                      <div className="flex-1">
                        <div className="font-medium mb-1 font-display">Shaktimaan GPT</div>
                        <div className="text-gray-400">
                          <div className="flex space-x-2">
                            <div className="h-2 w-2 rounded-full bg-neon-purple animate-pulse"></div>
                            <div className="h-2 w-2 rounded-full bg-neon-purple animate-pulse delay-150"></div>
                            <div className="h-2 w-2 rounded-full bg-neon-purple animate-pulse delay-300"></div>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </div>
                <div ref={messagesEndRef} />
              </div>
              <ChatInput 
                onSendMessage={handleSendMessage} 
                isLoading={isLoading} 
                // FIX 2: Check for 'checking' instead of 'error' so TS knows the logic overlaps correctly
                isDisabled={apiStatus === 'checking'} 
              />
            </>
          )}
        </main>
      </div>
    </div>
  );
};

export default ChatInterface;