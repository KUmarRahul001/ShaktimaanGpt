import React from 'react';
import { motion } from 'framer-motion';
import { MessageSquare, Trash2 } from 'lucide-react';
import { ChatHistory as ChatHistoryType } from '../lib/supabase';

interface ChatHistoryProps {
  histories: ChatHistoryType[];
  currentChatId: string | null;
  onSelectChat: (chatId: string) => void;
  onDeleteChat: (chatId: string) => void;
  onNewChat: () => void;
}

const ChatHistory: React.FC<ChatHistoryProps> = ({ 
  histories, 
  currentChatId, 
  onSelectChat, 
  onDeleteChat,
  onNewChat
}) => {
  return (
    <div className="h-full flex flex-col">
      <div className="p-4">
        <button 
          onClick={onNewChat}
          className="w-full py-2 px-4 bg-neon-purple hover:bg-purple-600 text-white font-medium rounded-lg transition-all"
        >
          New Chat
        </button>
      </div>
      
      <div className="flex-1 overflow-y-auto p-2">
        {histories.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <MessageSquare className="mx-auto mb-2 opacity-50" size={24} />
            <p>No chat history yet</p>
          </div>
        ) : (
          <ul className="space-y-1">
            {histories.map((chat) => (
              <motion.li 
                key={chat.id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2 }}
              >
                <div 
                  className={`flex items-center justify-between p-3 rounded-lg cursor-pointer group ${
                    currentChatId === chat.id 
                      ? 'bg-dark-surface-2 border border-neon-purple/30' 
                      : 'hover:bg-dark-surface-2 border border-transparent'
                  }`}
                >
                  <div 
                    className="flex-1 truncate pr-2"
                    onClick={() => onSelectChat(chat.id)}
                  >
                    <p className="truncate text-sm font-medium">
                      {chat.title}
                    </p>
                    <p className="text-xs text-gray-500 truncate">
                      {new Date(chat.updated_at).toLocaleDateString()}
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteChat(chat.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-500 transition-opacity"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </motion.li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default ChatHistory;