import React, { useState, useRef, useEffect } from 'react';
import { User, Sparkles, Zap, Brain, Heart, Flower, Copy, Check, Volume2, VolumeX } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeHighlight from 'rehype-highlight';
import { motion } from 'framer-motion';
import { ChatMessage as ChatMessageType } from '../lib/gemini';

interface ChatMessageProps {
  message: ChatMessageType & { imageUrl?: string };
  index: number;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, index }) => {
  const isUser = message.role === 'user';
  const [copiedCodes, setCopiedCodes] = useState<{ [key: number]: boolean }>({});
  const [isReading, setIsReading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const isMounted = useRef(true);
  
  const spiritualIcons = [Flower, Sparkles, Zap, Brain, Heart];
  const IconComponent = isUser ? User : spiritualIcons[index % spiritualIcons.length];

  useEffect(() => {
    return () => {
      isMounted.current = false;
      if (utteranceRef.current) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  const handleCopyCode = async (code: string, codeIndex: number) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedCodes(prev => ({ ...prev, [codeIndex]: true }));
      setTimeout(() => {
        setCopiedCodes(prev => ({ ...prev, [codeIndex]: false }));
      }, 2000);
    } catch (error) {
      console.error('Failed to copy code:', error);
    }
  };

  const cleanTextForSpeech = (text: string): string => {
    return text
      .replace(/```[\s\S]*?```/g, 'code block omitted') // Replace code blocks with placeholder
      .replace(/`.*?`/g, '') // Remove inline code
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Replace links with just text
      .replace(/[#*_~`]/g, '') // Remove markdown symbols
      .replace(/\n\n/g, '. ') // Replace double newlines with period and space
      .replace(/\n/g, ' ') // Replace single newlines with space
      .trim();
  };

  const handleReadAloud = () => {
    if (!window.speechSynthesis) {
      setError('Speech synthesis is not supported in your browser');
      return;
    }

    if (isReading) {
      window.speechSynthesis.cancel();
      setIsReading(false);
      return;
    }

    try {
      if (!isUser && message.content) {
        const cleanText = cleanTextForSpeech(message.content);
        
        if (cleanText.trim().length === 0) {
          setError('No readable text found in the message');
          return;
        }

        utteranceRef.current = new SpeechSynthesisUtterance(cleanText);
        
        // Configure speech settings
        utteranceRef.current.rate = 1;
        utteranceRef.current.pitch = 1;
        utteranceRef.current.volume = 1;
        
        // Handle speech events
        utteranceRef.current.onend = () => {
          if (isMounted.current) {
            setIsReading(false);
            setError(null);
          }
        };

        utteranceRef.current.onerror = (event) => {
          if (isMounted.current) {
            console.error('Speech synthesis error:', event);
            setIsReading(false);
            setError('An error occurred while reading the text');
          }
        };

        setIsReading(true);
        setError(null);
        window.speechSynthesis.cancel(); // Cancel any ongoing speech
        window.speechSynthesis.speak(utteranceRef.current);
      }
    } catch (error) {
      console.error('Speech synthesis error:', error);
      setError('Failed to initialize speech synthesis');
      setIsReading(false);
    }
  };

  return (
    <motion.div 
      className={`flex gap-3 p-4 ${isUser ? 'bg-dark-surface-2/50' : 'bg-dark-surface/70'}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: Math.min(index * 0.1, 1) }}
    >
      <div className="flex-shrink-0">
        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
          isUser ? 'bg-blue-900/50 text-blue-400' : 'bg-purple-900/50 text-neon-purple'
        }`}>
          <IconComponent size={18} />
        </div>
      </div>
      <div className="flex-1">
        <div className="font-medium mb-1 font-display">
          {isUser ? 'You' : 'Shaktimaan GPT'}
        </div>
        {message.imageUrl && (
          <div className="mb-3">
            <img 
              src={message.imageUrl} 
              alt="Uploaded content"
              className="max-w-sm rounded-lg border border-gray-800"
            />
          </div>
        )}
        {isUser ? (
          <div className="text-gray-300 whitespace-pre-wrap">{message.content}</div>
        ) : (
          <div className="markdown-content">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeRaw, rehypeHighlight]}
              components={{
                h1: ({ node, ...props }) => <h1 className="text-2xl font-bold my-4 text-neon-purple font-display" {...props} />,
                h2: ({ node, ...props }) => <h2 className="text-xl font-bold my-3 text-neon-purple font-display" {...props} />,
                h3: ({ node, ...props }) => <h3 className="text-lg font-bold my-2 text-neon-purple font-display" {...props} />,
                h4: ({ node, ...props }) => <h4 className="text-base font-bold my-2 text-neon-purple font-display" {...props} />,
                p: ({ node, ...props }) => <p className="my-2 text-gray-200" {...props} />,
                ul: ({ node, ...props }) => <ul className="list-disc pl-6 my-2 text-gray-200" {...props} />,
                ol: ({ node, ...props }) => <ol className="list-decimal pl-6 my-2 text-gray-200" {...props} />,
                li: ({ node, ...props }) => <li className="my-1" {...props} />,
                a: ({ node, ...props }) => <a className="text-neon-purple hover:underline" {...props} />,
                blockquote: ({ node, ...props }) => <blockquote className="border-l-4 border-neon-purple pl-4 italic my-2 text-gray-300" {...props} />,
                code: ({ node, inline, className, children, ...props }) => {
                  const match = /language-(\w+)/.exec(className || '');
                  const codeString = String(children).replace(/\n$/, '');
                  const codeIndex = `${index}-${codeString.slice(0, 32)}`;
                  
                  if (inline) {
                    return (
                      <code className="bg-gray-800 px-1 py-0.5 rounded text-sm font-mono text-neon-purple" {...props}>
                        {children}
                      </code>
                    );
                  }

                  return (
                    <div className="relative group">
                      <button
                        onClick={() => handleCopyCode(codeString, index)}
                        className="absolute right-2 top-2 p-2 bg-gray-800/50 rounded-lg opacity-0 group-hover:opacity-100 hover:bg-gray-700/50 transition-all duration-200"
                        title="Copy code"
                      >
                        {copiedCodes[index] ? (
                          <Check size={16} className="text-green-400" />
                        ) : (
                          <Copy size={16} className="text-gray-400" />
                        )}
                      </button>
                      <pre className="bg-gray-900 text-gray-100 p-4 rounded-md my-3 overflow-x-auto border border-gray-800">
                        <code className={`${className || ''} text-sm font-mono`} {...props}>
                          {children}
                        </code>
                      </pre>
                    </div>
                  );
                },
                table: ({ node, ...props }) => <table className="border-collapse border border-gray-700 my-4 w-full" {...props} />,
                thead: ({ node, ...props }) => <thead className="bg-gray-800" {...props} />,
                tbody: ({ node, ...props }) => <tbody {...props} />,
                tr: ({ node, ...props }) => <tr className="border-b border-gray-700" {...props} />,
                th: ({ node, ...props }) => <th className="border border-gray-700 px-4 py-2 text-left" {...props} />,
                td: ({ node, ...props }) => <td className="border border-gray-700 px-4 py-2" {...props} />,
                img: ({ node, ...props }) => <img className="max-w-full h-auto my-4 rounded" {...props} />,
                hr: ({ node, ...props }) => <hr className="my-4 border-t border-gray-700" {...props} />,
              }}
            >
              {message.content}
            </ReactMarkdown>
            {!isUser && (
              <div className="mt-4 flex flex-col gap-2">
                <button
                  onClick={handleReadAloud}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors ${
                    isReading 
                      ? 'bg-neon-purple text-white' 
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white'
                  }`}
                >
                  {isReading ? <VolumeX size={16} /> : <Volume2 size={16} />}
                  <span>{isReading ? 'Stop Reading' : 'Read Aloud'}</span>
                </button>
                {error && (
                  <p className="text-sm text-red-400">{error}</p>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default ChatMessage;