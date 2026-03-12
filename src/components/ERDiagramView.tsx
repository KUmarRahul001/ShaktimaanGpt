import React from 'react';
import { motion } from 'framer-motion';
import { Database, ArrowLeft } from 'lucide-react';

const ERDiagramView = ({ onClose }) => {
  return (
    <motion.div
      className="fixed inset-0 z-50 bg-black bg-opacity-80 flex items-center justify-center p-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="bg-dark-surface w-full max-w-6xl rounded-xl overflow-hidden shadow-2xl border border-gray-800"
        initial={{ scale: 0.9, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        transition={{ type: 'spring', damping: 25 }}
      >
        <div className="p-4 border-b border-gray-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="text-neon-purple" size={20} />
            <h2 className="text-xl font-display font-semibold text-white">Database Schema - ER Diagram</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-800 rounded-full transition-colors"
          >
            <ArrowLeft className="text-gray-400 hover:text-white" size={20} />
          </button>
        </div>
        
        <div className="p-6">
          <div className="mb-6 text-gray-300">
            <p>Entity Relationship Diagram for the Shaktimaan GPT application showing database tables and their relationships.</p>
          </div>
          
          <div className="bg-dark-bg p-4 rounded-lg">
            <svg width="100%" height="600" viewBox="0 0 1200 600" className="border border-gray-800 rounded-lg">
              {/* Background */}
              <rect x="0" y="0" width="1200" height="600" fill="#121212" rx="8" ry="8" />
              
              {/* Tables */}
              {/* auth.users table */}
              <g>
                <rect x="100" y="50" width="250" height="150" fill="#1E1E1E" stroke="#333333" strokeWidth="2" rx="8" ry="8" />
                <rect x="100" y="50" width="250" height="40" fill="#9D00FF" rx="8" ry="8" />
                <text x="225" y="75" textAnchor="middle" fill="white" fontFamily="Arial" fontSize="16" fontWeight="bold">auth.users</text>
                <line x1="100" y1="90" x2="350" y2="90" stroke="#333333" strokeWidth="2" />
                <text x="115" y="115" fill="white" fontFamily="Arial" fontSize="14">id (UUID) PK</text>
                <text x="115" y="140" fill="white" fontFamily="Arial" fontSize="14">email (TEXT)</text>
                <text x="115" y="165" fill="white" fontFamily="Arial" fontSize="14">password (TEXT)</text>
                <text x="115" y="190" fill="white" fontFamily="Arial" fontSize="14">created_at (TIMESTAMP)</text>
              </g>
              
              {/* profiles table */}
              <g>
                <rect x="100" y="300" width="250" height="220" fill="#1E1E1E" stroke="#333333" strokeWidth="2" rx="8" ry="8" />
                <rect x="100" y="300" width="250" height="40" fill="#9D00FF" rx="8" ry="8" />
                <text x="225" y="325" textAnchor="middle" fill="white" fontFamily="Arial" fontSize="16" fontWeight="bold">profiles</text>
                <line x1="100" y1="340" x2="350" y2="340" stroke="#333333" strokeWidth="2" />
                <text x="115" y="365" fill="white" fontFamily="Arial" fontSize="14">id (UUID) PK/FK</text>
                <text x="115" y="390" fill="white" fontFamily="Arial" fontSize="14">email (TEXT)</text>
                <text x="115" y="415" fill="white" fontFamily="Arial" fontSize="14">display_name (TEXT)</text>
                <text x="115" y="440" fill="white" fontFamily="Arial" fontSize="14">phone_number (TEXT)</text>
                <text x="115" y="465" fill="white" fontFamily="Arial" fontSize="14">provider_type (TEXT)</text>
                <text x="115" y="490" fill="white" fontFamily="Arial" fontSize="14">avatar_url (TEXT)</text>
                <text x="115" y="515" fill="white" fontFamily="Arial" fontSize="14">created_at (TIMESTAMP)</text>
              </g>
              
              {/* chat_histories table */}
              <g>
                <rect x="500" y="300" width="250" height="200" fill="#1E1E1E" stroke="#333333" strokeWidth="2" rx="8" ry="8" />
                <rect x="500" y="300" width="250" height="40" fill="#9D00FF" rx="8" ry="8" />
                <text x="625" y="325" textAnchor="middle" fill="white" fontFamily="Arial" fontSize="16" fontWeight="bold">chat_histories</text>
                <line x1="500" y1="340" x2="750" y2="340" stroke="#333333" strokeWidth="2" />
                <text x="515" y="365" fill="white" fontFamily="Arial" fontSize="14">id (UUID) PK</text>
                <text x="515" y="390" fill="white" fontFamily="Arial" fontSize="14">user_id (UUID) FK</text>
                <text x="515" y="415" fill="white" fontFamily="Arial" fontSize="14">messages (JSONB)</text>
                <text x="515" y="440" fill="white" fontFamily="Arial" fontSize="14">title (TEXT)</text>
                <text x="515" y="465" fill="white" fontFamily="Arial" fontSize="14">created_at (TIMESTAMP)</text>
                <text x="515" y="490" fill="white" fontFamily="Arial" fontSize="14">updated_at (TIMESTAMP)</text>
              </g>
              
              {/* storage.objects table */}
              <g>
                <rect x="500" y="50" width="250" height="150" fill="#1E1E1E" stroke="#333333" strokeWidth="2" rx="8" ry="8" />
                <rect x="500" y="50" width="250" height="40" fill="#9D00FF" rx="8" ry="8" />
                <text x="625" y="75" textAnchor="middle" fill="white" fontFamily="Arial" fontSize="16" fontWeight="bold">storage.objects</text>
                <line x1="500" y1="90" x2="750" y2="90" stroke="#333333" strokeWidth="2" />
                <text x="515" y="115" fill="white" fontFamily="Arial" fontSize="14">id (UUID) PK</text>
                <text x="515" y="140" fill="white" fontFamily="Arial" fontSize="14">bucket_id (TEXT) FK</text>
                <text x="515" y="165" fill="white" fontFamily="Arial" fontSize="14">name (TEXT)</text>
                <text x="515" y="190" fill="white" fontFamily="Arial" fontSize="14">owner (UUID)</text>
              </g>
              
              {/* storage.buckets table */}
              <g>
                <rect x="850" y="50" width="250" height="150" fill="#1E1E1E" stroke="#333333" strokeWidth="2" rx="8" ry="8" />
                <rect x="850" y="50" width="250" height="40" fill="#9D00FF" rx="8" ry="8" />
                <text x="975" y="75" textAnchor="middle" fill="white" fontFamily="Arial" fontSize="16" fontWeight="bold">storage.buckets</text>
                <line x1="850" y1="90" x2="1100" y2="90" stroke="#333333" strokeWidth="2" />
                <text x="865" y="115" fill="white" fontFamily="Arial" fontSize="14">id (TEXT) PK</text>
                <text x="865" y="140" fill="white" fontFamily="Arial" fontSize="14">name (TEXT)</text>
                <text x="865" y="165" fill="white" fontFamily="Arial" fontSize="14">public (BOOLEAN)</text>
                <text x="865" y="190" fill="white" fontFamily="Arial" fontSize="14">created_at (TIMESTAMP)</text>
              </g>
              
              {/* Legend */}
              <g>
                <rect x="850" y="300" width="250" height="150" fill="#1E1E1E" stroke="#333333" strokeWidth="2" rx="8" ry="8" />
                <rect x="850" y="300" width="250" height="40" fill="#9D00FF" rx="8" ry="8" />
                <text x="975" y="325" textAnchor="middle" fill="white" fontFamily="Arial" fontSize="16" fontWeight="bold">Legend</text>
                <line x1="850" y1="340" x2="1100" y2="340" stroke="#333333" strokeWidth="2" />
                <text x="865" y="365" fill="white" fontFamily="Arial" fontSize="14">PK - Primary Key</text>
                <text x="865" y="390" fill="white" fontFamily="Arial" fontSize="14">FK - Foreign Key</text>
                <text x="865" y="415" fill="white" fontFamily="Arial" fontSize="14">1 - One record</text>
                <text x="865" y="440" fill="white" fontFamily="Arial" fontSize="14">N - Many records</text>
              </g>
              
              {/* Relationships */}
              {/* auth.users to profiles (1:1) */}
              <line x1="225" y1="200" x2="225" y2="300" stroke="#9D00FF" strokeWidth="2" />
              <text x="235" y="220" fill="white" fontFamily="Arial" fontSize="14" fontWeight="bold">1</text>
              <text x="235" y="280" fill="white" fontFamily="Arial" fontSize="14" fontWeight="bold">1</text>
              
              {/* profiles to chat_histories (1:N) */}
              <line x1="350" y1="400" x2="500" y2="400" stroke="#9D00FF" strokeWidth="2" />
              <text x="365" y="390" fill="white" fontFamily="Arial" fontSize="14" fontWeight="bold">1</text>
              <text x="485" y="390" fill="white" fontFamily="Arial" fontSize="14" fontWeight="bold">N</text>
              <path d="M500,400 L485,395 L485,405 Z" fill="#9D00FF" />
              
              {/* storage.buckets to storage.objects (1:N) */}
              <line x1="750" y1="125" x2="850" y2="125" stroke="#9D00FF" strokeWidth="2" />
              <text x="765" y="115" fill="white" fontFamily="Arial" fontSize="14" fontWeight="bold">N</text>
              <text x="835" y="115" fill="white" fontFamily="Arial" fontSize="14" fontWeight="bold">1</text>
              <path d="M750,125 L765,120 L765,130 Z" fill="#9D00FF" />
              
              {/* storage.objects to profiles (avatar) (1:N) */}
              <line x1="625" y1="200" x2="625" y2="300" stroke="#9D00FF" strokeWidth="2" strokeDasharray="5,5" />
              <text x="635" y="220" fill="white" fontFamily="Arial" fontSize="14" fontWeight="bold">1</text>
              <text x="635" y="280" fill="white" fontFamily="Arial" fontSize="14" fontWeight="bold">N</text>
              <path d="M625,300 L620,285 L630,285 Z" fill="#9D00FF" />
            </svg>
          </div>
          
          <div className="mt-6 text-gray-300">
            <h3 className="text-lg font-medium mb-2 text-white">Database Schema Description</h3>
            <ul className="list-disc pl-5 space-y-2">
              <li><strong>auth.users</strong>: Supabase authentication table that stores user credentials.</li>
              <li><strong>profiles</strong>: Stores user profile information with a 1:1 relationship to auth.users.</li>
              <li><strong>chat_histories</strong>: Stores chat conversations between users and the AI. Each user can have multiple chat histories.</li>
              <li><strong>storage.buckets</strong>: Supabase storage buckets configuration.</li>
              <li><strong>storage.objects</strong>: Stores files like profile pictures in the storage buckets.</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default ERDiagramView;