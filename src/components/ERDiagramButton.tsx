import React, { useState } from 'react';
import { Database } from 'lucide-react';
import { AnimatePresence } from 'framer-motion';
import ERDiagramView from './ERDiagramView';

const ERDiagramButton = () => {
  const [showDiagram, setShowDiagram] = useState(false);
  
  return (
    <>
      <button
        onClick={() => setShowDiagram(true)}
        className="flex items-center gap-2 px-4 py-2 bg-dark-surface-2 hover:bg-gray-800 rounded-lg transition-colors"
      >
        <Database size={18} className="text-neon-purple" />
        <span>View ER Diagram</span>
      </button>
      
      <AnimatePresence>
        {showDiagram && (
          <ERDiagramView onClose={() => setShowDiagram(false)} />
        )}
      </AnimatePresence>
    </>
  );
};

export default ERDiagramButton;