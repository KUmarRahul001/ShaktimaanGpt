import React from 'react';
import { motion } from 'framer-motion';

interface IlluminatiLogoProps {
  size?: number;
  className?: string;
}

const ChakraLogo: React.FC<IlluminatiLogoProps> = ({ size = 50, className = '' }) => {
  // Define chakra colors
  const chakraColors = [
    "#FF0000", // Root - Red
    "#FFA500", // Sacral - Orange
    "#FFFF00", // Solar Plexus - Yellow
    "#00FF00", // Heart - Green
    "#00FFFF", // Throat - Light Blue
    "#0000FF", // Third Eye - Indigo
    "#9D00FF"  // Crown - Violet/Purple
  ];

  return (
    <motion.div 
      className={`relative ${className}`}
      style={{ width: size, height: size }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <motion.svg 
        width={size} 
        height={size} 
        viewBox="0 0 100 100" 
        fill="none" 
        xmlns="http://www.w3.org/2000/svg"
        className="chakra-glow"
      >
        {/* Background circle */}
        <motion.circle 
          cx="50" 
          cy="50" 
          r="45" 
          fill="#121212" 
          stroke="#9D00FF" 
          strokeWidth="2"
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        />
        
        {/* Chakra circles - from outer to inner */}
        {chakraColors.map((color, index) => {
          const radius = 40 - (index * 5);
          const delay = 0.3 + (index * 0.1);
          
          return (
            <motion.circle 
              key={index}
              cx="50" 
              cy="50" 
              r={radius} 
              fill="transparent"
              stroke={color}
              strokeWidth="2"
              strokeOpacity="0.7"
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5, delay }}
            />
          );
        })}
        
        {/* Central dot */}
        <motion.circle 
          cx="50" 
          cy="50" 
          r="5" 
          fill="#9D00FF"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.3, delay: 1 }}
        />
        
        {/* Lotus petals */}
        {[...Array(7)].map((_, i) => {
          const angle = (i * 360 / 7) * (Math.PI / 180);
          const x1 = 50 + 25 * Math.cos(angle);
          const y1 = 50 + 25 * Math.sin(angle);
          const x2 = 50 + 35 * Math.cos(angle);
          const y2 = 50 + 35 * Math.sin(angle);
          
          return (
            <motion.path
              key={i}
              d={`M50 50 L${x1} ${y1} Q${x2} ${y2} 50 50`}
              stroke={chakraColors[i % chakraColors.length]}
              strokeWidth="1.5"
              fill="none"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.8 }}
              transition={{ duration: 0.8, delay: 0.5 + (i * 0.1) }}
            />
          );
        })}
      </motion.svg>
      
      {/* Pulsing glow effect */}
      <motion.div 
        className="absolute inset-0 rounded-full bg-neon-purple opacity-20 blur-xl"
        animate={{ 
          scale: [1, 1.2, 1],
          opacity: [0.1, 0.2, 0.1]
        }}
        transition={{ 
          duration: 4,
          repeat: Infinity,
          repeatType: "reverse"
        }}
      />
    </motion.div>
  );
};

export default ChakraLogo;