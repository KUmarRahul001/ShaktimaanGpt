import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

const ERDiagram = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    
    // Set canvas size with device pixel ratio for sharp rendering
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    // Set canvas styles
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Set colors
    const bgColor = '#121212';
    const tableColor = '#1E1E1E';
    const borderColor = '#333333';
    const headerColor = '#9D00FF';
    const textColor = '#FFFFFF';
    const relationColor = '#9D00FF';
    
    // Fill background
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw tables
    drawTable(ctx, 'auth.users', ['id (UUID) PK', 'email (TEXT)', 'password (TEXT)', 'created_at (TIMESTAMP)'], 100, 50, 250, 150);
    
    drawTable(ctx, 'profiles', ['id (UUID) PK/FK', 'email (TEXT)', 'display_name (TEXT)', 'phone_number (TEXT)', 'provider_type (TEXT)', 'avatar_url (TEXT)', 'created_at (TIMESTAMP)'], 100, 300, 250, 220);
    
    drawTable(ctx, 'chat_histories', ['id (UUID) PK', 'user_id (UUID) FK', 'messages (JSONB)', 'title (TEXT)', 'created_at (TIMESTAMP)', 'updated_at (TIMESTAMP)'], 500, 300, 250, 200);
    
    drawTable(ctx, 'storage.objects', ['id (UUID) PK', 'bucket_id (TEXT) FK', 'name (TEXT)', 'owner (UUID)', 'created_at (TIMESTAMP)'], 500, 50, 250, 150);
    
    drawTable(ctx, 'storage.buckets', ['id (TEXT) PK', 'name (TEXT)', 'public (BOOLEAN)', 'created_at (TIMESTAMP)'], 850, 50, 250, 150);
    
    // Draw relationships
    drawRelationship(ctx, 225, 200, 225, 300, '1', '1'); // auth.users to profiles
    drawRelationship(ctx, 350, 400, 500, 400, '1', 'N'); // profiles to chat_histories
    drawRelationship(ctx, 750, 125, 850, 125, '1', 'N'); // storage.buckets to storage.objects
    drawRelationship(ctx, 625, 200, 625, 300, '1', 'N'); // storage.objects to profiles (avatar)
    
    // Draw legend
    drawLegend(ctx, 850, 300);
    
  }, []);
  
  // Function to draw a table
  const drawTable = (ctx, name, fields, x, y, width, height) => {
    // Table background
    ctx.fillStyle = tableColor;
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(x, y, width, height, 8);
    ctx.fill();
    ctx.stroke();
    
    // Table header
    ctx.fillStyle = headerColor;
    ctx.beginPath();
    ctx.roundRect(x, y, width, 40, [8, 8, 0, 0]);
    ctx.fill();
    
    // Table name
    ctx.fillStyle = textColor;
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(name, x + width/2, y + 25);
    
    // Table fields
    ctx.font = '14px Arial';
    ctx.textAlign = 'left';
    fields.forEach((field, index) => {
      ctx.fillText(field, x + 15, y + 65 + (index * 25));
    });
    
    // Divider line
    ctx.strokeStyle = borderColor;
    ctx.beginPath();
    ctx.moveTo(x, y + 40);
    ctx.lineTo(x + width, y + 40);
    ctx.stroke();
  };
  
  // Function to draw a relationship
  const drawRelationship = (ctx, x1, y1, x2, y2, card1, card2) => {
    ctx.strokeStyle = relationColor;
    ctx.lineWidth = 2;
    
    // Draw line
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    
    // Draw cardinality
    ctx.font = 'bold 14px Arial';
    ctx.fillStyle = textColor;
    
    // First cardinality
    ctx.fillText(card1, x1 + 10, y1 - 5);
    
    // Second cardinality
    ctx.fillText(card2, x2 - 15, y2 - 5);
    
    // Draw arrow at the N side
    if (card2 === 'N') {
      drawArrow(ctx, x2, y2, x1, y1);
    }
    if (card1 === 'N') {
      drawArrow(ctx, x1, y1, x2, y2);
    }
  };
  
  // Function to draw an arrow
  const drawArrow = (ctx, fromX, fromY, toX, toY) => {
    const headLength = 15;
    const angle = Math.atan2(toY - fromY, toX - fromX);
    
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(fromX - headLength * Math.cos(angle - Math.PI/6), fromY - headLength * Math.sin(angle - Math.PI/6));
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(fromX - headLength * Math.cos(angle + Math.PI/6), fromY - headLength * Math.sin(angle + Math.PI/6));
    ctx.stroke();
  };
  
  // Function to draw legend
  const drawLegend = (ctx, x, y) => {
    ctx.fillStyle = tableColor;
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(x, y, 250, 150, 8);
    ctx.fill();
    ctx.stroke();
    
    ctx.fillStyle = headerColor;
    ctx.beginPath();
    ctx.roundRect(x, y, 250, 40, [8, 8, 0, 0]);
    ctx.fill();
    
    ctx.fillStyle = textColor;
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Legend', x + 125, y + 25);
    
    ctx.font = '14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('PK - Primary Key', x + 15, y + 65);
    ctx.fillText('FK - Foreign Key', x + 15, y + 90);
    ctx.fillText('1 - One record', x + 15, y + 115);
    ctx.fillText('N - Many records', x + 15, y + 140);
    
    ctx.strokeStyle = borderColor;
    ctx.beginPath();
    ctx.moveTo(x, y + 40);
    ctx.lineTo(x + 250, y + 40);
    ctx.stroke();
  };

  return (
    <canvas 
      ref={canvasRef} 
      style={{ 
        width: '100%', 
        height: '600px',
        borderRadius: '8px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
      }}
    />
  );
};

export default ERDiagram;