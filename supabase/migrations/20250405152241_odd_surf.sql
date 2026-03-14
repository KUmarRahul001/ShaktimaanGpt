/*
  # Add Pro Status Fields to Profiles

  1. Changes
    - Add `is_pro` boolean column to profiles table
    - Add `pro_expires_at` timestamp column to profiles table
  
  2. Security
    - No changes to security policies needed
    - Existing profile policies cover these new fields
*/

-- Add pro status columns to profiles table
ALTER TABLE IF EXISTS profiles 
ADD COLUMN IF NOT EXISTS is_pro BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS pro_expires_at TIMESTAMPTZ;

-- Create storage bucket for chat images
INSERT INTO storage.buckets (id, name, public)
VALUES ('chat-images', 'chat-images', true)
ON CONFLICT (id) DO NOTHING;

-- Set up security policies for the chat-images bucket
CREATE POLICY "Chat images are publicly accessible"
ON storage.objects FOR SELECT
USING (bucket_id = 'chat-images');

CREATE POLICY "Pro users can upload chat images"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (
  bucket_id = 'chat-images' 
  AND EXISTS (
    SELECT 1 FROM profiles
    WHERE id = auth.uid()
    AND (is_pro = true OR is_admin = true)
  )
);

CREATE POLICY "Pro users can update their chat images"
ON storage.objects FOR UPDATE
TO authenticated
USING (
  bucket_id = 'chat-images'
  AND auth.uid()::text = (storage.foldername(name))[1]
  AND EXISTS (
    SELECT 1 FROM profiles
    WHERE id = auth.uid()
    AND (is_pro = true OR is_admin = true)
  )
);

CREATE POLICY "Pro users can delete their chat images"
ON storage.objects FOR DELETE
TO authenticated
USING (
  bucket_id = 'chat-images'
  AND auth.uid()::text = (storage.foldername(name))[1]
  AND EXISTS (
    SELECT 1 FROM profiles
    WHERE id = auth.uid()
    AND (is_pro = true OR is_admin = true)
  )
);