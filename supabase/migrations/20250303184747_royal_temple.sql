/*
  # Add profile fields

  1. Changes
    - Add new columns to the profiles table:
      - `display_name` (text, nullable)
      - `phone_number` (text, nullable)
      - `provider_type` (text, nullable)
      - `avatar_url` (text, nullable)
  
  2. Security
    - No changes to security policies
*/

-- Add new columns to profiles table
ALTER TABLE IF EXISTS profiles 
ADD COLUMN IF NOT EXISTS display_name TEXT,
ADD COLUMN IF NOT EXISTS phone_number TEXT,
ADD COLUMN IF NOT EXISTS provider_type TEXT,
ADD COLUMN IF NOT EXISTS avatar_url TEXT;