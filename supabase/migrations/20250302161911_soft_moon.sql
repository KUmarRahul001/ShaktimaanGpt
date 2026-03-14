/*
  # Create chat histories table

  1. New Tables
    - `chat_histories`
      - `id` (uuid, primary key)
      - `user_id` (uuid, references profiles.id)
      - `messages` (jsonb, not null)
      - `title` (text, not null, default: 'New Chat')
      - `created_at` (timestamp with time zone, default: now())
      - `updated_at` (timestamp with time zone, default: now())
  2. Security
    - Enable RLS on `chat_histories` table
    - Add policy for users to read their own chat histories
    - Add policy for users to insert their own chat histories
    - Add policy for users to update their own chat histories
    - Add policy for users to delete their own chat histories
*/

-- Create chat_histories table
CREATE TABLE IF NOT EXISTS chat_histories (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  messages JSONB NOT NULL,
  title TEXT NOT NULL DEFAULT 'New Chat',
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE chat_histories ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Users can view their own chat histories"
  ON chat_histories
  FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own chat histories"
  ON chat_histories
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own chat histories"
  ON chat_histories
  FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own chat histories"
  ON chat_histories
  FOR DELETE
  USING (auth.uid() = user_id);

-- Create a function to update the updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create a trigger to automatically update the updated_at column
CREATE TRIGGER update_chat_histories_updated_at
  BEFORE UPDATE ON chat_histories
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();