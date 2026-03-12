/*
  # Database Schema Update Based on ER Diagram

  1. New Tables
    - `apis`: Store API information
    - `chatbots`: Store chatbot configurations
    - `admins`: Store admin information

  2. Changes
    - Add new relationships between tables
    - Add foreign key constraints
    - Update RLS policies
*/

-- Create apis table
CREATE TABLE IF NOT EXISTS apis (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  access_level TEXT NOT NULL,
  api_key TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Create chatbots table
CREATE TABLE IF NOT EXISTS chatbots (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  api_id UUID REFERENCES apis(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  api_connection TEXT NOT NULL,
  model_name TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Create admins table
CREATE TABLE IF NOT EXISTS admins (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  api_id UUID REFERENCES apis(id) ON DELETE SET NULL,
  name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE,
  role TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Add bot_id to chat_histories
ALTER TABLE chat_histories
ADD COLUMN IF NOT EXISTS bot_id UUID REFERENCES chatbots(id) ON DELETE SET NULL;

-- Enable RLS on new tables
ALTER TABLE apis ENABLE ROW LEVEL SECURITY;
ALTER TABLE chatbots ENABLE ROW LEVEL SECURITY;
ALTER TABLE admins ENABLE ROW LEVEL SECURITY;

-- Create updated_at triggers for new tables
CREATE TRIGGER update_apis_updated_at
  BEFORE UPDATE ON apis
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chatbots_updated_at
  BEFORE UPDATE ON chatbots
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_admins_updated_at
  BEFORE UPDATE ON admins
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- RLS Policies for apis table
CREATE POLICY "APIs are viewable by authenticated users"
  ON apis
  FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "APIs are insertable by admins"
  ON apis
  FOR INSERT
  TO authenticated
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM admins
      WHERE user_id = auth.uid()
    )
  );

CREATE POLICY "APIs are updatable by admins"
  ON apis
  FOR UPDATE
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM admins
      WHERE user_id = auth.uid()
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM admins
      WHERE user_id = auth.uid()
    )
  );

CREATE POLICY "APIs are deletable by admins"
  ON apis
  FOR DELETE
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM admins
      WHERE user_id = auth.uid()
    )
  );

-- RLS Policies for chatbots table
CREATE POLICY "Chatbots are viewable by authenticated users"
  ON chatbots
  FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Chatbots are insertable by admins"
  ON chatbots
  FOR INSERT
  TO authenticated
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM admins
      WHERE user_id = auth.uid()
    )
  );

CREATE POLICY "Chatbots are updatable by admins"
  ON chatbots
  FOR UPDATE
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM admins
      WHERE user_id = auth.uid()
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM admins
      WHERE user_id = auth.uid()
    )
  );

CREATE POLICY "Chatbots are deletable by admins"
  ON chatbots
  FOR DELETE
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM admins
      WHERE user_id = auth.uid()
    )
  );

-- RLS Policies for admins table
CREATE POLICY "Admins are viewable by authenticated users"
  ON admins
  FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Admins are insertable by super admins"
  ON admins
  FOR INSERT
  TO authenticated
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM admins
      WHERE user_id = auth.uid()
      AND role = 'super_admin'
    )
  );

CREATE POLICY "Admins are updatable by super admins"
  ON admins
  FOR UPDATE
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM admins
      WHERE user_id = auth.uid()
      AND role = 'super_admin'
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM admins
      WHERE user_id = auth.uid()
      AND role = 'super_admin'
    )
  );

CREATE POLICY "Admins are deletable by super admins"
  ON admins
  FOR DELETE
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM admins
      WHERE user_id = auth.uid()
      AND role = 'super_admin'
    )
  );

-- Update chat_histories policies to include bot_id
CREATE POLICY "Chat histories are viewable by users or admins"
  ON chat_histories
  FOR SELECT
  TO authenticated
  USING (
    auth.uid() = user_id OR
    EXISTS (
      SELECT 1 FROM admins
      WHERE user_id = auth.uid()
    )
  );

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_apis_name ON apis(name);
CREATE INDEX IF NOT EXISTS idx_chatbots_api_id ON chatbots(api_id);
CREATE INDEX IF NOT EXISTS idx_admins_user_id ON admins(user_id);
CREATE INDEX IF NOT EXISTS idx_admins_api_id ON admins(api_id);
CREATE INDEX IF NOT EXISTS idx_chat_histories_bot_id ON chat_histories(bot_id);