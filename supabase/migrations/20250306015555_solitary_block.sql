/*
  # Add admin role and policies

  1. Changes
    - Add `is_admin` boolean column to profiles table with default false
    - Add policies for admin access to profiles and chat_histories
    - Use DO blocks to safely check and create policies
*/

-- Add is_admin column to profiles table if it doesn't exist
DO $$ 
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'profiles' AND column_name = 'is_admin'
  ) THEN
    ALTER TABLE profiles ADD COLUMN is_admin BOOLEAN DEFAULT false;
  END IF;
END $$;

-- Create a helper function to check if the current user is an admin.
-- This function is defined with SECURITY DEFINER so it bypasses RLS.
CREATE OR REPLACE FUNCTION public.is_admin_user()
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  is_admin_flag BOOLEAN;
BEGIN
  SELECT is_admin INTO is_admin_flag FROM profiles WHERE id = auth.uid();
  RETURN is_admin_flag;
END;
$$;

-- Safely create policies for admin access using the helper function
DO $$ 
BEGIN
  -- "Admins can view all profiles" policy
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE tablename = 'profiles' AND policyname = 'Admins can view all profiles'
  ) THEN
    CREATE POLICY "Admins can view all profiles"
      ON profiles
      FOR SELECT
      TO authenticated
      USING (
        auth.uid() = id OR public.is_admin_user()
      );
  END IF;

  -- "Admins can update all profiles" policy
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE tablename = 'profiles' AND policyname = 'Admins can update all profiles'
  ) THEN
    CREATE POLICY "Admins can update all profiles"
      ON profiles
      FOR UPDATE
      TO authenticated
      USING (
        auth.uid() = id OR public.is_admin_user()
      );
  END IF;

  -- "Admins can view all chat histories" policy
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE tablename = 'chat_histories' AND policyname = 'Admins can view all chat histories'
  ) THEN
    CREATE POLICY "Admins can view all chat histories"
      ON chat_histories
      FOR SELECT
      TO authenticated
      USING (
        public.is_admin_user()
      );
  END IF;

  -- "Admins can manage all chat histories" policy
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'chat_histories' AND policyname = 'Admins can manage all chat histories'
  ) THEN
    CREATE POLICY "Admins can manage all chat histories"
      ON chat_histories
      FOR ALL
      TO authenticated
      USING (
        public.is_admin_user()
      );
  END IF;
END $$;
