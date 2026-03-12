/*
  # Fix RLS Policies

  1. Changes
    - Remove existing policies safely
    - Re-enable RLS on tables
    - Create simplified policies for user access
    
  2. Security
    - Basic user data access policies
    - Self-service profile management
    - Chat history management for users
*/

-- Safely handle policy removal
DO $$ 
BEGIN
  -- Drop profile policies if they exist
  IF EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE schemaname = 'public' 
    AND tablename = 'profiles' 
    AND policyname = 'Enable read access for users to their own profile'
  ) THEN
    DROP POLICY "Enable read access for users to their own profile" ON profiles;
  END IF;

  IF EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE schemaname = 'public' 
    AND tablename = 'profiles' 
    AND policyname = 'Enable update access for users to their own profile'
  ) THEN
    DROP POLICY "Enable update access for users to their own profile" ON profiles;
  END IF;

  -- Drop chat histories policies if they exist
  IF EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE schemaname = 'public' 
    AND tablename = 'chat_histories' 
    AND policyname = 'Enable read access for users to their own chat histories'
  ) THEN
    DROP POLICY "Enable read access for users to their own chat histories" ON chat_histories;
  END IF;

  IF EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE schemaname = 'public' 
    AND tablename = 'chat_histories' 
    AND policyname = 'Enable insert access for users to their own chat histories'
  ) THEN
    DROP POLICY "Enable insert access for users to their own chat histories" ON chat_histories;
  END IF;

  IF EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE schemaname = 'public' 
    AND tablename = 'chat_histories' 
    AND policyname = 'Enable update access for users to their own chat histories'
  ) THEN
    DROP POLICY "Enable update access for users to their own chat histories" ON chat_histories;
  END IF;

  IF EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE schemaname = 'public' 
    AND tablename = 'chat_histories' 
    AND policyname = 'Enable delete access for users to their own chat histories'
  ) THEN
    DROP POLICY "Enable delete access for users to their own chat histories" ON chat_histories;
  END IF;
END $$;

-- Enable RLS on tables
DO $$ 
BEGIN
  ALTER TABLE IF EXISTS public.profiles ENABLE ROW LEVEL SECURITY;
  ALTER TABLE IF EXISTS public.chat_histories ENABLE ROW LEVEL SECURITY;
END $$;

-- Create new policies for profiles
DO $$ 
BEGIN
  -- Profile policies
  CREATE POLICY "Users can read own profile"
    ON public.profiles
    FOR SELECT
    TO authenticated
    USING (auth.uid() = id);

  CREATE POLICY "Users can update own profile"
    ON public.profiles
    FOR UPDATE
    TO authenticated
    USING (auth.uid() = id)
    WITH CHECK (auth.uid() = id);

  -- Chat histories policies
  CREATE POLICY "Users can read own chat histories"
    ON public.chat_histories
    FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

  CREATE POLICY "Users can insert own chat histories"
    ON public.chat_histories
    FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

  CREATE POLICY "Users can update own chat histories"
    ON public.chat_histories
    FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

  CREATE POLICY "Users can delete own chat histories"
    ON public.chat_histories
    FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);
END $$;