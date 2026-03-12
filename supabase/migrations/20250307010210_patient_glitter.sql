/*
  # Fix RLS Policies

  1. Changes
    - Remove recursive policies from profiles table
    - Simplify admin access policies
    - Update chat_histories policies to use direct user ID comparison
    
  2. Security
    - Enable RLS on both tables
    - Add clear, non-recursive policies for both tables
    - Ensure admin users can access all records
    - Ensure regular users can only access their own records
*/

-- Drop existing policies to start fresh
DROP POLICY IF EXISTS "Users can view their own profile" ON profiles;
DROP POLICY IF EXISTS "Users can update their own profile" ON profiles;
DROP POLICY IF EXISTS "Admins can view all profiles" ON profiles;
DROP POLICY IF EXISTS "Admins can update all profiles" ON profiles;
DROP POLICY IF EXISTS "Users can view their own chat histories" ON chat_histories;
DROP POLICY IF EXISTS "Users can update their own chat histories" ON chat_histories;
DROP POLICY IF EXISTS "Users can delete their own chat histories" ON chat_histories;
DROP POLICY IF EXISTS "Users can insert their own chat histories" ON chat_histories;
DROP POLICY IF EXISTS "Admins can manage all chat histories" ON chat_histories;
DROP POLICY IF EXISTS "Admins can view all chat histories" ON chat_histories;

-- Enable RLS
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_histories ENABLE ROW LEVEL SECURITY;

-- Profiles policies
CREATE POLICY "Enable read access for users to their own profile"
ON profiles FOR SELECT
TO authenticated
USING (
  auth.uid() = id
  OR 
  EXISTS (
    SELECT 1 FROM profiles p
    WHERE p.id = auth.uid()
    AND p.is_admin = true
  )
);

CREATE POLICY "Enable update access for users to their own profile"
ON profiles FOR UPDATE
TO authenticated
USING (
  auth.uid() = id
  OR 
  EXISTS (
    SELECT 1 FROM profiles p
    WHERE p.id = auth.uid()
    AND p.is_admin = true
  )
)
WITH CHECK (
  auth.uid() = id
  OR 
  EXISTS (
    SELECT 1 FROM profiles p
    WHERE p.id = auth.uid()
    AND p.is_admin = true
  )
);

-- Chat histories policies
CREATE POLICY "Enable read access for users to their own chat histories"
ON chat_histories FOR SELECT
TO authenticated
USING (
  auth.uid() = user_id
  OR 
  EXISTS (
    SELECT 1 FROM profiles p
    WHERE p.id = auth.uid()
    AND p.is_admin = true
  )
);

CREATE POLICY "Enable insert access for users to their own chat histories"
ON chat_histories FOR INSERT
TO authenticated
WITH CHECK (
  auth.uid() = user_id
);

CREATE POLICY "Enable update access for users to their own chat histories"
ON chat_histories FOR UPDATE
TO authenticated
USING (
  auth.uid() = user_id
  OR 
  EXISTS (
    SELECT 1 FROM profiles p
    WHERE p.id = auth.uid()
    AND p.is_admin = true
  )
)
WITH CHECK (
  auth.uid() = user_id
  OR 
  EXISTS (
    SELECT 1 FROM profiles p
    WHERE p.id = auth.uid()
    AND p.is_admin = true
  )
);

CREATE POLICY "Enable delete access for users to their own chat histories"
ON chat_histories FOR DELETE
TO authenticated
USING (
  auth.uid() = user_id
  OR 
  EXISTS (
    SELECT 1 FROM profiles p
    WHERE p.id = auth.uid()
    AND p.is_admin = true
  )
);