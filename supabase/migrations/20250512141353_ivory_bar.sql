/*
  # Fix User Signup Process

  1. Changes
    - Update handle_new_user trigger function to handle phone_number from auth metadata
    - Ensure phone number is properly set during user creation
    
  2. Security
    - No changes to security policies needed
*/

-- Update the handle_new_user function to handle phone number
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (
    id,
    email,
    phone_number
  )
  VALUES (
    new.id,
    new.email,
    COALESCE(
      (new.raw_user_meta_data->>'phone_number')::text,
      'PENDING_UPDATE'
    )
  );
  RETURN new;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;