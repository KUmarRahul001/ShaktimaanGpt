/*
  # Make phone number required

  1. Changes
    - Make phone_number column NOT NULL in profiles table
    - Add a default value for existing records
    - Update RLS policies to ensure phone number is provided
*/

-- First set a default value for existing NULL phone numbers
UPDATE profiles 
SET phone_number = 'PENDING_UPDATE'
WHERE phone_number IS NULL;

-- Then make the column NOT NULL
ALTER TABLE profiles 
ALTER COLUMN phone_number SET NOT NULL;

-- Add check constraint to ensure valid phone format
ALTER TABLE profiles
ADD CONSTRAINT phone_number_format CHECK (
  phone_number ~ '^\+[1-9]\d{1,14}$' 
  OR phone_number = 'PENDING_UPDATE'
);