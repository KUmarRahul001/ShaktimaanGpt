import { createClient } from '@supabase/supabase-js';
// Delete this line:
// import { Database } from '../types/supabase'; --- IGNORE ---

// Initialize the Supabase client using environment variables
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

// Check if environment variables are available
if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables. Please check your .env file.');
}

// Create regular client for normal operations
export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: true
  },
  global: {
    headers: { 'x-application-name': 'shaktimaan-gpt' }
  },
  db: {
    schema: 'public'
  }
});

export type User = {
  id: string;
  email: string;
  created_at: string;
};

export type UserProfile = {
  id: string;
  email: string;
  display_name?: string;
  phone_number?: string;
  provider_type?: string;
  avatar_url?: string;
  created_at: string;
  is_admin?: boolean;
  is_pro?: boolean;
  pro_expires_at?: string;
};

export type ChatHistory = {
  id: string;
  user_id: string;
  messages: any[];
  title: string;
  created_at: string;
  updated_at: string;
};

export async function signUp(email: string, password: string, phoneNumber: string) {
  try {
    const { data: existingUser, error: checkError } = await supabase
      .from('profiles')
      .select('email')
      .eq('email', email)
      .maybeSingle();
    
    if (checkError && checkError.code !== 'PGRST116') {
      console.error('Error checking existing user:', checkError);
      throw checkError;
    }
    
    if (existingUser) {
      throw new Error('User already registered');
    }
    
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        emailRedirectTo: window.location.origin,
        data: {
          phone_number: phoneNumber
        }
      }
    });
    
    if (error) throw error;
    return data;
  } catch (error) {
    console.error('SignUp error:', error);
    if (error instanceof Error && 
        (error.message.includes('User already registered') || 
         error.message.includes('user_already_exists'))) {
      throw new Error('User already registered');
    }
    throw error;
  }
}

export async function signIn(email: string, password: string) {
  try {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    
    if (error) throw error;
    return data;
  } catch (error) {
    console.error('SignIn error:', error);
    throw error;
  }
}

export async function signOut() {
  const { error } = await supabase.auth.signOut();
  if (error) {
    console.error('SignOut error:', error);
    throw error;
  }
}

export async function resetPassword(email: string) {
  try {
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/reset-password`,
    });
    
    if (error) throw error;
    return true;
  } catch (error) {
    console.error('Reset password error:', error);
    throw error;
  }
}

export async function getCurrentUser() {
  try {
    const { data: { session }, error } = await supabase.auth.getSession();
    if (error) {
      if (error.name === 'AuthSessionMissingError') {
        return null;
      }
      throw error;
    }
    return session?.user || null;
  } catch (error) {
    console.error('GetCurrentUser error:', error);
    return null;
  }
}

export async function getUserProfile(userId: string): Promise<UserProfile | null> {
  try {
    const { data, error } = await supabase
      .from('profiles')
      .select('*')
      .eq('id', userId)
      .single();
    
    if (error) {
      console.error('Error fetching user profile:', error);
      return null;
    }
    
    return data as UserProfile;
  } catch (error) {
    console.error('GetUserProfile error:', error);
    return null;
  }
}

export async function updateUserProfile(userId: string, updates: Partial<UserProfile>) {
  try {
    const { data, error } = await supabase
      .from('profiles')
      .update(updates)
      .eq('id', userId)
      .select();
    
    if (error) throw error;
    return data[0] as UserProfile;
  } catch (error) {
    console.error('UpdateUserProfile error:', error);
    throw error;
  }
}

export async function uploadProfilePicture(userId: string, file: File) {
  try {
    const fileExt = file.name.split('.').pop();
    const fileName = `${userId}-${Math.random().toString(36).substring(2)}.${fileExt}`;
    const filePath = `avatars/${fileName}`;

    const { error: uploadError } = await supabase.storage
      .from('user-avatars')
      .upload(filePath, file);

    if (uploadError) throw uploadError;

    const { data } = supabase.storage
      .from('user-avatars')
      .getPublicUrl(filePath);

    // Update user profile with avatar URL
    await updateUserProfile(userId, { avatar_url: data.publicUrl });
    
    return data.publicUrl;
  } catch (error) {
    console.error('UploadProfilePicture error:', error);
    throw error;
  }
}

export async function uploadChatImage(userId: string, file: File) {
  try {
    const fileExt = file.name.split('.').pop();
    const fileName = `${userId}-${Date.now()}.${fileExt}`;
    const filePath = `chat-images/${fileName}`;

    const { error: uploadError } = await supabase.storage
      .from('chat-images')
      .upload(filePath, file);

    if (uploadError) throw uploadError;

    const { data } = supabase.storage
      .from('chat-images')
      .getPublicUrl(filePath);
    
    return data.publicUrl;
  } catch (error) {
    console.error('UploadChatImage error:', error);
    throw error;
  }
}

export async function deleteUserAccount(userId: string) {
  try {
    // Delete all user data
    await supabase
      .from('chat_history')
      .delete()
      .eq('user_id', userId);
    
    await supabase
      .from('profiles')
      .delete()
      .eq('id', userId);
    
    await supabase.auth.admin.deleteUser(userId);
    
    return true;
  } catch (error) {
    console.error('DeleteUserAccount error:', error);
    throw error;
  }
}

export async function saveChatHistory(userId: string, messages: any[], title: string = 'New Chat') {
  try {
    const { data, error } = await supabase
      .from('chat_history')
      .insert([
        { user_id: userId, messages, title }
      ])
      .select();
    
    if (error) throw error;
    return data[0];
  } catch (error) {
    console.error('SaveChatHistory error:', error);
    throw error;
  }
}

export async function updateChatHistory(chatId: string, messages: any[], title?: string) {
  try {
    const updates: any = { messages, updated_at: new Date().toISOString() };
    if (title) updates.title = title;
    
    const { data, error } = await supabase
      .from('chat_history')
      .update(updates)
      .eq('id', chatId)
      .select();
    
    if (error) throw error;
    return data[0];
  } catch (error) {
    console.error('UpdateChatHistory error:', error);
    throw error;
  }
}

export async function getChatHistories(userId: string) {
  try {
    const { data, error } = await supabase
      .from('chat_history')
      .select('*')
      .eq('user_id', userId)
      .order('updated_at', { ascending: false });
    
    if (error) throw error;
    return data;
  } catch (error) {
    console.error('GetChatHistories error:', error);
    throw error;
  }
}

export async function getChatHistory(chatId: string) {
  try {
    const { data, error } = await supabase
      .from('chat_history')
      .select('*')
      .eq('id', chatId)
      .single();
    
    if (error) throw error;
    return data;
  } catch (error) {
    console.error('GetChatHistory error:', error);
    throw error;
  }
}

export async function deleteChatHistory(chatId: string) {
  try {
    const { error } = await supabase
      .from('chat_history')
      .delete()
      .eq('id', chatId);
    
    if (error) throw error;
  } catch (error) {
    console.error('DeleteChatHistory error:', error);
    throw error;
  }
}

export async function updateProStatus(userId: string, isPro: boolean, expiresAt?: string) {
  try {
    const updates: Partial<UserProfile> = {
      is_pro: isPro,
      pro_expires_at: expiresAt
    };

    const { data, error } = await supabase
      .from('profiles')
      .update(updates)
      .eq('id', userId)
      .select();

    if (error) throw error;
    return data[0] as UserProfile;
  } catch (error) {
    console.error('UpdateProStatus error:', error);
    throw error;
  }
}

export async function checkProStatus(userId: string): Promise<boolean> {
  try {
    const { data, error } = await supabase
      .from('profiles')
      .select('is_pro, pro_expires_at')
      .eq('id', userId)
      .single();

    if (error) throw error;

    if (!data.is_pro) return false;
    if (!data.pro_expires_at) return true;

    const expiresAt = new Date(data.pro_expires_at);
    return expiresAt > new Date();
  } catch (error) {
    console.error('CheckProStatus error:', error);
    return false;
  }
}