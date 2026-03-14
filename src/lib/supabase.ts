import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables. Please check your .env file.')
}

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
})

/* ---------------- TYPES ---------------- */

export type UserProfile = {
  id: string
  email: string
  display_name?: string
  phone_number?: string
  avatar_url?: string
  created_at: string
  is_admin?: boolean
  is_pro?: boolean
  pro_expires_at?: string
}

export type ChatHistory = {
  id: string
  user_id: string
  messages: any[]
  title: string
  created_at: string
  updated_at: string
}

/* ---------------- AUTH ---------------- */

export async function signUp(email: string, password: string, phoneNumber: string) {

  const { data: existingUser } = await supabase
    .from('profiles')
    .select('email')
    .eq('email', email)
    .maybeSingle()

  if (existingUser) {
    throw new Error('User already registered')
  }

  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      emailRedirectTo: window.location.origin,
      data: { phone_number: phoneNumber }
    }
  })

  if (error) throw error

  return data
}

export async function signIn(email: string, password: string) {

  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password
  })

  if (error) throw error

  return data
}

export async function signOut() {

  const { error } = await supabase.auth.signOut()

  if (error) throw error
}

/* -------- PASSWORD RESET -------- */

export async function resetPassword(email: string) {

  const { error } = await supabase.auth.resetPasswordForEmail(email, {
    redirectTo: `${window.location.origin}/reset-password`
  })

  if (error) throw error

  return true
}

/* ---------------- USER ---------------- */

export async function getCurrentUser() {

  const { data: { session } } = await supabase.auth.getSession()

  return session?.user ?? null
}

export async function getUserProfile(userId: string): Promise<UserProfile | null> {

  const { data, error } = await supabase
    .from('profiles')
    .select('*')
    .eq('id', userId)
    .single()

  if (error) return null

  return data as UserProfile
}

export async function updateUserProfile(userId: string, updates: Partial<UserProfile>) {

  const { data, error } = await supabase
    .from('profiles')
    .update(updates)
    .eq('id', userId)
    .select()

  if (error) throw error

  return data[0] as UserProfile
}

export async function updateProStatus(
  userId: string,
  isPro: boolean,
  expiresAt?: string
) {

  const updates: any = {
    is_pro: isPro,
    pro_expires_at: expiresAt
  }

  const { data, error } = await supabase
    .from('profiles')
    .update(updates)
    .eq('id', userId)
    .select()

  if (error) throw error

  return data[0]
}

/* ---------------- STORAGE ---------------- */

export async function uploadProfilePicture(userId: string, file: File) {

  const ext = file.name.split('.').pop()
  const fileName = `${userId}-${Date.now()}.${ext}`
  const filePath = `avatars/${fileName}`

  const { error } = await supabase.storage
    .from('user-avatars')
    .upload(filePath, file)

  if (error) throw error

  const { data } = supabase.storage
    .from('user-avatars')
    .getPublicUrl(filePath)

  await updateUserProfile(userId, { avatar_url: data.publicUrl })

  return data.publicUrl
}

export async function uploadChatImage(userId: string, file: File) {

  const ext = file.name.split('.').pop()
  const fileName = `${userId}-${Date.now()}.${ext}`
  const filePath = `chat-images/${fileName}`

  const { error } = await supabase.storage
    .from('chat-images')
    .upload(filePath, file)

  if (error) throw error

  const { data } = supabase.storage
    .from('chat-images')
    .getPublicUrl(filePath)

  return data.publicUrl
}

/* ---------------- CHAT HISTORY ---------------- */

export async function saveChatHistory(userId: string, messages: any[], title: string = 'New Chat') {

  const { data, error } = await supabase
    .from('chat_history')
    .insert([{ user_id: userId, messages, title }])
    .select()

  if (error) throw error

  return data[0]
}

export async function updateChatHistory(chatId: string, messages: any[], title?: string) {

  // ✅ Fixed: removed updated_at — column does not exist in chat_history table
  const updates: any = { messages }

  if (title) updates.title = title

  const { data, error } = await supabase
    .from('chat_history')
    .update(updates)
    .eq('id', chatId)
    .select()

  if (error) throw error

  return data[0]
}

export async function getChatHistories(userId: string) {

  const { data, error } = await supabase
    .from('chat_history')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false }) // ✅ Fixed: was updated_at, column does not exist

  if (error) throw error

  return data
}

export async function getChatHistory(chatId: string) {

  const { data, error } = await supabase
    .from('chat_history')
    .select('*')
    .eq('id', chatId)
    .single()

  if (error) throw error

  return data
}

export async function deleteChatHistory(chatId: string) {

  const { error } = await supabase
    .from('chat_history')
    .delete()
    .eq('id', chatId)

  if (error) throw error
}

/* ---------------- ACCOUNT DELETE ---------------- */

export async function deleteUserAccount(userId: string) {

  await supabase
    .from('chat_history')
    .delete()
    .eq('user_id', userId)

  await supabase
    .from('profiles')
    .delete()
    .eq('id', userId)

  return true
}