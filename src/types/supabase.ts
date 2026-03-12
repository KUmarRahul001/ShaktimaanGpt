export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      chat_histories: {
        Row: {
          id: string
          user_id: string
          messages: Json
          title: string
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          messages: Json
          title?: string
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          messages?: Json
          title?: string
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "chat_histories_user_id_fkey"
            columns: ["user_id"]
            referencedRelation: "users"
            referencedColumns: ["id"]
          }
        ]
      }
      profiles: {
        Row: {
          id: string
          email: string
          display_name?: string
          phone_number?: string
          provider_type?: string
          avatar_url?: string
          created_at: string
          is_admin?: boolean
        }
        Insert: {
          id: string
          email: string
          display_name?: string
          phone_number?: string
          provider_type?: string
          avatar_url?: string
          created_at?: string
          is_admin?: boolean
        }
        Update: {
          id?: string
          email?: string
          display_name?: string
          phone_number?: string
          provider_type?: string
          avatar_url?: string
          created_at?: string
          is_admin?: boolean
        }
        Relationships: [
          {
            foreignKeyName: "profiles_id_fkey"
            columns: ["id"]
            referencedRelation: "users"
            referencedColumns: ["id"]
          }
        ]
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}


