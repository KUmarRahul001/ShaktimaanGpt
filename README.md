# Shaktimaan AI / Shaktimaan GPT Chatbot

## 1. Project Introduction

**Shaktimaan AI** is an intelligent AI chatbot platform that allows users to interact with advanced AI models through a modern web interface. The application supports AI conversations, user authentication, conversation limits for free users, Pro subscription upgrades via Cashfree payments, and scalable backend services powered by Supabase.

The project is designed to be **production-ready, scalable, and deployable using modern cloud infrastructure**.

---

# 2. Key Features

* **AI chatbot conversations** – interact with advanced language models.
* **Free user daily conversation limit** – 10 AI messages per day.
* **Pro subscription for unlimited AI usage**
* **Secure authentication via Supabase**
* **Cashfree payment integration for Pro upgrades**
* **Supabase Edge Functions for backend logic**
* **Conversation history storage**
* **Responsive UI built with React + Tailwind**
* **Serverless deployment support**
* **Cloudflare / Netlify compatible build**

---

# 3. Technology Stack

## Frontend

* React
* Vite
* TailwindCSS
* TypeScript

## Backend / Serverless

* Supabase Edge Functions
* Supabase PostgreSQL
* Supabase Auth
* Supabase Storage

## AI Integration

* Google **Gemini API**

## Payments

* **Cashfree Payment Gateway**

## Infrastructure

* Cloudflare Pages / Workers
* Netlify

---

# 4. Project Structure

```
src/
  components/       # Reusable React components
  pages/            # Main application pages
  lib/              # Utility libraries and API integrations
  hooks/            # Custom React hooks
  styles/           # Global styles and Tailwind configuration

supabase/
  functions/        # Supabase Edge Functions
  migrations/       # Database schemas and seed data

public/             # Static assets
```

### Important Files

**src/lib/supabase.ts**
Initializes Supabase client for authentication and database access.

**src/lib/cashfree.ts**
Handles communication with Supabase Edge Functions for payment creation and verification.

**supabase/functions/create-payment**
Creates Cashfree payment order.

**supabase/functions/verify-payment**
Verifies payment status and upgrades user to Pro.

---

# 5. Environment Variables

Create a `.env` file in the project root.

Example:

```
VITE_SUPABASE_URL=
VITE_SUPABASE_ANON_KEY=

SUPABASE_SERVICE_ROLE_KEY=

VITE_GEMINI_API_KEY=

CASHFREE_APP_ID=
CASHFREE_SECRET_KEY=
```

---

# 6. How To Get API Keys (Step-by-Step)

## 6.1 Supabase Setup

1. Create an account
   https://supabase.com

2. Create a new project

3. After project creation go to:

```
Settings → API
```

Copy the following:

```
Project URL → VITE_SUPABASE_URL
Anon Public Key → VITE_SUPABASE_ANON_KEY
Service Role Key → SUPABASE_SERVICE_ROLE_KEY
```

Docs:
https://supabase.com/docs

---

## 6.2 Gemini AI API Key

1. Go to Google AI Studio
   https://aistudio.google.com

2. Login with Google account

3. Click **Get API Key**

4. Create new key

Copy the key into:

```
VITE_GEMINI_API_KEY
```

Docs:
https://ai.google.dev/docs

---

## 6.3 Cashfree Payment Gateway Setup

1. Create account
   https://www.cashfree.com

2. Go to **Dashboard**

3. Navigate to:

```
Developers → API Keys
```

4. Copy credentials:

```
App ID → CASHFREE_APP_ID
Secret Key → CASHFREE_SECRET_KEY
```

5. Choose environment:

Sandbox (for testing)
Production (for live payments)

Docs:
https://docs.cashfree.com

---

# 7. Installation & Setup

## Step 1 — Clone Repository

```
git clone https://github.com/YOUR_USERNAME/shaktimaan-ai.git
cd shaktimaan-ai
```

---

## Step 2 — Install Dependencies

```
npm install
```

---

## Step 3 — Create Environment File

```
cp .env.example .env
```

Then edit `.env` and paste your API keys.

---

## Step 4 — Install Supabase CLI

```
npm install -g supabase
```

Login:

```
supabase login
```

Docs:
https://supabase.com/docs/guides/cli

---

## Step 5 — Link Supabase Project

```
supabase link --project-ref YOUR_PROJECT_ID
```

You can find the project ID in:

```
Supabase Dashboard → Settings → General
```

---

## Step 6 — Run Database Migrations

```
supabase db push
```

This creates all required tables such as:

* profiles
* ai_usage
* messages
* payments
* conversations

---

## Step 7 — Deploy Edge Functions

```
supabase functions deploy create-payment
supabase functions deploy verify-payment
```

---

## Step 8 — Start Development Server

```
npm run dev
```

App runs at:

```
http://localhost:5173
```

---

# 8. AI Usage Limits

To control API usage and cost:

## Free Users

* 10 AI messages per day

## Pro Users

* Unlimited AI messages

Usage is stored in:

```
ai_usage
```

Database fields:

```
user_id
date
conversation_count
```

---

# 9. Payment Flow (Cashfree)

```
User clicks Upgrade
      ↓
Frontend calls create-payment
      ↓
Supabase Edge Function creates order
      ↓
Cashfree checkout page opens
      ↓
User completes payment
      ↓
verify-payment function validates payment
      ↓
Database updated → user becomes Pro
```

---

# 10. Deployment

## Cloudflare Pages

Install Wrangler:

```
npm install -g wrangler
```

Deploy:

```
npm run build
npm run deploy
```

Docs:
https://developers.cloudflare.com/pages

---

## Netlify

Connect your GitHub repository to Netlify.

Build command:

```
npm run build
```

Publish directory:

```
dist
```

Docs:
https://docs.netlify.com

---

# 11. Contributing

1. Fork the repository
2. Create a new branch

```
git checkout -b feature/your-feature
```

3. Commit your changes

```
git commit -m "Add new feature"
```

4. Push to branch

```
git push origin feature/your-feature
```

5. Open a Pull Request

---

# 12. Security Notes

* Never commit `.env` file
* Keep `SUPABASE_SERVICE_ROLE_KEY` secret
* Use Cashfree **Sandbox mode** for testing
* Rotate API keys if exposed

---

# 13. License

MIT License

You are free to use, modify, and distribute this software under the terms of the MIT License.
