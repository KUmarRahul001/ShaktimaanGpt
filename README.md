# Shaktimaan AI / Shaktimaan GPT Chatbot

## 1. Project Introduction

**Shaktimaan AI** is an intelligent AI chatbot platform that allows users to interact with advanced AI models through a modern web interface. The application supports AI conversations, user authentication, conversation limits for free users, Pro subscription upgrades via Cashfree payments, and scalable backend services powered by Supabase.

The project is designed to be **production-ready, scalable, and deployable using modern cloud infrastructure**.

---

## 2. Key Features

*   **AI chatbot conversations**: Interact intelligently with advanced language models.
*   **Free user daily conversation limit**: Cap usage at 10 messages per day for free-tier users.
*   **Pro subscription for unlimited AI usage**: Unlocked through a one-time seamless payment.
*   **Secure authentication via Supabase**: Reliable user registration, login, and session management.
*   **Cashfree payment integration for Pro upgrades**: Integrated payment flow for subscribing.
*   **Supabase Edge Functions for backend logic**: Securely handles payment creation and verification.
*   **Conversation history storage**: Seamlessly tracks your past interactions.
*   **Responsive UI built with React + Tailwind**: Modern, mobile-first design.
*   **Serverless deployment support**: Engineered to deploy on edge networks.
*   **Cloudflare / Netlify compatible build**: Easy hosting on modern edge platforms.

---

## 3. Technology Stack

**Frontend**
*   React
*   Vite
*   TailwindCSS
*   TypeScript

**Backend / Serverless**
*   Supabase Edge Functions
*   Supabase PostgreSQL
*   Supabase Auth
*   Supabase Storage

**AI Integration**
*   Gemini API (Google Generative AI)

**Payments**
*   Cashfree Payment Gateway

**Infrastructure**
*   Cloudflare Pages / Workers
*   Netlify (optional deployment)

---

## 4. Project Structure

```
src/
  components/       # Reusable React components
  pages/            # Main application pages
  lib/              # Utility libraries and API integrations
  hooks/            # Custom React hooks
  styles/           # Global styles and tailwind configuration

supabase/
  functions/        # Supabase Edge Functions (Deno/TypeScript)
  migrations/       # Database schemas and seed data

public/             # Static assets like images and icons
```

**Key Files & Directories:**
*   `src/lib/supabase.ts`: Initializes the Supabase client for authentication and database interactions.
*   `src/lib/cashfree.ts`: Handles requests to the backend for creating and verifying Cashfree payment orders.
*   `supabase/functions/create-payment`: An Edge Function that securely calls the Cashfree API to generate an order session.
*   `supabase/functions/verify-payment`: An Edge Function that validates the payment status with Cashfree after a user completes the checkout.

---

## 5. Environment Variables

To run the project locally, you must configure the following environment variables. Create a `.env` file in the root directory (you can copy from a `.env.example` if available).

```env
# Frontend Supabase keys (safe to expose to the client)
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key

# Supabase Service Role (backend only, keep secret)
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

# AI Provider integration
VITE_GEMINI_API_KEY=your_gemini_api_key

# Payment Gateway Configuration
CASHFREE_APP_ID=your_cashfree_app_id
CASHFREE_SECRET_KEY=your_cashfree_secret_key
```

*   `VITE_SUPABASE_URL` & `VITE_SUPABASE_ANON_KEY`: Used by the React client to talk to Supabase.
*   `SUPABASE_SERVICE_ROLE_KEY`: Used securely in Edge Functions for admin-level database operations.
*   `VITE_GEMINI_API_KEY`: Used to authenticate requests to Google's Generative AI.
*   `CASHFREE_APP_ID` & `CASHFREE_SECRET_KEY`: Required by backend Edge Functions to create and verify payments.

---

## 6. Installation & Setup

Follow these steps to set up the project locally:

1.  **Clone the repository:**
    ```bash
    git clone <repository>
    cd shaktimaan-gpt-chatbot
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Set up environment variables:**
    ```bash
    cp .env.example .env
    ```
    *Open `.env` and fill in the required keys.*

4.  **Start the development server:**
    ```bash
    npm run dev
    ```
    *The app will be available at `http://localhost:5173`.*

---

## 7. Supabase Setup

To set up the Supabase backend (Database and Edge Functions):

1.  **Create a Supabase project** at [supabase.com](https://supabase.com).
2.  **Link your local project** (requires Supabase CLI):
    ```bash
    npx supabase login
    npx supabase link --project-ref your-project-ref
    ```
3.  **Run database migrations** to set up tables and RLS policies:
    ```bash
    npx supabase db push
    ```
4.  **Deploy Edge Functions** to your project:
    ```bash
    npx supabase functions deploy create-payment --no-verify-jwt
    npx supabase functions deploy verify-payment --no-verify-jwt
    ```
    *(Note: adjust `--no-verify-jwt` depending on your exact auth setup for functions)*

---

## 8. AI Usage Limits

To manage API costs and prevent abuse, the platform implements a dual-tier usage model tracked in the database:

**Free Users**
*   Capped at **10 AI conversations/messages per day**.
*   Usage is tracked in a Supabase table (`ai_usage`) based on the user's ID and the current date.

**Pro Users**
*   Enjoy **unlimited conversations**.
*   Users gain this status when their `is_pro` flag is set to `true` in their profile after a successful Cashfree payment.

---

## 9. Payment Flow

The application uses the Cashfree payment gateway to upgrade users to Pro. The architecture uses Edge Functions to keep secrets out of the frontend:

```
User clicks Upgrade
       ↓
Frontend calls 'create-payment' edge function
       ↓
Edge function calls Cashfree API & Order is created
       ↓
User is redirected to the Cashfree checkout page
       ↓
Payment is completed by the user
       ↓
Frontend calls 'verify-payment' edge function
       ↓
Edge function validates with Cashfree and updates DB
       ↓
User upgraded to Pro (is_pro = true)
```

---

## 10. Deployment

This project is built to be serverless compatible and can be deployed easily to modern edge platforms.

**Cloudflare Pages**
To build and deploy using Wrangler:
```bash
npm run build
npm run deploy
```

**Netlify**
To deploy on Netlify, use their CLI or connect your Git repository. The build command is:
```bash
npm run build
```

The output directory is `dist`.

---

## 11. Contributing

We welcome contributions! To help improve the project:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Submit a Pull Request.

---

## 12. License

MIT License
