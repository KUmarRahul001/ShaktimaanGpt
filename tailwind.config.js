/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'neon-green': '#00FFAA',
        'neon-blue': '#00EEFF',
        'neon-purple': '#9D00FF',
        'dark-bg': '#000000',
        'dark-surface': '#121212',
        'dark-surface-2': '#1E1E1E',
      },
      fontFamily: {
        sans: ['Space Grotesk', 'sans-serif'],
        display: ['Rajdhani', 'sans-serif'],
      },
      boxShadow: {
        'neon': '0 0 10px rgba(157, 0, 255, 0.3)',
        'neon-strong': '0 0 20px rgba(157, 0, 255, 0.5)',
        'neon-glow': '0 0 30px rgba(157, 0, 255, 0.7)',
      },
      animation: {
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
        'slide-up': 'slideUp 0.5s ease-out',
        'fade-in': 'fadeIn 0.5s ease-out',
        'scale-in': 'scaleIn 0.3s ease-out',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: 0 },
          '100%': { transform: 'translateY(0)', opacity: 1 },
        },
        fadeIn: {
          '0%': { opacity: 0 },
          '100%': { opacity: 1 },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: 0 },
          '100%': { transform: 'scale(1)', opacity: 1 },
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'grid-pattern': "url(\"data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 0L60 30L30 60L0 30L30 0Z' fill='none' stroke='%239D00FF' stroke-opacity='0.05' stroke-width='1'/%3E%3C/svg%3E\")",
      },
      transitionProperty: {
        'height': 'height',
        'spacing': 'margin, padding',
      },
    },
  },
  plugins: [],
};