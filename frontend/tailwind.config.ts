import type { Config } from "tailwindcss";

export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Cyber Forensic Color Palette
        cyber: {
          50: '#f0fdff',
          100: '#ccf7fe',
          200: '#99eefd',
          300: '#66e5fc',
          400: '#00d9ff', // Neon cyan
          500: '#00bcd4',
          600: '#0097a7',
          700: '#00796b',
          800: '#004d40',
          900: '#002d26',
        },
        crimson: {
          50: '#fff0f0',
          100: '#ffdddd',
          200: '#ffc0c0',
          300: '#ff9999',
          400: '#ff4444', // Crimson red
          500: '#dc2626',
          600: '#b91c1c',
          700: '#991b1b',
          800: '#7f1d1d',
          900: '#450a0a',
        },
        electric: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6', // Electric blue
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
        matrix: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80', // Matrix green
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
        },
        void: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b', // Deep charcoal
          900: '#0f172a', // Jet black
          950: '#020617', // Void black
        },
        primary: {
          50: '#f0fdff',
          100: '#ccf7fe',
          200: '#99eefd',
          300: '#66e5fc',
          400: '#00d9ff',
          500: '#00bcd4',
          600: '#0097a7',
          700: '#00796b',
          800: '#004d40',
          900: '#002d26',
        },
        background: "var(--background)",
        foreground: "var(--foreground)",
      },
      animation: {
        'glow-pulse': 'glow-pulse 2s ease-in-out infinite',
        'scan-line': 'scan-line 3s linear infinite',
        'matrix-rain': 'matrix-rain 20s linear infinite',
        'glitch': 'glitch 2s linear infinite',
        'data-flow': 'data-flow 4s ease-in-out infinite',
        'neon-flicker': 'neon-flicker 1.5s ease-in-out infinite alternate',
      },
      keyframes: {
        'glow-pulse': {
          '0%, 100%': { 
            boxShadow: '0 0 5px rgba(0, 217, 255, 0.5), 0 0 20px rgba(0, 217, 255, 0.3)',
          },
          '50%': { 
            boxShadow: '0 0 20px rgba(0, 217, 255, 0.8), 0 0 40px rgba(0, 217, 255, 0.5)',
          },
        },
        'scan-line': {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100vw)' },
        },
        'matrix-rain': {
          '0%': { transform: 'translateY(-100vh)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        'glitch': {
          '0%, 100%': { transform: 'translate(0)' },
          '20%': { transform: 'translate(-2px, 2px)' },
          '40%': { transform: 'translate(-2px, -2px)' },
          '60%': { transform: 'translate(2px, 2px)' },
          '80%': { transform: 'translate(2px, -2px)' },
        },
        'data-flow': {
          '0%, 100%': { opacity: '0.4', transform: 'translateX(-10px)' },
          '50%': { opacity: '1', transform: 'translateX(10px)' },
        },
        'neon-flicker': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.8' },
        },
      },
      fontFamily: {
        'cyber': ['Orbitron', 'monospace'],
        'professional': ['Inter', 'system-ui', 'sans-serif'],
        'mono': ['JetBrains Mono', 'Monaco', 'Cascadia Code', 'Roboto Mono', 'monospace'],
      },
      backgroundImage: {
        'cyber-grid': `
          linear-gradient(rgba(0, 217, 255, 0.1) 1px, transparent 1px),
          linear-gradient(90deg, rgba(0, 217, 255, 0.1) 1px, transparent 1px)
        `,
        'matrix-pattern': `
          radial-gradient(circle at 25% 25%, rgba(0, 217, 255, 0.1) 0%, transparent 50%),
          radial-gradient(circle at 75% 75%, rgba(68, 222, 128, 0.1) 0%, transparent 50%)
        `,
      },
    },
  },
  plugins: [],
} satisfies Config;