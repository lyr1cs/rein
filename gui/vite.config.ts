import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8680',
      '/mcp': 'http://localhost:8680',
    },
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
  },
})
