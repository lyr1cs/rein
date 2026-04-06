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
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('react-force-graph') || id.includes('d3-')) {
              return 'graph-vendor'
            }
            if (id.includes('recharts')) {
              return 'charts-vendor'
            }
            if (id.includes('react') || id.includes('@tanstack')) {
              return 'react-vendor'
            }
            return 'vendor'
          }
        },
      },
    },
  },
})
