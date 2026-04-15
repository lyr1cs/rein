import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const reinBackend = process.env.REIN_GUI_PROXY_TARGET
  ?? `http://localhost:${process.env.REIN_SSE_PORT ?? '8680'}`

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': reinBackend,
      '/mcp': reinBackend,
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
