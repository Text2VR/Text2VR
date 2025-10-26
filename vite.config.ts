import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'web-dist',
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/generate': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/status': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/panorama': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/segmentation': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/inpainted': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})