import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  server: {
    port: 5173,
    // ส่งต่อคำขอที่ขึ้นต้นด้วย /api ไปยัง FastAPI ที่ port 8000
    // ทำแบบนี้เพื่อให้เบราว์เซอร์มองว่าเป็น origin เดียวกัน ไม่ติดปัญหา CORS
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } }
  }
})
