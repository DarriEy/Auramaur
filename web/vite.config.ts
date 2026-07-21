import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api to the FastAPI service (`auramaur web`), so the SPA
// and the API share an origin in both dev and production.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8484",
        changeOrigin: false,
      },
    },
  },
});
