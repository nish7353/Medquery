import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // In `npm run dev`, forward API calls to the FastAPI backend.
  server: { proxy: { "/api": "http://localhost:8000" } },
  test: { environment: "jsdom", include: ["tests/**/*.test.{js,jsx}"] },
});
