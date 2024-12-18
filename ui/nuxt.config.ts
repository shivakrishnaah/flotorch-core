// https://nuxt.com/docs/api/configuration/nuxt-config
import { loadEnv } from "vite";

const env = loadEnv(
  process.env.NODE_ENV || "development",
  process.cwd(),
  "NUXT_"
);

export default defineNuxtConfig({
  compatibilityDate: "2024-11-28",
  ssr: false,
  future: {
    compatibilityVersion: 4,
  },
  colorMode: {
    preference: "light",
  },
  devtools: { enabled: true },
  modules: ["@vueuse/nuxt", "@nuxt/eslint", "@nuxt/ui"],
  css: ["~/assets/css/main.css"],
  experimental: {
    typedPages: true,
    viewTransition: true,
  },
  vite: {
    server: {
      proxy: {
        "/api": {
          target: env.NUXT_API_ENDPOINT,
          changeOrigin: true,
        },
      },
    },
  },
});
