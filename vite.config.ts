import { defineConfig, UserConfig } from "vite";

/** @type {import('vite').UserConfig} */
export default defineConfig({
  base: "./public",
  root: "./src",
  build: {
    outDir: "../public/dist", // corresponds to "outDir": "./public/dist"
    target: "es2022", // corresponds to "target": "es2022"
    minify: false,
  },
  resolve: {
    alias: {
      "@src": "/src",
    },
  },
}) satisfies UserConfig;
