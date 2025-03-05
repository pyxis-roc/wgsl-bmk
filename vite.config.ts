import { type UserConfig, defineConfig } from "vite";

/** @type {import('vite').UserConfig} */
export default defineConfig({
	root: "./src",
	publicDir: "../public",
	base: "/wgsl-bmk",
	build: {
		outDir: "../dist",
		emptyOutDir: true,
		target: "es2022",
		minify: false,
		rollupOptions: {
			input: {
				index: "/index.html",
				dequantize: "/benchmarks/dequantize.html",
				matmul: "/benchmarks/matmul.html",
			},
		},
	},
	resolve: {
		alias: {
			"#classes": "/scripts/classes",
			"#scripts": "/scripts",
			"#shaders": "/shaders",
		},
	},
}) satisfies UserConfig;
