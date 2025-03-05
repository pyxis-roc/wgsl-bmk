import { type UserConfig, defineConfig } from "vite";

/** @type {import('vite').UserConfig} */
export default defineConfig({
	root: ".",
	base: "/wgsl-bmk",
	build: {
		outDir: "dist", // corresponds to "outDir": "./public/dist"
		target: "es2022", // corresponds to "target": "es2022"
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
			scripts: "/src/scripts",
			src: "/src",
			"@": "/src/scripts",
		},
	},
}) satisfies UserConfig;
