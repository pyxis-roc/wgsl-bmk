import { type UserConfig, defineConfig } from "vite";

export default defineConfig(({ mode }): UserConfig => {
	return {
		root: "./src",
		publicDir: "../public",
		base: mode === "production" ? "/wgsl-bmk" : undefined,
		envDir: "../",
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
	};
});
