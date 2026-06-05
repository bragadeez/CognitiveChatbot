/// <reference types="vite/client" />

// Tell TypeScript to treat CSS and SVG imports as modules
declare module '*.css' {
  const css: string;
  export default css;
}

declare module '*.svg' {
  const src: string;
  export default src;
}
