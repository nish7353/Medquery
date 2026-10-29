/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Outfit"', "system-ui", "sans-serif"],
      },
      // Warm, natural palette: paper ground, forest ink, one lime accent used sparingly.
      colors: {
        paper: "#EFEDE2",
        surface: "#F8F7F0",
        raised: "#FFFFFF",
        ink: { DEFAULT: "#1F2A1C", hover: "#2B3827", muted: "#545B4A" },
        line: "#E3E0D2",
        lime: { DEFAULT: "#C2CB52", hover: "#B4BD45", soft: "#E6EAC9" },
        sage: "#D8E5D6",
        butter: "#F5F1B0",
        success: { DEFAULT: "#2F5D20", bg: "#E6EAC9" },
        warning: { DEFAULT: "#8A5A00", bg: "#FBEFD3" },
        danger: { DEFAULT: "#9B2C1F", hover: "#86251A", bg: "#F8DED8" },
        info: { DEFAULT: "#2D4F6B", bg: "#DCE8F0" },
      },
      fontSize: {
        display: ["40px", { lineHeight: "44px", fontWeight: "600" }],
        h2: ["24px", { lineHeight: "30px", fontWeight: "600" }],
        h3: ["18px", { lineHeight: "24px", fontWeight: "500" }],
        label: ["12px", { lineHeight: "16px", fontWeight: "600", letterSpacing: "1px" }],
      },
      borderRadius: {
        md: "14px",
        lg: "20px",
        xl: "28px",
      },
      boxShadow: {
        pop: "0 8px 24px rgba(31, 42, 28, 0.08)",
      },
      keyframes: {
        "fade-up": { from: { opacity: 0, transform: "translateY(6px)" }, to: { opacity: 1, transform: "none" } },
        blink: { "0%, 80%, 100%": { opacity: 0.25 }, "40%": { opacity: 1 } },
      },
      animation: {
        "fade-up": "fade-up 240ms ease-out both",
        blink: "blink 1.2s infinite ease-in-out",
      },
    },
  },
  plugins: [],
};
