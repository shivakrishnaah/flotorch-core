// @ts-check
// @ts-expect-error
import eslintConfigPrettier from "eslint-config-prettier";
import withNuxt from "./.nuxt/eslint.config.mjs";

export default withNuxt(
  {
    rules: {
      "vue/first-attribute-linebreak": "off",
      "vue/multi-word-component-names": "off",
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
  eslintConfigPrettier
);
