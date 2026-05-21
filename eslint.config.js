import js from "@eslint/js";
import prettier from "eslint-config-prettier";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: [
      "dist/**",
      "coverage/**",
      ".harness/**",
      "node_modules/**",
      "eslint.config.js",
      "prettier.config.js",
      "src/server/dev-chat/**"
    ]
  },
  js.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        project: true,
        tsconfigRootDir: import.meta.dirname
      }
    },
    rules: {
      "@typescript-eslint/consistent-type-definitions": ["error", "type"],
      "@typescript-eslint/no-confusing-void-expression": "off",
      "@typescript-eslint/no-empty-function": "off",
      "@typescript-eslint/no-inferrable-types": "off",
      "@typescript-eslint/restrict-template-expressions": [
        "error",
        {
          "allowBoolean": true,
          "allowNullish": true,
          "allowNumber": true,
          "allowRegExp": true
        }
      ],
      "no-console": ["warn", { "allow": ["info", "warn", "error"] }]
    }
  },
  prettier
);
