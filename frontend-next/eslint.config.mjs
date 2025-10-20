import next from "eslint-config-next";

export default [
  ...next,
  {
    ignores: ["**/pnpm-lock.yaml", "node_modules"]
  }
];
