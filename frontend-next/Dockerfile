# syntax=docker/dockerfile:1.4

FROM node:20-alpine AS deps
RUN apk add --no-cache libc6-compat \
  && corepack enable
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN pnpm install

FROM node:20-alpine AS builder
RUN corepack enable
WORKDIR /app
ENV NODE_ENV=production
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN pnpm install --prod=false && pnpm build

FROM node:20-alpine AS runner
RUN corepack enable
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY package.json pnpm-lock.yaml ./
COPY --from=deps /app/node_modules ./node_modules
EXPOSE 3000
CMD ["pnpm", "start"]
