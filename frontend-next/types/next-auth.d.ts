import NextAuth, { type DefaultSession } from "next-auth";
import "next-auth/jwt";
import type { User } from "@/lib/types";

declare module "next-auth" {
  interface Session extends DefaultSession {
    accessToken?: string;
    user?: User;
  }

  interface User {
    accessToken?: string;
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    accessToken?: string;
    user?: User;
  }
}
