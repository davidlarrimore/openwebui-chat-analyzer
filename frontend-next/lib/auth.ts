import type { NextAuthOptions } from "next-auth";
import { getServerSession } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import GitHubProvider from "next-auth/providers/github";
import { getServerConfig, ENABLE_GITHUB_OAUTH } from "@/lib/config";
import type { User } from "@/lib/types";

const serverConfig = getServerConfig();

export const authOptions: NextAuthOptions = {
  secret: serverConfig.NEXTAUTH_SECRET,
  pages: {
    signIn: "/login"
  },
  session: {
    strategy: "jwt"
  },
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        username: { label: "Username", type: "text" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        const username = credentials?.username;
        const password = credentials?.password;

        if (!username || !password) {
          throw new Error("Username and password are required.");
        }

        const response = await fetch(`${serverConfig.BACKEND_BASE_URL}/api/v1/auth/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password })
        });

        if (!response.ok) {
          const message = response.status === 401 ? "Invalid credentials." : "Unable to authenticate.";
          throw new Error(message);
        }

        const payload = (await response.json()) as {
          access_token: string;
          token_type: string;
          user: User;
        };

        if (!payload?.access_token || !payload?.user) {
          throw new Error("Malformed authentication response.");
        }

        return {
          ...payload.user,
          accessToken: payload.access_token
        };
      }
    }),
    ...(ENABLE_GITHUB_OAUTH
      ? [
          GitHubProvider({
            clientId: process.env.GITHUB_CLIENT_ID ?? "",
            clientSecret: process.env.GITHUB_CLIENT_SECRET ?? ""
          })
        ]
      : [])
  ],
  callbacks: {
    async jwt({ token, user, account }) {
      if (user) {
        token.accessToken = (user as User & { accessToken?: string }).accessToken ?? token.accessToken;
        token.user = user as User;
      }

      if (account?.provider === "github" && account.access_token) {
        token.accessToken = account.access_token;
      }

      return token;
    },
    async session({ session, token }) {
      if (token.user) {
        session.user = token.user as User;
      }

      if (token.accessToken) {
        session.accessToken = token.accessToken as string;
      }

      return session;
    }
  }
};

export function getServerAuthSession() {
  return getServerSession(authOptions);
}
