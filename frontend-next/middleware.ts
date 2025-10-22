import { withAuth } from "next-auth/middleware";
import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

async function backendHasUsers(request: NextRequest): Promise<boolean> {
  try {
    const statusUrl = new URL("/api/proxy/api/v1/auth/status", request.nextUrl.origin);
    const response = await fetch(statusUrl, {
      headers: {
        "x-next-auth-skip": "true"
      }
    });
    if (!response.ok) {
      return true;
    }
    const data = (await response.json()) as { has_users?: boolean };
    return Boolean(data.has_users);
  } catch {
    return true;
  }
}

export default withAuth(
  () => NextResponse.next(),
  {
    callbacks: {
      authorized: async ({ token, req }) => {
        const hasUsers = await backendHasUsers(req as NextRequest);
        if (!hasUsers) {
          return true;
        }
        return !!token;
      }
    },
    pages: {
      signIn: "/login"
    }
  }
);

export const config = {
  // Only protect dashboard routes - explicitly exclude auth pages
  matcher: ["/dashboard/:path*"]
};
