import { withAuth } from "next-auth/middleware";
import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { logAuthEvent } from "@/lib/logger";

async function backendHasUsers(request: NextRequest): Promise<boolean> {
  try {
    const statusUrl = new URL("/api/proxy/api/v1/auth/status", request.nextUrl.origin);
    const response = await fetch(statusUrl, {
      headers: {
        "x-next-auth-skip": "true"
      }
    });
    if (!response.ok) {
      logAuthEvent("warn", "Auth status proxy returned non-OK response.", {
        status: response.status,
        url: statusUrl.toString()
      });
      return true;
    }
    const data = (await response.json()) as { has_users?: boolean };
    logAuthEvent("debug", "Auth status proxy resolved.", {
      hasUsers: data.has_users,
      url: statusUrl.toString()
    });
    return Boolean(data.has_users);
  } catch (error) {
    logAuthEvent("warn", "Auth status proxy request failed.", {
      message: error instanceof Error ? error.message : "unknown-error"
    });
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
          logAuthEvent("info", "Authorization bypassed; no users provisioned.", {
            path: req.nextUrl.pathname
          });
          return true;
        }
        const authorized = !!token;
        if (!authorized) {
          logAuthEvent("warn", "Middleware rejected request without session token.", {
            path: req.nextUrl.pathname
          });
        }
        return authorized;
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
