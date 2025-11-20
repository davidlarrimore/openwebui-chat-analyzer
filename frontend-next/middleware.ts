import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { logAuthEvent } from "@/lib/logger";

const PUBLIC_PATHS = new Set(["/", "/login", "/register"]);
const PUBLIC_PREFIXES = ["/_next", "/assets", "/favicon", "/api/backend/auth", "/api/health", "/api/public"];

function isPublic(pathname: string): boolean {
  if (PUBLIC_PATHS.has(pathname)) {
    return true;
  }
  return PUBLIC_PREFIXES.some((prefix) => pathname.startsWith(prefix));
}

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  if (!pathname.startsWith("/dashboard") || isPublic(pathname)) {
    return NextResponse.next();
  }

  try {
    const sessionUrl = new URL("/api/backend/auth/session", request.nextUrl.origin);
    const response = await fetch(sessionUrl, {
      headers: {
        cookie: request.headers.get("cookie") ?? "",
        "x-analyzer-internal": "true"
      },
      credentials: "include"
    });

    if (response.status === 200) {
      return NextResponse.next();
    }

    logAuthEvent("warn", "Middleware detected missing session; redirecting to login.", {
      path: pathname,
      status: response.status
    });
  } catch (error) {
    logAuthEvent("error", "Session check failed; redirecting.", {
      path: pathname,
      message: error instanceof Error ? error.message : "unknown"
    });
  }

  const loginUrl = new URL("/login", request.nextUrl.origin);
  loginUrl.searchParams.set("callbackUrl", pathname + request.nextUrl.search);
  return NextResponse.redirect(loginUrl);
}

export const config = {
  matcher: ["/dashboard/:path*"]
};
