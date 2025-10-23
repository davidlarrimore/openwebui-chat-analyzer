import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { getServerConfig } from "@/lib/config";

/**
 * Custom logout endpoint that:
 * 1. Revokes the backend access token
 * 2. Returns instructions to clear NextAuth session
 */
export async function POST(request: NextRequest) {
  const session = await getServerSession(authOptions);

  // Revoke backend token if we have one
  if (session?.accessToken) {
    const serverConfig = getServerConfig();
    try {
      await fetch(`${serverConfig.BACKEND_BASE_URL}/api/v1/auth/logout`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${session.accessToken}`,
          "Content-Type": "application/json",
        },
      });
    } catch (error) {
      // Even if backend revocation fails, we should still sign out locally
      console.error("Failed to revoke backend token:", error);
    }
  }

  // Return success - the client will handle NextAuth signOut
  return NextResponse.json({ ok: true });
}
