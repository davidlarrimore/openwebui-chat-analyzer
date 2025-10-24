import { NextRequest, NextResponse } from "next/server";
import { getServerAuthSession } from "@/lib/auth";
import { getServerConfig } from "@/lib/config";

const ALLOWED_PREFIX = /^api\/(?:v1)\//i;

async function handler(request: NextRequest, context: { params: { path?: string[] } }) {
  const { path = [] } = context.params;
  const targetPath = path.join("/");
  if (!ALLOWED_PREFIX.test(targetPath)) {
    return NextResponse.json({ error: "Forbidden path" }, { status: 403 });
  }

  const { BACKEND_BASE_URL } = getServerConfig();
  const url = new URL(`${BACKEND_BASE_URL.replace(/\/$/, "")}/${targetPath}`);
  const searchParams = request.nextUrl.searchParams;
  searchParams.forEach((value, key) => url.searchParams.set(key, value));

  const session = await getServerAuthSession();

  const headers = new Headers(request.headers);
  const skipAuth = headers.get("x-next-auth-skip") === "true";
  if (skipAuth) {
    headers.delete("x-next-auth-skip");
  }
  headers.delete("host");
  headers.delete("content-length");

  if (!skipAuth && session?.accessToken && !headers.has("authorization")) {
    headers.set("authorization", `Bearer ${session.accessToken}`);
  }

  let body: BodyInit | undefined;
  if (!["GET", "HEAD"].includes(request.method.toUpperCase())) {
    body = await request.text();
  }

  try {
    const backendResponse = await fetch(url.toString(), {
      method: request.method,
      headers,
      body,
      redirect: "manual"
    });

    const responseHeaders = new Headers(backendResponse.headers);
    responseHeaders.delete("transfer-encoding");

    const responseBody = await backendResponse.arrayBuffer();
    return new NextResponse(responseBody, {
      status: backendResponse.status,
      statusText: backendResponse.statusText,
      headers: responseHeaders
    });
  } catch (error: unknown) {
    return NextResponse.json({ error: (error as Error).message ?? "Proxy error" }, { status: 502 });
  }
}

export { handler as GET, handler as POST, handler as PUT, handler as PATCH, handler as DELETE };
