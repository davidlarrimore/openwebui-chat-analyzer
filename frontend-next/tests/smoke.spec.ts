import type { NextRequest } from "next/server";
import { middleware } from "@/middleware";

function buildRequest(url: string): NextRequest {
  const nextUrl = new URL(url);
  const headers = new Headers();
  return {
    nextUrl,
    headers,
    cookies: {
      get: () => undefined,
      getAll: () => [],
      has: () => false
    }
  } as unknown as NextRequest;
}

describe("middleware", () => {
  beforeEach(() => {
    jest.resetModules();
  });

  it("allows dashboard access when backend session is valid", async () => {
    global.fetch = jest.fn().mockResolvedValue(new Response(null, { status: 200 }));

    const result = await middleware(buildRequest("http://localhost:3000/dashboard"));
    expect(result.cookies.getAll().length).toBe(0);
    expect(global.fetch).toHaveBeenCalledWith(new URL("/api/backend/auth/session", "http://localhost:3000"), expect.any(Object));
  });

  it("redirects to login when the session check fails", async () => {
    global.fetch = jest.fn().mockResolvedValue(new Response(null, { status: 401 }));

    const response = await middleware(buildRequest("http://localhost:3000/dashboard?tab=models"));
    expect(response?.status).toBe(307);
    expect(response?.headers.get("location")).toContain("/login");
  });
});
