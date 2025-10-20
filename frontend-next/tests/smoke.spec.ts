describe("API smoke test", () => {
  beforeEach(() => {
    jest.resetModules();
  });

  it("returns 401 for unauthenticated session requests", async () => {
    jest.mock("@/lib/auth", () => ({
      getServerAuthSession: jest.fn().mockResolvedValue(null)
    }));

    const { GET: sessionGet } = await import("@/app/api/session/route");
    const { getServerAuthSession } = await import("@/lib/auth");

    const response = await sessionGet(new Request("http://localhost:3000/api/session"));
    expect(response.status).toBe(401);
    expect(getServerAuthSession).toHaveBeenCalled();
  });

  it("proxies the login endpoint successfully", async () => {
    jest.mock("@/lib/auth", () => ({
      getServerAuthSession: jest.fn().mockResolvedValue({ accessToken: "token", user: { id: "1" } })
    }));

    process.env.BACKEND_BASE_URL = "http://backend.test";

    global.fetch = jest.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    );

    const { POST: proxyPost } = await import("@/app/api/proxy/[...path]/route");
    const { getServerAuthSession } = await import("@/lib/auth");

    const request = new Request("http://localhost:3000/api/proxy/api/v1/auth/login", {
      method: "POST",
      body: JSON.stringify({ username: "demo", password: "demo" }),
      headers: { "content-type": "application/json" }
    });

    const response = await proxyPost(
      Object.assign(request, {
        text: () => Promise.resolve(JSON.stringify({ username: "demo", password: "demo" }))
      }),
      { params: { path: ["api", "v1", "auth", "login"] } }
    );

    expect(response.status).toBe(200);
    expect(global.fetch).toHaveBeenCalledWith(
      "http://backend.test/api/v1/auth/login",
      expect.objectContaining({ method: "POST" })
    );
    expect(getServerAuthSession).toHaveBeenCalled();
  });
});
