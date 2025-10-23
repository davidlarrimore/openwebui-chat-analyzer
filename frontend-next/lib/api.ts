const ALLOWED_PATH = /^\/?api\/v1\//;

function normalise(path: string) {
  const formatted = path.startsWith("/") ? path : `/${path}`;
  if (!ALLOWED_PATH.test(formatted)) {
    throw new Error(`Disallowed API path: ${formatted}`);
  }
  return formatted;
}

async function handleJsonResponse<T>(response: Response): Promise<T> {
  const contentType = response.headers.get("content-type") ?? "";
  if (!response.ok) {
    // Handle 401 Unauthorized - redirect to login on client side
    if (response.status === 401 && typeof window !== "undefined") {
      // Clear any stale session and redirect to login
      window.location.href = "/login?error=SessionExpired";
      throw new Error("Session expired, redirecting to login...");
    }
    const message = response.statusText || `Request failed with status ${response.status}`;
    throw new Error(message);
  }
  if (contentType.includes("application/json")) {
    return (await response.json()) as T;
  }
  return {} as T;
}

export async function apiGet<T>(path: string, init?: RequestInit): Promise<T> {
  return request<T>("GET", path, undefined, init);
}

export async function apiPost<T>(path: string, body?: unknown, init?: RequestInit): Promise<T> {
  return request<T>("POST", path, body, init);
}

async function request<T>(method: string, path: string, body?: unknown, init?: RequestInit): Promise<T> {
  const normalised = normalise(path);
  const headers = new Headers(init?.headers);

  if (body !== undefined && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }

  if (typeof window === "undefined") {
    const [{ BACKEND_BASE_URL }, { getServerAuthSession }] = await Promise.all([import("./config"), import("./auth")]);
    const session = await getServerAuthSession();
    if (session?.accessToken && !headers.has("authorization")) {
      headers.set("authorization", `Bearer ${session.accessToken}`);
    }

    const response = await fetch(`${BACKEND_BASE_URL.replace(/\/$/, "")}${normalised}`, {
      ...init,
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      cache: "no-store"
    });

    return handleJsonResponse<T>(response);
  }

  const response = await fetch(`/api/proxy${normalised}`, {
    ...init,
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined
  });

  return handleJsonResponse<T>(response);
}
