import "@testing-library/jest-dom";
import "whatwg-fetch";
import { TextDecoder, TextEncoder } from "util";


const globalAny = globalThis as unknown as {
  TextEncoder?: typeof TextEncoder;
  TextDecoder?: typeof TextDecoder;
  Response?: typeof Response & { json?: typeof Response.json };
};

if (!globalAny.TextEncoder) {
  globalAny.TextEncoder = TextEncoder;
}
if (!globalAny.TextDecoder) {
  globalAny.TextDecoder = TextDecoder;
}

if (globalAny.Response && typeof globalAny.Response.json !== "function") {
  globalAny.Response.json = (body: unknown, init?: ResponseInit) => {
    const payload = typeof body === "string" ? body : JSON.stringify(body);
    const headers = new Headers(init?.headers as HeadersInit);
    if (!headers.has("content-type")) {
      headers.set("content-type", "application/json");
    }
    return new Response(payload, { ...init, headers });
  };
}

jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: jest.fn(),
    replace: jest.fn(),
    refresh: jest.fn(),
    back: jest.fn(),
    forward: jest.fn(),
    prefetch: jest.fn()
  })
}));
