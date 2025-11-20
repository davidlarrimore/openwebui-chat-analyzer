import { redirect } from "next/navigation";
import { apiGet } from "@/lib/api";
import type { AuthStatus } from "@/lib/types";

export default async function HomePage() {
  try {
    await apiGet("/api/backend/auth/session", undefined, { skipAuthRedirect: true });
    redirect("/dashboard");
  } catch (error) {
    // intentionally ignore – fall through to status check
  }

  try {
    const status = await apiGet<AuthStatus>("/api/backend/auth/status", undefined, { skipAuthRedirect: true });
    redirect(status.has_users ? "/login" : "/register");
  } catch {
    redirect("/register");
  }
}
