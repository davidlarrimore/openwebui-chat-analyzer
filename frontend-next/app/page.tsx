import { redirect } from "next/navigation";
import { getServerAuthSession } from "@/lib/auth";
import { apiGet } from "@/lib/api";
import type { AuthStatus } from "@/lib/types";

export default async function HomePage() {
  const session = await getServerAuthSession();
  if (session) {
    redirect("/dashboard");
  }

  try {
    const status = await apiGet<AuthStatus>("api/v1/auth/status");
    redirect(status.has_users ? "/login" : "/register");
  } catch {
    redirect("/login");
  }
}
