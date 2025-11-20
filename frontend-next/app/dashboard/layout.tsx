import { ReactNode } from "react";
import { redirect } from "next/navigation";
import { DashboardSidebar } from "@/components/dashboard-sidebar";
import { LogoutButton } from "@/components/logout-button";
import { SummarizerProgressProvider } from "@/components/summarizer-progress-provider";
import { ApiError, apiGet } from "@/lib/api";
import type { SessionEnvelope } from "@/lib/types";

const NAV_LINKS = [
  { href: "/dashboard", label: "Overview", icon: "📊" },
  { href: "/dashboard/models", label: "Model Analysis", icon: "🤖" },
  { href: "/dashboard/time", label: "Time Analysis", icon: "📈" },
  { href: "/dashboard/content", label: "Content Analysis", icon: "📝" },
  { href: "/dashboard/search", label: "Search", icon: "🔍" },
  { href: "/dashboard/browse", label: "Browse Chats", icon: "💬" },
  { href: "/dashboard/admin/connection", label: "Configuration", icon: "⚙️", separator: true }
];

export default async function DashboardLayout({ children }: { children: ReactNode }) {
  let session: SessionEnvelope | null = null;
  try {
    session = await apiGet<SessionEnvelope>("/api/backend/auth/session");
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) {
      redirect("/login");
    }
    throw error;
  }

  const userDisplayName = session.user.name ?? session.user.email;

  return (
    <SummarizerProgressProvider>
      <div className="flex min-h-screen bg-background text-foreground">
        <DashboardSidebar navLinks={NAV_LINKS} userDisplayName={userDisplayName} />
        <main className="flex-1 overflow-y-auto bg-muted/20">
          <header className="flex items-center justify-between border-b bg-background px-6 py-4 lg:hidden">
            <p className="text-sm font-medium">Chat Analyzer</p>
            <LogoutButton size="sm" variant="outline" />
          </header>
          <div className="w-full px-4 py-6 sm:px-6 lg:px-8">{children}</div>
        </main>
      </div>
    </SummarizerProgressProvider>
  );
}
