import { ReactNode } from "react";
import Link from "next/link";
import { redirect } from "next/navigation";
import SidebarHealthStatus from "@/components/sidebar-health-status";
import { Button } from "@/components/ui/button";
import { getServerAuthSession } from "@/lib/auth";

const NAV_LINKS = [
  { href: "/dashboard", label: "Overview" },
  { href: "/dashboard/time", label: "Time Analysis" },
  { href: "/dashboard/content", label: "Content Analysis" },
  { href: "/dashboard/load-data", label: "Load Data" },
  { href: "/dashboard/search", label: "Search" },
  { href: "/dashboard/browse", label: "Browse Chats" },
  { href: "/dashboard/exports", label: "Exports" }
];

export default async function DashboardLayout({ children }: { children: ReactNode }) {
  const session = await getServerAuthSession();

  if (!session) {
    redirect("/login");
  }

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <aside className="hidden w-64 flex-col border-r bg-card lg:flex">
        <div className="px-6 py-8">
          <h1 className="text-xl font-semibold">Chat Analyzer</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Logged in as {session.user?.name ?? session.user?.email ?? "user"}
          </p>
        </div>
        <nav className="flex-1 px-4">
          <ul className="space-y-1">
            {NAV_LINKS.map((item) => (
              <li key={item.href}>
                <Link
                  className="block rounded-md px-3 py-2 text-sm font-medium text-muted-foreground hover:bg-muted hover:text-foreground"
                  href={item.href}
                >
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
        <div className="border-t px-4 py-6">
          <SidebarHealthStatus />
        </div>
        <div className="border-t px-4 py-6">
          <form action="/api/auth/signout" method="post">
            <Button className="w-full" type="submit" variant="ghost">
              Sign out
            </Button>
          </form>
        </div>
      </aside>
      <main className="flex-1 overflow-y-auto bg-muted/20">
        <header className="flex items-center justify-between border-b bg-background px-6 py-4 lg:hidden">
          <p className="text-sm font-medium">Chat Analyzer</p>
          <form action="/api/auth/signout" method="post">
            <Button size="sm" type="submit" variant="outline">
              Sign out
            </Button>
          </form>
        </header>
        <div className="w-full px-4 py-6 sm:px-6 lg:px-8">{children}</div>
      </main>
    </div>
  );
}
