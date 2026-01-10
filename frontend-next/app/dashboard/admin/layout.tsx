"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";
import { cn } from "@/lib/utils";

interface AdminLayoutProps {
  children: React.ReactNode;
}

const adminTabs = [
  {
    name: "Connection",
    href: "/dashboard/admin/connection",
    description: "Data source and sync settings",
  },
  {
    name: "Summarizer",
    href: "/dashboard/admin/summarizer",
    description: "AI model configuration and metrics",
  },
] as const;

export default function AdminLayout({ children }: AdminLayoutProps) {
  const pathname = usePathname();

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="border-b border-border">
        <nav className="flex space-x-8" aria-label="Admin sections">
          {adminTabs.map((tab) => {
            const isActive = pathname?.startsWith(tab.href);
            return (
              <Link
                key={tab.name}
                href={tab.href}
                className={cn(
                  "group inline-flex items-center border-b-2 px-1 py-4 text-sm font-medium transition-colors",
                  isActive
                    ? "border-primary text-foreground"
                    : "border-transparent text-muted-foreground hover:border-border hover:text-foreground"
                )}
              >
                <span>{tab.name}</span>
                {!isActive && (
                  <span className="ml-2 hidden text-xs text-muted-foreground group-hover:inline">
                    {tab.description}
                  </span>
                )}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <div>{children}</div>
    </div>
  );
}
