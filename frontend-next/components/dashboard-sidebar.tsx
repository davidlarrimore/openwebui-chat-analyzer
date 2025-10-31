"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useState } from "react";
import SidebarHealthStatus from "@/components/sidebar-health-status";
import { LogoutButton } from "@/components/logout-button";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type NavLink = {
  href: string;
  label: string;
  icon: string;
  separator?: boolean;
};

interface DashboardSidebarProps {
  userDisplayName: string;
  navLinks: NavLink[];
}

function isNavActive(pathname: string, href: string) {
  if (pathname === href) {
    return true;
  }

  if (href === "/dashboard") {
    return false;
  }

  const normalizedHref = href.endsWith("/") ? href.slice(0, -1) : href;
  const prefix = `${normalizedHref}/`;
  return pathname.startsWith(prefix);
}

export function DashboardSidebar({ userDisplayName, navLinks }: DashboardSidebarProps) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  const toggleCollapsed = () => {
    setCollapsed((state) => !state);
  };

  return (
    <aside
      className={cn(
        "group/sidebar hidden min-h-full flex-shrink-0 border-r bg-card transition-[width] duration-300 ease-in-out lg:flex",
        collapsed ? "w-[88px]" : "w-72"
      )}
    >
      <div className="flex h-full w-full flex-col">
        <div
          className={cn(
            "relative flex items-center border-b border-border transition-all duration-300",
            collapsed ? "justify-center px-3 py-6" : "justify-between px-5 py-6"
          )}
        >
          <div className={cn("flex items-center", collapsed ? "justify-center" : "gap-3")}
            aria-label="Chat Analyzer branding"
          >
            <Image
              alt="Chat Analyzer icon"
              className="h-12 w-12 rounded-xl object-cover"
              height={48}
              priority
              src="/web-app-manifest-192x192.png"
              width={48}
            />
            {!collapsed && (
              <div className="flex flex-col">
                <span className="text-lg font-semibold leading-tight">Chat Analyzer</span>
                <span className="text-xs text-muted-foreground">Control Center</span>
              </div>
            )}
          </div>
          <Button
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-expanded={!collapsed}
            className={cn(
              "absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground transition-opacity hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-0",
              collapsed
                ? "pointer-events-none opacity-0 group-hover/sidebar:pointer-events-auto group-hover/sidebar:opacity-100 focus-visible:pointer-events-auto"
                : "opacity-100"
            )}
            onClick={toggleCollapsed}
            size="icon"
            variant="ghost"
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </Button>
        </div>

        <div className={cn("border-b border-border", collapsed ? "px-0 py-4" : "px-5 py-4")}
          aria-live="polite"
        >
          {collapsed ? (
            <div className="flex justify-center">
              <span
                aria-hidden
                className="text-2xl"
                title={`Logged in as: ${userDisplayName}`}
              >
                👤
              </span>
              <span className="sr-only">Logged in as {userDisplayName}</span>
            </div>
          ) : (
            <div>
              <p className="text-xs text-muted-foreground">Logged in as</p>
              <p className="mt-1 truncate text-sm font-medium text-foreground">{userDisplayName}</p>
            </div>
          )}
        </div>

        <nav className={cn("flex-1 overflow-y-auto", collapsed ? "px-2" : "px-4")}
          aria-label="Sidebar navigation"
        >
          <ul className="space-y-1 py-4">
            {navLinks.map((link) => {
              const active = isNavActive(pathname, link.href);

              return (
                <li key={link.href}
                  className={cn(link.separator && "pt-3")}
                >
                  {link.separator && <hr className="my-3 border-border" />}
                  <Link
                    className={cn(
                      "group flex items-center rounded-md px-3 py-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                      collapsed ? "justify-center" : "gap-3",
                      active
                        ? "bg-primary/10 text-foreground"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground"
                    )}
                    href={link.href}
                    aria-current={active ? "page" : undefined}
                    aria-label={link.label}
                    title={link.label}
                  >
                    <span aria-hidden className="text-lg leading-none">
                      {link.icon}
                    </span>
                    {!collapsed && <span className="truncate">{link.label}</span>}
                  </Link>
                  {link.href === "/dashboard/admin/connection" && (
                    <>
                      <div
                        className={cn(
                          "pt-4",
                          collapsed ? "flex justify-center" : undefined
                        )}
                      >
                        <SidebarHealthStatus collapsed={collapsed} />
                      </div>
                      <div
                        className={cn(
                          "pt-3",
                          collapsed ? "flex justify-center" : undefined
                        )}
                      >
                        <LogoutButton
                          className={cn(
                            "transition-colors",
                            collapsed ? "h-10 w-10 justify-center" : "w-full justify-start"
                          )}
                          label="Sign out"
                          showLabel={!collapsed}
                          size={collapsed ? "icon" : "default"}
                          title="Sign out"
                        />
                      </div>
                    </>
                  )}
                </li>
              );
            })}
          </ul>
        </nav>
      </div>
    </aside>
  );
}
