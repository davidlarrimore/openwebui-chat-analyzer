"use client";

import { ReactNode, useState } from "react";
import { LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";

interface LogoutButtonProps {
  className?: string;
  variant?: "default" | "destructive" | "outline" | "secondary" | "ghost";
  size?: "default" | "sm" | "lg" | "icon";
  label?: string;
  showLabel?: boolean;
  icon?: ReactNode;
  title?: string;
}

export function LogoutButton({
  className,
  variant = "destructive",
  size = "default",
  label = "Sign out",
  showLabel = true,
  icon,
  title
}: LogoutButtonProps) {
  const [isLoggingOut, setIsLoggingOut] = useState(false);

  const handleLogout = async () => {
    setIsLoggingOut(true);
    try {
      await fetch("/api/backend/auth/logout", {
        method: "POST",
        credentials: "include",
      });
    } catch (error) {
      console.error("Failed to revoke backend session:", error);
    } finally {
      window.location.href = "/login";
    }
  };

  const fallbackIcon = <LogOut aria-hidden className="h-4 w-4 text-white" />;
  const resolvedIcon = icon ?? fallbackIcon;
  const resolvedLabel = isLoggingOut ? "Signing out..." : label;

  const shouldShowLabel = showLabel && size !== "icon";

  return (
    <Button
      className={className}
      variant={variant}
      size={size}
      onClick={handleLogout}
      disabled={isLoggingOut}
      aria-label={resolvedLabel}
      title={title}
    >
      {resolvedIcon}
      {shouldShowLabel ? (
        <span className="ml-2">{resolvedLabel}</span>
      ) : (
        <span className="sr-only">{resolvedLabel}</span>
      )}
    </Button>
  );
}
