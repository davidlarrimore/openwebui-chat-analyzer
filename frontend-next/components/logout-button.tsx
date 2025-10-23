"use client";

import { signOut } from "next-auth/react";
import { Button } from "@/components/ui/button";
import { useState } from "react";

interface LogoutButtonProps {
  className?: string;
  variant?: "default" | "destructive" | "outline" | "secondary" | "ghost";
  size?: "default" | "sm" | "lg" | "icon";
}

export function LogoutButton({ className, variant = "ghost", size = "default" }: LogoutButtonProps) {
  const [isLoggingOut, setIsLoggingOut] = useState(false);

  const handleLogout = async () => {
    setIsLoggingOut(true);
    try {
      // First, revoke the backend token
      await fetch("/api/auth/logout", {
        method: "POST",
      });
    } catch (error) {
      console.error("Failed to revoke backend token:", error);
    } finally {
      // Always sign out from NextAuth, even if backend revocation fails
      await signOut({ callbackUrl: "/login" });
    }
  };

  return (
    <Button
      className={className}
      variant={variant}
      size={size}
      onClick={handleLogout}
      disabled={isLoggingOut}
    >
      {isLoggingOut ? "Signing out..." : "Sign out"}
    </Button>
  );
}
