"use client";

import type { Session } from "next-auth";
import { SessionProvider, useSession } from "next-auth/react";
import { usePathname } from "next/navigation";
import { Toaster } from "@/components/ui/toaster";
import { SummarizerProgressProvider } from "@/components/summarizer-progress-provider";

interface ProvidersProps {
  children: React.ReactNode;
  session?: Session | null;
}

function AuthenticatedProviders({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { status } = useSession();

  const isAuthRoute = pathname === "/login" || pathname === "/register";
  const isAuthenticated = status === "authenticated";

  if (isAuthRoute || !isAuthenticated) {
    return (
      <>
        {children}
        <Toaster />
      </>
    );
  }

  return (
    <SummarizerProgressProvider>
      {children}
      <Toaster />
    </SummarizerProgressProvider>
  );
}

export function Providers({ children, session }: ProvidersProps) {
  return (
    <SessionProvider session={session}>
      <AuthenticatedProviders>{children}</AuthenticatedProviders>
    </SessionProvider>
  );
}
