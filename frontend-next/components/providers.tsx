"use client";

import type { Session } from "next-auth";
import { SessionProvider } from "next-auth/react";
import { Toaster } from "@/components/ui/toaster";

interface ProvidersProps {
  children: React.ReactNode;
  session?: Session | null;
}

export function Providers({ children, session }: ProvidersProps) {
  return (
    <SessionProvider session={session}>
      {children}
      <Toaster />
    </SessionProvider>
  );
}
