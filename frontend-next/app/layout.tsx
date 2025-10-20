import "@/app/globals.css";
import { Providers } from "@/components/providers";
import { cn } from "@/lib/utils";
import type { Metadata } from "next";
import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "OpenWebUI Chat Analyzer",
  description: "Next.js dashboard for exploring chat analytics backed by FastAPI."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={cn(inter.className, "min-h-screen bg-background text-foreground")}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
