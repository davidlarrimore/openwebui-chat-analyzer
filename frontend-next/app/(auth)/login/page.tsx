"use client";

import { Suspense, useEffect, useState, useTransition } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { signIn } from "next-auth/react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/use-toast";
import { ENABLE_GITHUB_OAUTH } from "@/lib/config";
import { apiGet } from "@/lib/api";
import type { AuthStatus } from "@/lib/types";

function LoginPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { toast } = useToast();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isPending, startTransition] = useTransition();
  const [checkingStatus, setCheckingStatus] = useState(true);

  const callbackUrl = searchParams.get("callbackUrl") ?? "/dashboard";
  const errorFromQuery = searchParams.get("error");
  const errorMessage = errorFromQuery === "SessionExpired"
    ? "Your session has expired. Please sign in again."
    : errorFromQuery;

  useEffect(() => {
    let isMounted = true;
    (async () => {
      try {
        const status = await apiGet<AuthStatus>("api/v1/auth/status");
        if (!status.has_users) {
          router.replace("/register");
          return;
        }
      } catch {
        // ignore failures and allow login form to render
      } finally {
        if (isMounted) {
          setCheckingStatus(false);
        }
      }
    })();
    return () => {
      isMounted = false;
    };
  }, [router]);

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    startTransition(async () => {
      try {
        const response = await signIn("credentials", {
          username,
          password,
          redirect: false,
          callbackUrl
        });

        if (!response) {
          throw new Error("Unexpected authentication response.");
        }

        if (response.error) {
          throw new Error(response.error);
        }

        router.push(response.url ?? callbackUrl);
        router.refresh();
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Unable to sign in with those credentials.";
        toast({
          title: "Authentication error",
          description: message,
          variant: "destructive"
        });
      }
    });
  };

  const handleGitHubSignIn = () => {
    signIn("github", { callbackUrl }).catch(() => {
      toast({
        title: "GitHub sign-in failed",
        description: "Double-check OAuth credentials and try again.",
        variant: "destructive"
      });
    });
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-secondary/30 px-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Sign in</CardTitle>
          <CardDescription>Use your credentials to access the dashboard.</CardDescription>
        </CardHeader>
        <CardContent>
          {checkingStatus ? (
            <p className="text-sm text-muted-foreground">Checking configuration…</p>
          ) : (
            <form className="space-y-4" onSubmit={handleSubmit}>
              <div className="space-y-2">
                <Label htmlFor="username">Email</Label>
                <Input
                  id="username"
                  name="username"
                  type="email"
                  autoComplete="username"
                  placeholder="you@example.com"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="current-password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  required
                />
              </div>
              <Button className="w-full" type="submit" disabled={isPending}>
                {isPending ? "Signing in..." : "Sign in"}
              </Button>
            </form>
          )}
          {errorMessage && !checkingStatus && (
            <p className="mt-4 text-sm text-destructive">{errorMessage}</p>
          )}
          {!checkingStatus && ENABLE_GITHUB_OAUTH && (
            <Button className="mt-4 w-full" onClick={handleGitHubSignIn} variant="secondary" type="button">
              Continue with GitHub
            </Button>
          )}
        </CardContent>
        {!checkingStatus && (
          <CardFooter>
            <p className="text-sm text-muted-foreground">
              First time here?{" "}
              <Link className="text-primary underline" href="/register">
                Create an account
              </Link>
            </p>
          </CardFooter>
        )}
      </Card>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center">Loading…</div>}>
      <LoginPageInner />
    </Suspense>
  );
}
