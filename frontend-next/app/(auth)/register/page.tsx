"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { signIn } from "next-auth/react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/use-toast";
import { apiGet, apiPost } from "@/lib/api";
import type { AuthResponse, AuthStatus } from "@/lib/types";

function RegisterPageInner() {
  const router = useRouter();
  const { toast } = useToast();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState(true);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const status = await apiGet<AuthStatus>("api/v1/auth/status");
        if (status.has_users) {
          router.replace("/login");
          return;
        }
      } catch {
        // allow fallback to register form even if request fails
      } finally {
        if (active) {
          setLoadingStatus(false);
        }
      }
    })();
    return () => {
      active = false;
    };
  }, [router]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (password !== confirmPassword) {
      toast({
        title: "Passwords do not match",
        description: "Enter the same password in both fields.",
        variant: "destructive"
      });
      return;
    }

    setIsSubmitting(true);
    try {
      await apiPost<AuthResponse>("api/v1/auth/bootstrap", {
        username: email,
        password
      });

      toast({
        title: "Account created",
        description: "Signing you in…"
      });

      await signIn("credentials", {
        username: email,
        password,
        redirect: true,
        callbackUrl: "/dashboard"
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to create the account.";
      toast({
        title: "Registration failed",
        description: message,
        variant: "destructive"
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-secondary/30 px-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Create administrator</CardTitle>
          <CardDescription>Bootstrap the first user to unlock the dashboard.</CardDescription>
        </CardHeader>
        <CardContent>
          {loadingStatus ? (
            <p className="text-sm text-muted-foreground">Checking configuration…</p>
          ) : (
            <form className="space-y-4" onSubmit={handleSubmit}>
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  autoComplete="username"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  autoComplete="new-password"
                  placeholder="Create a password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="confirm-password">Confirm password</Label>
                <Input
                  id="confirm-password"
                  type="password"
                  autoComplete="new-password"
                  placeholder="Repeat the password"
                  value={confirmPassword}
                  onChange={(event) => setConfirmPassword(event.target.value)}
                  required
                />
              </div>
              <Button className="w-full" type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Creating account…" : "Create and sign in"}
              </Button>
            </form>
          )}
        </CardContent>
        {!loadingStatus && (
          <CardFooter>
            <p className="text-sm text-muted-foreground">
              Already have access?{" "}
              <Button
                variant="ghost"
                className="px-0 text-primary hover:bg-transparent"
                type="button"
                onClick={() => router.push("/login")}
              >
                Go to login
              </Button>
            </p>
          </CardFooter>
        )}
      </Card>
    </div>
  );
}

export default function RegisterPage() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center">Loading…</div>}>
      <RegisterPageInner />
    </Suspense>
  );
}
