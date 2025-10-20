import { NextResponse } from "next/server";
import { getServerAuthSession } from "@/lib/auth";

export async function GET(request: Request) {
  const session = await getServerAuthSession();
  if (!session) {
    return NextResponse.json({ authenticated: false }, { status: 401 });
  }

  return NextResponse.json(
    {
      authenticated: true,
      user: session.user,
      accessToken: session.accessToken
    },
    { status: 200 }
  );
}
