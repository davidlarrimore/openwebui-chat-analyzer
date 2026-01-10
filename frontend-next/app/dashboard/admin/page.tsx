import { redirect } from "next/navigation";

/**
 * Admin index page - redirects to Connection tab
 */
export default function AdminPage() {
  redirect("/dashboard/admin/connection");
}
