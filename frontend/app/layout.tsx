import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AIC 2026 — Multimedia Retrieval",
  description: "Semantic + metadata hybrid search with an agent assistant",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
