/** @type {import('next').NextConfig} */
const API_BASE = process.env.API_BASE || "http://localhost:8000";

const nextConfig = {
  async rewrites() {
    // Proxy all /api/* calls to the FastAPI backend (same-origin -> no CORS,
    // and <img>/<video> can use /api/image, /api/video directly).
    return [{ source: "/api/:path*", destination: `${API_BASE}/:path*` }];
  },
};

export default nextConfig;
