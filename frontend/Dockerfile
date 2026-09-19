FROM node:20-alpine

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

EXPOSE 3000
# API_BASE (read by next.config rewrites) is provided at runtime via env.
CMD ["npm", "start"]
