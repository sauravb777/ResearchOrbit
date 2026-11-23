
FROM node:20-alpine AS frontend-build
WORKDIR /frontend
COPY frontend-next/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend-next/ ./
ARG NEXT_PUBLIC_API_URL=http://localhost:8000/api
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
RUN npm run build

FROM python:3.11-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY api ./api
COPY eval ./eval
COPY app.py ./
COPY .streamlit ./.streamlit
RUN apt-get update \
    && apt-get install -y --no-install-recommends libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
RUN mkdir -p /app/data /app/reports
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/readiness', timeout=3)"
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]

FROM node:20-alpine AS web
WORKDIR /app
ENV NODE_ENV=production
COPY --from=frontend-build /frontend/.next/standalone ./
COPY --from=frontend-build /frontend/.next/static ./.next/static
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD wget -q -O - http://127.0.0.1:3000/ >/dev/null || exit 1
CMD ["node", "server.js"]
