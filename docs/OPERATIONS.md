# Operations and Security Checklist

- Configure explicit `RESEARCHPILOT_CORS_ORIGINS` in deployed environments.
- Put the API behind authentication and a reverse proxy before exposing it publicly.
- Add request rate limiting and per-user quotas.
- Restrict upload size and validate MIME type and file signature.
- Use external durable job storage for multi-instance deployments.
- Add structured logs with run ID, job ID, provider, latency, token usage, and error category.
- Add tracing around retrieval, enrichment, grounding, and synthesis.
- Keep secrets in environment variables or a secret manager; never commit `.env` files.
- Run dependency and container vulnerability scans in CI.
