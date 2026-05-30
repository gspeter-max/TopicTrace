FROM python:3.11 
RUN apt-get update && apt-get install -y curl 
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync
COPY ./src ./src
CMD ["uv", "run", "uvicorn", "topictrace.server.app:app", "--host", "0.0.0.0", "--port", "8080"] 

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1 

