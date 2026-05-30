FROM python:3.11 
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync
COPY ./src ./src
CMD ["uv", "run", "uvicorn", "topictrace.server.app:app", "--host", "0.0.0.0", "--port", "8080"] 


