FROM python:3.12-slim

WORKDIR /app

RUN pip install uv
RUN uv venv /venv
ENV PATH="/venv/bin:$PATH"
COPY pyproject.toml /app/
RUN uv pip install -r pyproject.toml

COPY . /app

EXPOSE 8000
CMD ["adk", "web", "--host", "0.0.0.0", "--port", "8000"]
