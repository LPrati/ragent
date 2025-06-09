FROM python:3.12-slim

WORKDIR /app/agents

RUN pip install uv
COPY requirements.txt /app/
RUN uv pip install -r /app/requirements.txt

COPY . /app

CMD ["adk", "web"]
