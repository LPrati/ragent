FROM python:3.12-slim

WORKDIR /app

RUN pip install uv
RUN uv venv /venv
ENV PATH="/venv/bin:$PATH"
COPY requirements.txt /app/
RUN uv pip install --python=/venv/bin/python -r /app/requirements.txt

COPY . /app

CMD ["adk", "web"]
