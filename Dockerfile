FROM python:3.12-slim

WORKDIR /app

RUN pip install uv
COPY requirements.txt /app/
RUN uv venv && uv pip install -r requirements.txt

COPY main.py /app

CMD ["uv", "run", "main.py"]