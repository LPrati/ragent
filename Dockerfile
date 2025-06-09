FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install uv
RUN uv pip install -r requirements.txt

COPY main.py /app

CMD ["uv", "run", "main.py"]