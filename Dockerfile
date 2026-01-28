FROM python:3.10

WORKDIR /workspace

ENV PYTHONPATH=/workspace/src

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/
COPY app/ ./app/
COPY artifacts/ ./artifacts/

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]