FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install all Python deps in single layer
# Use CPU-only torch index to avoid CUDA bloat
RUN pip install --no-cache-dir \
    -r requirements.txt \
    torch --index-url https://download.pytorch.org/whl/cpu \
    sentence-transformers

COPY . .

ENV PYTHONPATH="/app"
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
