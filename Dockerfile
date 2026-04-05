FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch first (avoids 2GB CUDA download)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install sentence-transformers (depends on torch)
RUN pip install --no-cache-dir sentence-transformers>=2.7.0

# Install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy env source
COPY . .

ENV PYTHONPATH="/app"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
