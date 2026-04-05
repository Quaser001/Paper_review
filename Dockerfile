FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch first (separate step to use the PyTorch index)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install all remaining deps from PyPI
RUN pip install --no-cache-dir sentence-transformers -r requirements.txt

COPY . .

ENV PYTHONPATH="/app"
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
