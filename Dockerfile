# AG News Text Classification - Inference Container
FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install CPU-only PyTorch to reduce image size
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY inference.py .
COPY utils.py .
COPY id2label.json .

# Set environment variables
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Default command runs inference
ENTRYPOINT ["python", "inference.py"]
CMD ["--text", "The stock market rallied today as tech companies reported strong earnings.", "--model", "Recurrent/ag-news-distilbert"]
