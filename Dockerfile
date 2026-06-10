# AG News Text Classification - Inference Container
FROM python:3.10-slim

# Accept HF model name as build argument with sensible default
ARG MODEL_NAME="YuvarajK-g25ait2054/ag-news-distilbert"
ENV MODEL_NAME=${MODEL_NAME}

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies (PyTorch CPU version)
RUN pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers==4.46.0 scikit-learn==1.3.0 pandas==2.0.3 numpy==1.24.3

# Copy application files
COPY inference.py .
COPY utils.py .
COPY id2label.json .

# Set environment variables
ENV HF_HOME=/app/cache

# Pre-download model into image so runtime needs no internet
RUN python -c "import os; from transformers import AutoTokenizer, AutoModelForSequenceClassification; m=os.environ.get('MODEL_NAME','YuvarajK-g25ait2054/ag-news-distilbert'); print('Pre-downloading:',m); AutoTokenizer.from_pretrained(m); AutoModelForSequenceClassification.from_pretrained(m); print('Model cached!')"

# Default: run all 4 notebook texts (inference.ipynb Cell 7 demo)
ENTRYPOINT ["python", "inference.py"]
CMD ["--demo", "--model", "YuvarajK-g25ait2054/ag-news-distilbert"]
