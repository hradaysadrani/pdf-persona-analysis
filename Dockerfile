# Persona-Driven Document Intelligence
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    libfreetype6-dev \
    libharfbuzz-dev \
    libopenjp2-7-dev \
    libjbig2dec0-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CPU-only PyTorch
ENV TORCH_CPU_ONLY=1

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install PyTorch CPU-only version first
RUN pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY utils.py .
COPY persona_analyzer.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Download the sentence transformer model during build to avoid download at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Run the persona analyzer
CMD ["python", "persona_analyzer.py"] 