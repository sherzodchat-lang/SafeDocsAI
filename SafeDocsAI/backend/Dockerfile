FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    tesseract-ocr \
    tesseract-ocr-rus \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY run.py .

# Шаблоны презентаций: неизменяемый артефакт релиза, а не данные инстанса.
# Отдельным слоем и рядом с app/, потому что реестр ищет их относительно
# каталога backend (см. app/modules/presentations/templates.py). В data/ им
# нельзя: тот каталог монтируется томом и переживает выкладку.
COPY templates/ ./templates/

# Create data directory
RUN mkdir -p data/uploads data/chroma

# Run as non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

# Default command (can be overridden in docker-compose)
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8001"]
