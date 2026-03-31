FROM python:3.11-slim

LABEL maintainer="OpenEnv-CR"
LABEL description="Code Review Simulation Environment"

# Set environment variables for Python and Port
ENV PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# 1. Create a non-root user for Hugging Face (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# 2. Install dependencies (as the non-root user)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 3. Copy project files and set ownership to 'user'
COPY --chown=user . .

# Hugging Face defaults to 7860
EXPOSE 7860

# 4. Updated Healthcheck for Port 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# 5. Run Uvicorn on Port 7860
CMD ["uvicorn", "server.env:app", "--host", "0.0.0.0", "--port", "7860"]