FROM python:3.10-slim

# Install necessary system dependencies for OpenCV and layout rendering
# Added build-essential and python3-dev to compile any C-extensions required by MinerU dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev wget curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install MinerU globally as root FIRST to avoid permission denied errors
RUN pip install --no-cache-dir --upgrade pip uv
RUN uv pip install --system -U "mineru[all]"

# Set Hugging Face specific environment variables
ENV MINERU_BACKEND="pipeline"
ENV HF_HOME="/data"
ENV PYTHONUNBUFFERED=1

# Hugging Face Spaces require running as a non-root user (switch at the end)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

EXPOSE 8000

CMD ["mineru-api", "--host", "0.0.0.0", "--port", "8000"]
