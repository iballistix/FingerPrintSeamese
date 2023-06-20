FROM python:3.9-slim AS final

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Turn off pip cache
ENV PIP_NO_CACHE_DIR=1

# Install pip requirements
COPY requirements.txt .
# On purpose pre-install CPU version so that we don't install GPU
# GPU can be used later, it just doubles the image size
RUN python -m pip install torch==1.10.0+cpu torchvision==0.11.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN python -m pip install -r requirements.txt

WORKDIR /app

COPY *.py /app/
# Copy pre-trained models
COPY saved_models/ /app/saved_models
COPY configs/ /app/configs

# Creating a shared cache dir
RUN mkdir .cache
# Setting torch env to shared directory
ENV TORCH_HOME=/app/.cache
# Pre-cache torch model (pth)
RUN python -c 'from model import BinaryClassifier; model = BinaryClassifier()'
# Giving full permissions to this folder to everyone
RUN chmod a+rw -R .cache

RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

ENTRYPOINT ["python", "detect.py"]