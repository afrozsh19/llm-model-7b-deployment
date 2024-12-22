FROM python:3.12-slim-bullseye AS foundation

# Install system dependencies
RUN apt-get update && apt-get install -y \
      build-essential \
      gcc \
      libssl-dev \
      zlib1g-dev \
      libbz2-dev \
      libreadline-dev \
      libsqlite3-dev \
      curl \
      llvm-9 \
      libncursesw5-dev \
      xz-utils \
      tk-dev \
      libxml2-dev \
      libxmlsec1-dev \
      libffi-dev \
      liblzma-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/llvm-config-9 /usr/bin/llvm-config

# install PDM
RUN pip install -U pip setuptools wheel
RUN pip install --no-cache-dir poetry

# Set work directory
WORKDIR /code

# Copy necessary project files
COPY pyproject.toml poetry.toml /code/

RUN poetry lock && \
    poetry install --no-root --only main

FROM python:3.12-slim-bullseye

COPY --from=foundation /usr/local /usr/local

RUN pip install -U setuptools
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


COPY --from=foundation /code /code
WORKDIR /code

COPY model_download.py /code/

ENV PATH=/code/.venv/bin:$PATH
ENV PYTHONUNBUFFERED 1
ENV DEPLOYMENT_ENVIRONMENT=dev

# Expose port for FastAPI
EXPOSE 8000

CMD ["python3", "model_download.py"]