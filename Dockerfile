FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 notebuddy \
    && useradd --uid 1000 --gid 1000 --create-home notebuddy

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY --chown=notebuddy:notebuddy . .
RUN mkdir -p /app/data/documents /app/data/chroma \
    && chown -R notebuddy:notebuddy /app/data

USER notebuddy

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "services.application.main:app", "--host", "0.0.0.0", "--port", "8000"]
