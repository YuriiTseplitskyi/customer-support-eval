FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY analyze/run.py ./analyze/run.py
COPY analyze/llm_council ./analyze/llm_council
COPY analyze/__init__.py ./analyze/__init__.py
COPY shared ./shared

CMD ["python", "-m", "analyze.run"]

