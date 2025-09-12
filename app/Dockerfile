FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx \
    && rm -rf /var/lib/apt/lists/*


RUN uv --version

WORKDIR /app

COPY pyproject.toml ./

RUN uv sync 




RUN apt-get update && apt-get install -y build-essential libpoppler-cpp-dev poppler-utils && rm -rf /var/lib/apt/lists/*


COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

