FROM python:3.10-slim

WORKDIR /app

# Устанавливаем dos2unix для конвертации переносов строк
RUN apt-get update && apt-get install -y dos2unix && rm -rf /var/lib/apt/lists/*

# Копируем файлы решения
COPY solution.py /app/
COPY run.sh /app/

# Конвертируем переносы строк и делаем скрипт исполняемым
RUN dos2unix /app/run.sh && chmod +x /app/run.sh

# Устанавливаем локаль и кодировку
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# Запускаем скрипт при старте контейнера
ENTRYPOINT ["/bin/bash", "/app/run.sh"] 