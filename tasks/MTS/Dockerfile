FROM python:3.10-slim

WORKDIR /app

# Копируем файлы решения
COPY solution.py /app/
COPY run.sh /app/

# Делаем скрипт run.sh исполняемым
RUN chmod +x /app/run.sh

# Устанавливаем локаль и кодировку
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONUNBUFFERED=1

# Запускаем скрипт при старте контейнера
ENTRYPOINT ["/app/run.sh"] 