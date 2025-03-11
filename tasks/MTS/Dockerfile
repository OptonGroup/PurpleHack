FROM python:3.10-slim

WORKDIR /app

# Копируем файлы решения
COPY solution.py /app/
COPY run.sh /app/

# Делаем скрипт run.sh исполняемым
RUN chmod +x /app/run.sh

# Запускаем скрипт при старте контейнера
ENTRYPOINT ["/app/run.sh"] 