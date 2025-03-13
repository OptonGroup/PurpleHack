FROM python:3.9-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование файлов проекта
COPY . .

# Создание необходимых директорий
RUN mkdir -p clients clients_showcase clients_shocases_per_time

# Порт для веб-сервера
EXPOSE 5000

# Запуск приложения
CMD ["python", "frontend_pred/server.py"] 