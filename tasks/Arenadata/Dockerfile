FROM python:3.9-slim

WORKDIR /app

# Установка утилит для отладки памяти и системных проблем
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Дополнительные зависимости для отладки памяти
RUN pip install --no-cache-dir memory_profiler psutil

# Копирование файлов проекта
COPY . .

# Создание необходимых директорий
RUN mkdir -p clients clients_showcase clients_shocases_per_time

# Настройка Python для работы с большими объемами данных
ENV PYTHONMALLOC=debug
ENV PYTHONUNBUFFERED=1
# Отключаем внутреннюю буферизацию stdout/stderr
ENV PYTHONFAULTHANDLER=1

# Порт для веб-сервера
EXPOSE 5000

# Запуск приложения
CMD ["python", "frontend_pred/server.py"] 