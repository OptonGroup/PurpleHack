FROM python:3.10

# Установка рабочей директории
WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание необходимых директорий
RUN mkdir -p /app/web_app/temp \
    && mkdir -p /app/web_app/static/css \
    && mkdir -p /app/web_app/static/js \
    && mkdir -p /app/web_app/templates \
    && mkdir -p /app/web_app/models

# Установка переменных среды
ENV PYTHONPATH=/app
ENV PORT=8000

# Открываем порт
EXPOSE 8000

# Запуск приложения (веб-интерфейс)
CMD ["python", "web_app/app.py"] 