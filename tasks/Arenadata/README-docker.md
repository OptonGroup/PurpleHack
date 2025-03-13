# Запуск с использованием Docker
![image](https://github.com/user-attachments/assets/2fdbf113-276d-4361-8150-c15321c618c0)


## Предварительные требования

- Docker
- Docker Compose

## Структура Docker-файлов

- `Dockerfile` - основной файл для сборки образа Docker
- `docker-compose.yml` - файл для оркестрации контейнеров
- `.dockerignore` - файл, указывающий какие файлы не включать в образ
- `requirements.txt` - список зависимостей Python

## Подготовка данных

Перед запуском контейнера необходимо подготовить данные телекома в следующей структуре:

```
./telecom100k/
  ├── data/
  │   ├── subscribers.csv
  │   ├── client.parquet
  │   ├── plan.json
  │   └── psxattrs.csv
  └── psx/
      └── (файлы с данными сессий)
```

## Запуск проекта

1. **Сборка и запуск контейнера:**

```bash
docker-compose up -d
```

2. **Проверка работы контейнера:**

```bash
docker-compose ps
```

3. **Просмотр логов:**

```bash
docker-compose logs -f
```

## Доступ к интерфейсу

После запуска контейнера веб-интерфейс будет доступен по адресу:

```
http://localhost:5000
```

## Запуск анализа данных

1. **Для запуска полного анализа данных (создание файла RESULT.csv):**

```bash
docker-compose exec telecomguardian python cluster_for_result.py
```

2. **Для создания витрин данных и отслеживания аномалий:**

```bash
docker-compose exec telecomguardian python showcase.py
```

## Остановка контейнера

```bash
docker-compose down
```

## Примечания

- Результаты анализа сохраняются в директории `./data_output` на хост-машине
- Для изменения конфигурации можно отредактировать файл `docker-compose.yml` 
