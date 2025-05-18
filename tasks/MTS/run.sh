#!/bin/bash
set -e

# Запускаем решение планировщика VM
# stdin будет перенаправлен из контейнера, результат будет выведен в stdout
exec python3 /app/solution.py 