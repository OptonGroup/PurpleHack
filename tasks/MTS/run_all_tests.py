#!/usr/bin/env python3
import os
import subprocess
import time
import glob
import json
from datetime import datetime

def run_test(test_file, output_dir):
    """
    Запускает тест из файла test_file и сохраняет результаты в output_dir.
    """
    test_name = os.path.basename(test_file).replace('.txt', '')
    output_file = os.path.join(output_dir, f"{test_name}_output.txt")
    debug_file = os.path.join(output_dir, f"{test_name}_debug.txt")
    
    print(f"Запуск теста {test_name}...")
    start_time = time.time()
    
    try:
        # Запускаем тест и перенаправляем вывод
        with open(test_file, 'r') as input_file, \
             open(output_file, 'w') as output, \
             open(debug_file, 'w') as debug:
            
            # Запускаем процесс
            process = subprocess.Popen(
                ['python', 'solution.py'],
                stdin=input_file,
                stdout=output,
                stderr=debug,
                text=True
            )
            
            # Ждем завершения процесса с таймаутом
            try:
                process.wait(timeout=60)  # Таймаут 60 секунд
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = -1
                print(f"Тест {test_name} превысил таймаут и был прерван")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Анализируем результаты
        success = exit_code == 0
        
        # Подсчитываем количество раундов
        rounds = 0
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    rounds += 1
        
        # Подсчитываем количество миграций и ошибок размещения
        migrations = 0
        allocation_failures = 0
        
        with open(output_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    migrations += len(data.get('migrations', []))
                    allocation_failures += len(data.get('allocation_failures', []))
                except json.JSONDecodeError:
                    pass
        
        result = {
            'test_name': test_name,
            'success': success,
            'exit_code': exit_code,
            'duration': duration,
            'rounds': rounds,
            'migrations': migrations,
            'allocation_failures': allocation_failures
        }
        
        print(f"Тест {test_name} завершен за {duration:.2f} сек. Раундов: {rounds}, Миграций: {migrations}, Ошибок размещения: {allocation_failures}")
        return result
        
    except Exception as e:
        print(f"Ошибка при запуске теста {test_name}: {e}")
        return {
            'test_name': test_name,
            'success': False,
            'error': str(e)
        }

def run_all_tests(tests_dir, output_dir):
    """
    Запускает все тесты из директории tests_dir и сохраняет результаты в output_dir.
    """
    # Создаем выходную директорию, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем список всех тестовых файлов
    test_files = sorted(glob.glob(os.path.join(tests_dir, "*.txt")))
    
    if not test_files:
        print(f"Нет тестовых файлов в {tests_dir}")
        return
    
    results = []
    
    # Запускаем каждый тест
    for test_file in test_files:
        result = run_test(test_file, output_dir)
        results.append(result)
    
    # Сохраняем общие результаты
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Выводим сводку
    print("\nСводка результатов:")
    print(f"Всего тестов: {len(results)}")
    successful_tests = sum(1 for r in results if r.get('success', False))
    print(f"Успешных тестов: {successful_tests}")
    print(f"Неуспешных тестов: {len(results) - successful_tests}")
    
    # Выводим детали по каждому тесту
    print("\nДетали по тестам:")
    for result in results:
        status = "Успех" if result.get('success', False) else "Ошибка"
        test_name = result.get('test_name', 'Неизвестный')
        rounds = result.get('rounds', 0)
        migrations = result.get('migrations', 0)
        allocation_failures = result.get('allocation_failures', 0)
        duration = result.get('duration', 0)
        
        print(f"{test_name}: {status}, Раундов: {rounds}, Миграций: {migrations}, Ошибок размещения: {allocation_failures}, Время: {duration:.2f} сек.")

if __name__ == "__main__":
    # Пути к директориям
    tests_dir = "converted_tests"
    output_dir = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    run_all_tests(tests_dir, output_dir)
    print(f"\nРезультаты сохранены в директории {output_dir}") 