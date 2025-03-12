import json
import math
import sys
from collections import defaultdict
import os

# Константы
TARGET_UTILIZATION = 0.807197  # Оптимальная утилизация для максимального балла
BONUS_THRESHOLD = 5  # Количество раундов с нулевой утилизацией для получения бонуса
SHUTDOWN_BONUS = 8.0  # Бонус за выключенный хост
ALLOCATION_FAILURE_PENALTY = -5.0  # Штраф за неразмещенную VM (умножается на количество хостов)

def calculate_reward(utilization):
    """Вычисляет награду за утилизацию хоста по формуле из документации."""
    if utilization <= 0:
        return 0
    if utilization >= 1:
        return 2.992622
    
    x = utilization
    return -0.67459 + (42.38075/(-2.5*x+5.96)) * math.exp(-2*(math.log(-2.5*x+2.96))**2)

def calculate_host_utilization(host, vms, vm_resources):
    """Рассчитывает утилизацию хоста на основе его VM."""
    if not host or not vms:
        return 0.0
    
    # Суммируем ресурсы всех VM
    total_cpu = sum(vm_resources.get(vm, {}).get('cpu', 0) for vm in vms)
    total_ram = sum(vm_resources.get(vm, {}).get('ram', 0) for vm in vms)
    
    # Рассчитываем утилизацию
    cpu_util = total_cpu / host.get('cpu', 1) if host.get('cpu', 0) > 0 else 0
    ram_util = total_ram / host.get('ram', 1) if host.get('ram', 0) > 0 else 0
    
    # Возвращаем максимальное значение утилизации
    return max(cpu_util, ram_util)

def main():
    # Параметры командной строки
    if len(sys.argv) >= 3:
        input_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
    else:
        input_file_path = "input.txt"
        output_file_path = "output.txt"
    
    print(f"Analyzing input file: {input_file_path}")
    print(f"Analyzing output file: {output_file_path}")
    
    # Проверяем существование файлов
    if not os.path.exists(input_file_path):
        print(f"Error: Input file {input_file_path} does not exist")
        return
    if not os.path.exists(output_file_path):
        print(f"Error: Output file {output_file_path} does not exist")
        return
    
    # Инициализируем переменные
    total_score = 0
    rounds_analyzed = 0
    all_scores = []
    
    # Отслеживаем состояние хостов и VM
    hosts = {}  # Информация о хостах
    vms = {}    # Информация о VM
    host_zero_util_counter = defaultdict(int)  # Счетчик нулевой утилизации
    host_was_used = set()  # Множество использованных хостов
    
    try:
        # Открываем файлы
        with open(input_file_path, 'r', encoding='utf-8') as input_file, \
             open(output_file_path, 'r', encoding='utf-8') as output_file:
            # Читаем строки из файлов
            input_lines = [line.strip() for line in input_file if line.strip()]
            output_lines = [line.strip() for line in output_file if line.strip()]
            
            # Проверяем, что количество строк ввода и вывода совпадает
            if len(input_lines) != len(output_lines):
                print(f"Warning: Number of input lines ({len(input_lines)}) does not match output lines ({len(output_lines)})")
            
            # Обрабатываем каждый раунд
            for round_num, (input_line, output_line) in enumerate(zip(input_lines, output_lines), 1):
                try:
                    # Анализируем JSON
                    input_data = json.loads(input_line)
                    output_data = json.loads(output_line)
                    
                    # Обновляем информацию о хостах (только в первом раунде)
                    if round_num == 1 and 'hosts' in input_data:
                        hosts = input_data['hosts']
                    
                    # Обновляем информацию о VM
                    if 'virtual_machines' in input_data:
                        vms.update(input_data['virtual_machines'])
                    
                    # Обрабатываем изменения (diff)
                    if 'diff' in input_data:
                        diff = input_data['diff']
                        # Добавление новых VM
                        if 'add' in diff and 'virtual_machines' in diff['add']:
                            added_vms = diff['add']['virtual_machines']
                            if isinstance(added_vms, dict):
                                vms.update(added_vms)
                        
                        # Удаление VM
                        if 'remove' in diff and 'virtual_machines' in diff['remove']:
                            removed_vms = diff['remove']['virtual_machines']
                            for vm_id in removed_vms:
                                if vm_id in vms:
                                    del vms[vm_id]
                    
                    # Получаем размещения, миграции и отказы
                    allocations = output_data.get('allocations', {})
                    migrations = output_data.get('migrations', [])
                    allocation_failures = output_data.get('allocation_failures', [])
                    
                    # Рассчитываем утилизацию каждого хоста
                    host_utils = {}
                    for host_id, host_info in hosts.items():
                        host_vms = allocations.get(host_id, [])
                        utilization = calculate_host_utilization(host_info, host_vms, vms)
                        host_utils[host_id] = utilization
                        
                        # Обновляем счетчики нулевой утилизации
                        if utilization == 0:
                            host_zero_util_counter[host_id] += 1
                        else:
                            host_zero_util_counter[host_id] = 0
                            host_was_used.add(host_id)
                    
                    # Рассчитываем оценку за раунд
                    round_score = 0
                    
                    # 1. Оценка за утилизацию хостов
                    utilization_score = 0
                    for host_id, utilization in host_utils.items():
                        host_score = calculate_reward(utilization)
                        utilization_score += host_score
                    
                    # 2. Бонусы за выключенные хосты
                    shutdown_bonus = 0
                    for host_id, zero_count in host_zero_util_counter.items():
                        if zero_count >= BONUS_THRESHOLD and host_id in host_was_used:
                            shutdown_bonus += SHUTDOWN_BONUS
                    
                    # 3. Штрафы за миграции
                    migration_penalty = -len(migrations) ** 2 if migrations else 0
                    
                    # 4. Штрафы за отказы размещения
                    allocation_failure_penalty = ALLOCATION_FAILURE_PENALTY * len(allocation_failures) * len(hosts) if allocation_failures else 0
                    
                    # Общая оценка за раунд
                    round_score = utilization_score + shutdown_bonus + migration_penalty + allocation_failure_penalty
                    
                    # Отслеживаем оценку
                    total_score += round_score
                    all_scores.append(round_score)
                    rounds_analyzed += 1
                    
                    # Выводим результаты раунда
                    print(f"Round {round_num}: {round_score:.2f} (util={utilization_score:.2f}, shutdown_bonus={shutdown_bonus:.2f}, migration={migration_penalty:.2f}, failure={allocation_failure_penalty:.2f})")
                    
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in round {round_num}: {e}")
                except Exception as e:
                    print(f"Error processing round {round_num}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Выводим итоговые результаты
            print(f"\nTotal rounds analyzed: {rounds_analyzed}")
            print(f"Total score: {total_score:.2f}")
            if rounds_analyzed > 0:
                print(f"Average score per round: {total_score / rounds_analyzed:.2f}")
            
            # Выводим статистику по группам раундов
            if rounds_analyzed > 10:
                group_size = min(10, rounds_analyzed // 10)
                for i in range(0, rounds_analyzed, group_size):
                    end = min(i + group_size, rounds_analyzed)
                    group_scores = all_scores[i:end]
                    group_total = sum(group_scores)
                    print(f"Rounds {i+1}-{end}: {group_total:.2f} (avg: {group_total/len(group_scores):.2f})")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 