#!/usr/bin/env python3
import json
import sys
from typing import Dict, List, Tuple, Set, Any, Optional
import math
import copy
import time
import traceback

# Константы для оптимизации
TARGET_UTILIZATION = 0.807197  # Оптимальная утилизация для максимального балла
UPPER_THRESHOLD = 0.85  # Верхний порог утилизации
LOWER_THRESHOLD = 0.25  # Нижний порог утилизации (уменьшен для более агрессивной консолидации)
MAX_MIGRATIONS = 3  # Максимальное количество миграций за раунд
MIN_BENEFIT_FOR_MIGRATION = 12.0  # Минимальная выгода для оправдания миграции (уменьшена для более активных миграций)
CONSOLIDATION_THRESHOLD = 0.25  # Порог для консолидации VM с хостов с низкой утилизацией (уменьшен)
BONUS_THRESHOLD = 5  # Количество раундов с нулевой утилизацией для получения бонуса
CACHE_STATS = {"hits": 0, "misses": 0}  # Статистика кэша

def calculate_reward(utilization: float) -> float:
    """Вычисляет награду за утилизацию хоста по формуле из документации."""
    if utilization <= 0:
        return 0
    if utilization >= 1:
        return 2.992622
    
    x = utilization
    return -0.67459 + (42.38075/(-2.5*x+5.96)) * math.exp(-2*(math.log(-2.5*x+2.96))**2)

class VMScheduler:
    def __init__(self):
        """Инициализирует планировщик VM."""
        # Константы
        self.TARGET_UTILIZATION = 0.807197  # Целевая утилизация для максимального балла
        self.UPPER_THRESHOLD = 0.85  # Верхний порог утилизации
        self.LOWER_THRESHOLD = 0.25  # Нижний порог утилизации
        self.MIN_BENEFIT_FOR_MIGRATION = 12.0  # Минимальная выгода для выполнения миграции
        self.MAX_MIGRATIONS = 3  # Максимальное число миграций за раунд
        
        # Данные о хостах и VM
        self.hosts = {}  # Информация о хостах
        self.vms = {}  # Информация о VM
        
        # Отслеживание изменений
        self.new_vms = set()  # Множество новых VM
        self.removed_vms = set()  # Множество удаленных VM
        self.previous_allocations = {}  # Предыдущие размещения
        
        # Статистика
        self.host_zero_utilization_counter = {}  # Счетчики нулевой утилизации хостов
        self.host_was_used = set()  # Множество использованных хостов
        
        # Кэш для расчетов вместимости хостов
        self.capacity_cache = {}  # dict: {cache_key: {cpu, ram}}
        self.cache_stats = {"hits": 0, "misses": 0}
        
        # Размещения VM по хостам
        self.vm_to_host_map = {}  # dict: {vm_id: host_id}
        
        # Множество хостов с предыдущими VM
        self.hosts_with_previous_vms = set()
        
        # Счетчик хостов с нулевой утилизацией
        self.host_zero_utilization_count = {}  # dict: {host_id: count}
        
        # Список хостов, готовых к отключению
        self.hosts_ready_for_shutdown = []
        
        # Список хостов для выключения
        self.hosts_to_shutdown = set()
        
        # Ошибки размещения и миграции
        self.allocation_failures = []  # list: [vm_ids]
        self.migrations = []  # list: [{vm_id, source, target}]
        
        # Отслеживание недавних миграций
        self.recent_migrations = set()  # set: {vm_ids}
        
        # Отслеживание миграций в текущем раунде
        self.current_round_migrations = set()  # set: {vm_ids}
        
        # Счетчик раундов
        self.round_counter = 0
        
        # Статистика производительности
        self.performance_stats = {
            "round_time": [],
            "migrations": []
        }
        
        # Константы для оптимизации
        self.CONSOLIDATION_THRESHOLD = CONSOLIDATION_THRESHOLD
        self.BONUS_THRESHOLD = BONUS_THRESHOLD
        
    def _get_allocation_key(self, host_id: str, allocations: Dict[str, List[str]] = None) -> str:
        """Генерирует ключ для кэширования на основе хоста и его размещений."""
        if allocations is None:
            host_vms = []
        elif isinstance(allocations, dict):
            host_vms = sorted(allocations.get(host_id, []))
        else:
            host_vms = sorted(allocations)
            
        # Добавляем характеристики хоста и VM в ключ
        key_parts = [
            f"host_{host_id}",
            f"cpu_{self.hosts[host_id]['cpu']}",
            f"ram_{self.hosts[host_id]['ram']}"
        ]
        
        for vm_id in host_vms:
            if vm_id in self.vms:
                key_parts.extend([
                    f"vm_{vm_id}",
                    f"cpu_{self.vms[vm_id]['cpu']}",
                    f"ram_{self.vms[vm_id]['ram']}"
                ])
        
        return "|".join(key_parts)
        
    def calculate_host_utilization(self, host_id, allocations):
        """Вычисляет утилизацию хоста на основе размещенных VM."""
        if host_id not in self.hosts:
            return 0.0
        
        # Собираем все VM, размещенные на этом хосте
        host_vms = []
        for vm_list in allocations.values():
            host_vms.extend(vm_list)
        
        # Вычисляем общее использование CPU и RAM
        total_cpu = sum(self.vms[vm]['cpu'] for vm in host_vms if vm in self.vms)
        total_ram = sum(self.vms[vm]['ram'] for vm in host_vms if vm in self.vms)
        
        # Вычисляем утилизацию как максимум из утилизации CPU и RAM
        cpu_utilization = total_cpu / self.hosts[host_id]['cpu']
        ram_utilization = total_ram / self.hosts[host_id]['ram']
        
        return max(cpu_utilization, ram_utilization)

    def calculate_host_capacity(self, host_id, host_vms=None):
        """Рассчитывает утилизацию хоста."""
        # Если host_vms не предоставлен, найдем все VM на этом хосте
        if host_vms is None:
            host_vms = [vm for vm, h in self.vm_to_host_map.items() if h == host_id]
        
        # Проверяем кэш
        cache_key = (host_id, tuple(sorted(host_vms)))
        if cache_key in self.capacity_cache:
            self.cache_stats["hits"] += 1
            return self.capacity_cache[cache_key]
        
        self.cache_stats["misses"] += 1
        
        if host_id not in self.hosts:
            return {'cpu_util': 0, 'ram_util': 0, 'total_cpu': 0, 'total_ram': 0, 'max_util': 0}
        
        host = self.hosts[host_id]
        
        # Рассчитываем использование ресурсов
        total_cpu = sum(self.vms[vm]['cpu'] for vm in host_vms if vm in self.vms)
        total_ram = sum(self.vms[vm]['ram'] for vm in host_vms if vm in self.vms)
        
        # Рассчитываем утилизацию
        cpu_util = total_cpu / host['cpu'] if host['cpu'] > 0 else 0
        ram_util = total_ram / host['ram'] if host['ram'] > 0 else 0
        
        # Сохраняем результат в кэш
        result = {
            'cpu_util': cpu_util,
            'ram_util': ram_util,
            'total_cpu': total_cpu,
            'total_ram': total_ram,
            'max_util': max(cpu_util, ram_util),
            'free_cpu': host['cpu'] - total_cpu,
            'free_ram': host['ram'] - total_ram
        }
        
        self.capacity_cache[cache_key] = result
        return result

    def _can_host_vm(self, host_id, vm_id, host_vms=None):
        """Проверяет, может ли хост разместить VM."""
        if host_id not in self.hosts or vm_id not in self.vms:
            return False
        
        # Если host_vms не предоставлен, найдем все VM на этом хосте
        if host_vms is None:
            host_vms = [vm for vm, h in self.vm_to_host_map.items() if h == host_id]
        elif isinstance(host_vms, dict):
            host_vms = host_vms.get(host_id, [])
        
        host = self.hosts[host_id]
        vm = self.vms[vm_id]
        
        # Получаем текущую емкость хоста
        capacity = self.calculate_host_capacity(host_id, host_vms)
        
        # Проверяем, достаточно ли ресурсов
        if capacity['free_cpu'] < vm['cpu'] or capacity['free_ram'] < vm['ram']:
            return False
            
        # Проверяем, не будет ли утилизация слишком высокой
        new_cpu_util = (capacity['total_cpu'] + vm['cpu']) / host['cpu']
        new_ram_util = (capacity['total_ram'] + vm['ram']) / host['ram']
        
        # Если утилизация будет выше 95%, это нежелательно
        if new_cpu_util > 0.95 or new_ram_util > 0.95:
            return False
        
        return True

    def clear_capacity_cache(self):
        """Очищает кэш вычисленных емкостей."""
        self.capacity_cache = {}
        # Обновляем статистику кэша
        self.performance_stats["cache_hits"] = self.cache_stats["hits"]
        self.performance_stats["cache_misses"] = self.cache_stats["misses"]
        self.cache_stats["hits"] = 0
        self.cache_stats["misses"] = 0

    def process_input(self, input_data):
        """Обрабатывает входные данные.
        
        Args:
            input_data: Словарь с входными данными
        
        Returns:
            Dict: Словарь с размещениями VM, миграциями и отказами
        """
        try:
            # Очищаем кэш в начале каждого раунда
            self.clear_capacity_cache()
            
            # Обновляем информацию о хостах и VM
            if 'hosts' in input_data:
                self.hosts = input_data['hosts']
            if 'virtual_machines' in input_data:
                self.vms = input_data['virtual_machines']
            
            # Инициализируем размещения из входных данных, если это первый раунд
            if not self.previous_allocations and 'allocations' in input_data:
                self.previous_allocations = input_data['allocations']
                # Обновляем множество использованных хостов
                for host, vms in self.previous_allocations.items():
                    if vms:
                        self.host_was_used.add(host)
                return {
                    'allocations': self.previous_allocations,
                    'migrations': [],
                    'allocation_failures': []
                }
            
            # Обрабатываем изменения (diff)
            vms_to_place = set()
            if 'diff' in input_data:
                diff = input_data['diff']
                # Добавление новых VM
                if 'add' in diff and 'virtual_machines' in diff['add']:
                    added_vms = diff['add']['virtual_machines']
                    # Проверяем, является ли added_vms словарем или списком
                    if isinstance(added_vms, dict):
                        for vm_id, vm_info in added_vms.items():
                            self.vms[vm_id] = vm_info
                            vms_to_place.add(vm_id)
                    elif isinstance(added_vms, list):
                        # Если список, просто добавляем каждую VM в множество для размещения
                        for vm_id in added_vms:
                            if vm_id in self.vms:
                                vms_to_place.add(vm_id)
                
                # Удаление VM
                if 'remove' in diff and 'virtual_machines' in diff['remove']:
                    removed_vms = diff['remove']['virtual_machines']
                    for vm_id in removed_vms:
                        if vm_id in self.vms:
                            del self.vms[vm_id]
                        # Удаляем VM из размещений
                        for host_vms in self.previous_allocations.values():
                            if vm_id in host_vms:
                                host_vms.remove(vm_id)
                
                # Обработка оптимизации и балансировки
                if ('optimize' in diff and diff['optimize'].get('target') == 'all') or \
                   ('balance' in diff and diff['balance'].get('target') == 'all'):
                    # Добавляем все VM для переразмещения
                    for host_vms in self.previous_allocations.values():
                        vms_to_place.update(host_vms)
                    # Очищаем предыдущие размещения
                    self.previous_allocations = {host: [] for host in self.hosts}
            
            # Если нет VM для размещения, возвращаем текущие размещения
            if not vms_to_place:
                return {
                    'allocations': self.previous_allocations,
                    'migrations': [],
                    'allocation_failures': []
                }
            
            # Пытаемся разместить VM
            result = self.place_vms(list(vms_to_place), list(self.hosts.keys()), self.previous_allocations)
            if result is None:
                # Если размещение не удалось, возвращаем предыдущие размещения
                return {
                    'allocations': self.previous_allocations,
                    'migrations': [],
                    'allocation_failures': list(vms_to_place)
                }
            
            # Обновляем предыдущие размещения
            self.previous_allocations = result['allocations']
            
            # Обновляем множество использованных хостов
            for host, vms in self.previous_allocations.items():
                if vms:
                    self.host_was_used.add(host)
            
            return result
            
        except Exception as e:
            # Оставляем сообщения об ошибках
            print(f"Error in process_input: {str(e)}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # В случае ошибки возвращаем предыдущие размещения
            return {
                'allocations': self.previous_allocations,
                'migrations': [],
                'allocation_failures': []
            }

    def load_data(self, data: Dict) -> None:
        """Загружает входные данные."""
        try:
            # Загружаем хосты
            if "hosts" in data:
                self.hosts = data.get("hosts", {})
                
            # Загружаем виртуальные машины
            self.vms = data.get("virtual_machines", {})
            
            # Обрабатываем изменения
            diff = data.get("diff", {})
            
            # Добавляем новые VM
            if "add" in diff:
                new_vms = diff["add"].get("virtual_machines", [])
                # Сбрасываем список неразмещенных VM при добавлении новых
                self.allocation_failures = []
                # Добавляем новые VM в список для размещения
                self.vms_to_place = new_vms
            else:
                self.vms_to_place = []
            
            # Удаляем VM
            if "remove" in diff:
                removed_vms = diff["remove"].get("virtual_machines", [])
                # Удаляем VM из предыдущих размещений
                for host_id in list(self.previous_allocations.keys()):
                    self.previous_allocations[host_id] = [
                        vm for vm in self.previous_allocations.get(host_id, [])
                        if vm not in removed_vms
                    ]
            
            # Обновляем vm_to_host_map на основе предыдущих размещений
            self.vm_to_host_map.clear()
            for host_id, vm_list in self.previous_allocations.items():
                for vm_id in vm_list:
                    if vm_id in self.vms:  # Проверяем, что VM все еще существует
                        self.vm_to_host_map[vm_id] = host_id
            
            # Обновляем множество хостов с предыдущими VM
            self.hosts_with_previous_vms = {
                host_id for host_id, vm_list in self.previous_allocations.items()
                if vm_list and host_id in self.hosts
            }
            
            # Инициализируем счетчик хостов с нулевой утилизацией при необходимости
            if not self.host_zero_utilization_count:
                self.host_zero_utilization_count = {host_id: 0 for host_id in self.hosts}
            
            # Обновляем счетчики хостов с нулевой утилизацией
            for host_id in self.hosts:
                host_vms = self.previous_allocations.get(host_id, [])
                if not host_vms:
                    self.host_zero_utilization_count[host_id] = self.host_zero_utilization_count.get(host_id, 0) + 1
                else:
                    self.host_zero_utilization_count[host_id] = 0
            
            # Обновляем список хостов для выключения
            self.update_hosts_to_shutdown()
            
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise

    def consolidate_vms(self, vm_to_host_map):
        """Консолидирует VM на минимальном количестве хостов для получения бонусов за выключенные хосты.
        
        Args:
            vm_to_host_map: Текущие размещения VM
            
        Returns:
            Tuple[Dict[str, str], List[Dict[str, str]]]: Новые размещения и список миграций
        """
        # Вызываем оптимизацию выключения хостов для консолидации VM
        return self._optimize_host_shutdown(vm_to_host_map)

    def _calculate_vm_size(self, vm_id):
        """Рассчитывает относительный размер VM."""
        if vm_id not in self.vms:
            return 0
            
        vm = self.vms[vm_id]
        cpu = vm['cpu']
        ram = vm['ram']
        
        # Нормализация относительно средних значений ресурсов всех VM
        avg_cpu = sum(v['cpu'] for v in self.vms.values()) / len(self.vms) if self.vms else 1
        avg_ram = sum(v['ram'] for v in self.vms.values()) / len(self.vms) if self.vms else 1
        
        norm_cpu = cpu / avg_cpu
        norm_ram = ram / avg_ram
        
        # Учитываем соотношение CPU/RAM
        ratio = cpu / ram if ram > 0 else float('inf')
        
        # Для VM с высоким дисбалансом даем больший вес тому ресурсу, которого больше
        if ratio > 1.5:  # CPU-интенсивная
            return (norm_cpu * 0.7 + norm_ram * 0.3)
        elif ratio < 0.7:  # RAM-интенсивная
            return (norm_cpu * 0.3 + norm_ram * 0.7)
        else:  # Сбалансированная
            return (norm_cpu * 0.5 + norm_ram * 0.5)

    def _calculate_utilization_after_migration(self, host_id, vm_id, current_vms):
        """Рассчитывает утилизацию хоста после миграции VM на него."""
        if host_id not in self.hosts or vm_id not in self.vms:
            return 1.0  # Возвращаем высокую утилизацию в случае ошибки
            
        # Рассчитываем текущую утилизацию
        host = self.hosts[host_id]
        
        # Суммарные ресурсы текущих VM
        current_cpu = sum(self.vms[vm]['cpu'] for vm in current_vms if vm in self.vms)
        current_ram = sum(self.vms[vm]['ram'] for vm in current_vms if vm in self.vms)
        
        # Добавляем ресурсы новой VM
        new_vm = self.vms[vm_id]
        total_cpu = current_cpu + new_vm['cpu']
        total_ram = current_ram + new_vm['ram']
        
        # Рассчитываем утилизацию
        cpu_util = total_cpu / host['cpu'] if host['cpu'] > 0 else 1.0
        ram_util = total_ram / host['ram'] if host['ram'] > 0 else 1.0
        
        return max(cpu_util, ram_util)

    def _optimize_host_shutdown(self, vm_to_host_map):
        """Оптимизирует выключение хостов для получения бонусов.
        
        Args:
            vm_to_host_map: Текущие размещения VM по хостам
            
        Returns:
            Tuple[Dict[str, str], List[List]]: Новые размещения и список миграций
        """
        # Выбираем хосты для выключения
        hosts_to_shutdown, host_info = self._select_hosts_for_shutdown(vm_to_host_map)
        
        # Если нет хостов для выключения, ничего не делаем
        if not hosts_to_shutdown:
            return vm_to_host_map, []
            
        # Начинаем процесс миграции VM с выбранных хостов
        migrations = []
        new_placements = vm_to_host_map.copy()
        
        # Ограничиваем количество миграций за раунд
        migrations_left = self.MAX_MIGRATIONS
        
        # Для каждого хоста, который мы хотим выключить
        successful_shutdowns = []
        
        for source_host in hosts_to_shutdown:
            if migrations_left <= 0:
                break
                
            # Получаем VM на этом хосте
            source_vms = host_info[source_host]['vms']
            
            # Если хост уже пустой, просто добавляем его в список для выключения
            if not source_vms:
                self.hosts_to_shutdown.add(source_host)
                successful_shutdowns.append(source_host)
                continue
                
            # Сортируем VM по размеру - сначала мигрируем маленькие VM
            source_vms.sort(key=lambda vm: self._calculate_vm_size(vm) if vm in self.vms else float('inf'))
            
            # Подготавливаем временные структуры для тестирования миграций
            temp_migrations = []
            temp_placements = new_placements.copy()
            
            # Пытаемся мигрировать все VM с этого хоста
            all_migrated = True
            
            for vm_id in source_vms:
                if vm_id not in self.vms:
                    continue
                    
                # Ищем лучший хост для миграции
                best_target = None
                best_score = float('-inf')
                
                # Проверяем все хосты, кроме выключаемых
                for target_host in self.hosts:
                    if target_host == source_host or target_host in hosts_to_shutdown:
                        continue
                        
                    # Собираем VM на целевом хосте
                    target_vms = [vm for vm, h in temp_placements.items() if h == target_host]
                    
                    # Проверяем, можно ли разместить VM на этом хосте
                    if not self._can_host_vm(target_host, vm_id, target_vms):
                        continue
                        
                    # Оцениваем миграцию
                    score = self._evaluate_migration_target(target_host, vm_id, target_vms)
                    
                    if score > best_score:
                        best_score = score
                        best_target = target_host
                
                # Если не нашли подходящий хост, не можем мигрировать все VM
                if best_target is None:
                    all_migrated = False
                    break
                    
                # Добавляем миграцию
                temp_migrations.append([vm_id, source_host, best_target])
                temp_placements[vm_id] = best_target
            
            # Если все VM успешно мигрированы и не превышен лимит миграций
            if all_migrated and len(temp_migrations) <= migrations_left:
                # Применяем миграции
                for migration in temp_migrations:
                    migrations.append(migration)
                
                new_placements = temp_placements
                migrations_left -= len(temp_migrations)
                
                # Добавляем хост в список выключенных
                self.hosts_to_shutdown.add(source_host)
                successful_shutdowns.append(source_host)
        
        # Комментируем вывод отладочной информации
        # if successful_shutdowns:
        #     print(f"Successfully prepared {len(successful_shutdowns)} hosts for shutdown: {successful_shutdowns}", file=sys.stderr)
            
        return new_placements, migrations
        
    def _evaluate_migration_target(self, host_id, vm_id, current_vms):
        """Оценивает целевой хост для миграции VM.
        
        Args:
            host_id: ID целевого хоста
            vm_id: ID мигрируемой VM
            current_vms: Текущие VM на хосте
            
        Returns:
            float: Оценка привлекательности миграции (выше = лучше)
        """
        if host_id not in self.hosts or vm_id not in self.vms:
            return float('-inf')
            
        # Получаем характеристики VM и хоста
        vm = self.vms[vm_id]
        host = self.hosts[host_id]
        
        # Рассчитываем утилизацию после миграции
        util_after = self._calculate_utilization_after_migration(host_id, vm_id, current_vms)
        
        # Базовая оценка
        score = 100.0
        
        # Штраф за отклонение от целевой утилизации
        util_diff = abs(util_after - self.TARGET_UTILIZATION)
        util_penalty = 80.0 * util_diff
        score -= util_penalty
        
        # Штраф за приближение к верхней границе утилизации
        if util_after > self.UPPER_THRESHOLD:
            score -= 100.0 * (util_after - self.UPPER_THRESHOLD) / (1.0 - self.UPPER_THRESHOLD)
            
        # Бонус за тип хоста, соответствующий типу VM
        vm_type = self.get_vm_type(vm_id)
        host_type = self.get_host_type(host_id)
        
        # Предпочтительные сочетания типов
        type_matches = {
            "cpu_heavy": "cpu_optimized",
            "ram_heavy": "ram_optimized",
            "storage_heavy": "storage_optimized",
            "balanced": "balanced",
            "priority": "cpu_optimized"
        }
        
        if type_matches.get(vm_type) == host_type:
            score += 30.0
            
        # Бонус за группировку похожих VM
        if current_vms:
            vm_types = [self.get_vm_type(v) for v in current_vms if v in self.vms]
            matching_types = sum(1 for t in vm_types if t == vm_type)
            if matching_types > 0:
                score += 20.0 * (matching_types / len(current_vms))
                
        return score

    def get_migrations(self, previous_vm_to_host_map, new_vm_to_host_map):
        """Определяет миграции между двумя состояниями размещения VM.
        
        Args:
            previous_vm_to_host_map: Предыдущие размещения
            new_vm_to_host_map: Новые размещения
            
        Returns:
            List[Dict[str, str]]: Список миграций
        """
        migrations = []
        
        # Находим VM, которые переместились
        for vm_id, new_host in new_vm_to_host_map.items():
            if vm_id in previous_vm_to_host_map:
                old_host = previous_vm_to_host_map[vm_id]
                if old_host != new_host:
                    migrations.append({
                        'vm': vm_id,
                        'source': old_host,
                        'target': new_host
                    })
        
        return migrations

    def place_vms(self, vms_to_place, hosts, existing_allocations=None):
        """Размещает VM на хостах.
        
        Args:
            vms_to_place (list): Список VM для размещения
            hosts (list): Список доступных хостов
            existing_allocations (dict, optional): Существующие размещения. По умолчанию None.
        
        Returns:
            dict: Словарь с размещениями, миграциями и отказами
        """
        try:
            start_time = time.time()
            
            # Если размещения не предоставлены, используем пустой словарь
            if existing_allocations is None:
                allocations = {host: [] for host in hosts}
            else:
                # Создаем копию существующих размещений
                allocations = {host: list(vms) for host, vms in existing_allocations.items()}
            
            # Сортируем VM по размеру (сначала размещаем самые большие)
            vms_to_place = sorted(vms_to_place, key=lambda vm: (
                self.vms[vm]['cpu'] + self.vms[vm]['ram'] if vm in self.vms else 0
            ), reverse=True)
            
            # Преобразуем размещения в соответствие VM -> хост
            vm_to_host_map = {}
            for host, vms in allocations.items():
                for vm in vms:
                    vm_to_host_map[vm] = host
            
            # Пытаемся разместить каждую VM
            allocation_failures = []
            
            for vm in vms_to_place:
                # Проверяем, что VM есть в списке
                if vm not in self.vms:
                    allocation_failures.append(vm)
                    continue
                
                # Получаем размер VM
                vm_cpu = self.vms[vm]['cpu']
                vm_ram = self.vms[vm]['ram']
                
                # Находим подходящий хост
                best_host = None
                best_score = float('-inf')
                
                for host in hosts:
                    # Получаем список VM на хосте
                    host_vms = allocations.get(host, [])
                    
                    # Проверяем, можно ли разместить VM на этом хосте
                    host_capacity = self.calculate_host_capacity(host, host_vms)
                    if host_capacity is None or host_capacity['free_cpu'] < vm_cpu or host_capacity['free_ram'] < vm_ram:
                        continue
                    
                    # Оцениваем размещение VM на этом хосте
                    score = self._evaluate_placement(host, vm, host_vms)
                    
                    # Выбираем хост с лучшей оценкой
                    if score > best_score:
                        best_host = host
                        best_score = score
                
                # Если найден подходящий хост, размещаем VM
                if best_host is not None:
                    # Добавляем VM в список размещений
                    if best_host not in allocations:
                        allocations[best_host] = []
                    allocations[best_host].append(vm)
                    
                    # Обновляем соответствие VM -> хост
                    vm_to_host_map[vm] = best_host
                else:
                    # Если не удалось разместить VM, добавляем в список отказов
                    allocation_failures.append(vm)
            
            # Вычисляем миграции
            migrations = self.get_migrations(
                {vm: host for vm, host in vm_to_host_map.items() if vm not in vms_to_place},
                vm_to_host_map
            )
            
            # Обновляем vm_to_host_map
            self.vm_to_host_map = vm_to_host_map
            
            # Консолидируем VM если есть такая возможность
            new_vm_to_host_map, migrations_from_consolidation = self.consolidate_vms(self.vm_to_host_map)
            
            # Если были миграции при консолидации, обновляем результат
            if migrations_from_consolidation:
                migrations.extend(migrations_from_consolidation)
            
            end_time = time.time()
            # Комментируем вывод отладочной информации
            # print(f"Place VMs execution time: {end_time - start_time:.4f} seconds", file=sys.stderr)
            
            return {
                'allocations': allocations,
                'migrations': migrations,
                'allocation_failures': allocation_failures
            }
        
        except Exception as e:
            print(f"Error in place_vms: {str(e)}", file=sys.stderr)
            return None

    def _evaluate_placement(self, host_id, vm_id, host_vms):
        """Оценивает размещение VM на хосте.
        
        Args:
            host_id: ID хоста
            vm_id: ID VM
            host_vms: Список VM на хосте
            
        Returns:
            float: Оценка размещения (выше = лучше)
        """
        if host_id not in self.hosts or vm_id not in self.vms:
            return float('-inf')
            
        # Получаем характеристики хоста и VM
        host = self.hosts[host_id]
        vm = self.vms[vm_id]
        
        # Рассчитываем утилизацию после размещения
        current_cpu = sum(self.vms[existing_vm]['cpu'] for existing_vm in host_vms if existing_vm in self.vms)
        current_ram = sum(self.vms[existing_vm]['ram'] for existing_vm in host_vms if existing_vm in self.vms)
        
        total_cpu = current_cpu + vm['cpu']
        total_ram = current_ram + vm['ram']
        
        cpu_util = total_cpu / host['cpu']
        ram_util = total_ram / host['ram']
        
        new_util = max(cpu_util, ram_util)
        
        # Базовая оценка
        score = 100.0
        
        # Штраф за отклонение от целевой утилизации
        util_diff = abs(new_util - self.TARGET_UTILIZATION)
        util_penalty = 80.0 * util_diff
        score -= util_penalty
        
        # Штраф за высокую утилизацию
        if new_util > self.UPPER_THRESHOLD:
            score -= 100.0 * (new_util - self.UPPER_THRESHOLD) / (1.0 - self.UPPER_THRESHOLD)
        
        # Бонус за консолидацию
        if host_vms:
            score += 20.0
        
        return score

    def _select_hosts_for_shutdown(self, vm_to_host_map):
        """Выбирает хосты для выключения на основе утилизации и других параметров.
        
        Args:
            vm_to_host_map: Текущие размещения VM
            
        Returns:
            Tuple[List[str], Dict]: Список хостов для выключения и информация о хостах
        """
        # Собираем информацию о хостах
        host_info = {}
        
        for host_id in self.hosts:
            host_vms = [vm for vm, h in vm_to_host_map.items() if h == host_id]
            
            # Вычисляем утилизацию хоста
            host_capacity = self.calculate_host_capacity(host_id, host_vms)
            host_utilization = host_capacity['max_util']
            
            # Добавляем информацию о хосте
            host_info[host_id] = {
                'utilization': host_utilization,
                'vms': host_vms,
                'vm_count': len(host_vms),
                'zero_count': self.host_zero_utilization_count.get(host_id, 0)
            }
            
        # Выбираем хосты с низкой утилизацией
        low_util_hosts = [h for h, info in host_info.items() 
                          if info['utilization'] < self.LOWER_THRESHOLD]
        
        if not low_util_hosts:
            return [], host_info
            
        # Подсчитываем необходимое количество хостов
        total_vm_count = sum(len(info['vms']) for info in host_info.values())
        avg_vms_per_host = total_vm_count / len(self.hosts) if self.hosts else 0
        
        # Определяем минимальное количество хостов, которые должны остаться активными
        min_active_hosts = max(1, len(self.hosts) - len(low_util_hosts))
        
        # Не отключаем все хосты - оставляем хотя бы один активным
        if len(low_util_hosts) >= len(self.hosts):
            low_util_hosts = low_util_hosts[:-1]
            
        # Оцениваем каждый хост для выключения
        host_shutdown_scores = {}
        for host_id in low_util_hosts:
            # Стоимость выключения (перемещения VM)
            shutdown_cost = sum(self._calculate_vm_size(vm) for vm in host_info[host_id]['vms'])
            
            # Выгода от выключения
            shutdown_benefit = 0
            
            # Если хост приближается к бонусному порогу, увеличиваем выгоду
            zero_count = host_info[host_id]['zero_count']
            if self.BONUS_THRESHOLD - zero_count <= 3:
                shutdown_benefit += 50.0 * (zero_count / self.BONUS_THRESHOLD)
                
            # Если хост был использован (что нужно для бонуса), это дополнительная выгода
            if host_id in self.host_was_used:
                shutdown_benefit += 30.0
                
            # Если хост почти пуст, его легче выключить
            if host_info[host_id]['vm_count'] < avg_vms_per_host / 2:
                shutdown_benefit += 20.0
                
            # Общая оценка
            host_shutdown_scores[host_id] = shutdown_benefit - shutdown_cost
            
        # Сортируем хосты по оценке выключения
        sorted_hosts = sorted(host_shutdown_scores.keys(), 
                            key=lambda h: host_shutdown_scores[h], 
                            reverse=True)
        
        # Ограничиваем количество хостов для выключения
        max_hosts_to_shutdown = max(1, len(sorted_hosts) // 2)
        candidates = sorted_hosts[:max_hosts_to_shutdown]
        
        # Отбираем хосты, для которых выгода положительна
        return [h for h in candidates if host_shutdown_scores[h] > 0], host_info

    def update_hosts_to_shutdown(self):
        """Обновляет список хостов, которые следует выключить."""
        # Получаем текущий маппинг VM -> хост
        vm_to_host_map = {}
        
        for host_id, vm_list in self.previous_allocations.items():
            for vm_id in vm_list:
                if vm_id in self.vms:
                    vm_to_host_map[vm_id] = host_id
                    
        # Определяем хосты, которые можно выключить
        hosts_to_shutdown, _ = self._select_hosts_for_shutdown(vm_to_host_map)
        
        # Обновляем список хостов для выключения
        self.hosts_to_shutdown = set(hosts_to_shutdown)
        
        # Комментируем вывод отладочной информации
        # if hosts_to_shutdown:
        #     print(f"Hosts to shutdown: {hosts_to_shutdown}", file=sys.stderr)

    def get_vm_type(self, vm_id):
        """Определяет тип VM на основе соотношения CPU/RAM.
        
        Args:
            vm_id: ID VM
            
        Returns:
            str: Тип VM (cpu_heavy, ram_heavy, balanced)
        """
        if vm_id not in self.vms:
            return "unknown"
            
        vm = self.vms[vm_id]
        cpu = vm['cpu']
        ram = vm['ram']
        
        # Определяем тип VM на основе соотношения CPU/RAM
        ratio = cpu / ram if ram > 0 else float('inf')
        
        if ratio > 1.5:
            return "cpu_heavy"
        elif ratio < 0.7:
            return "ram_heavy"
        else:
            return "balanced"

    def get_host_type(self, host_id):
        """Определяет тип хоста на основе соотношения CPU/RAM.
        
        Args:
            host_id: ID хоста
            
        Returns:
            str: Тип хоста (cpu_optimized, ram_optimized, balanced)
        """
        if host_id not in self.hosts:
            return "unknown"
            
        host = self.hosts[host_id]
        cpu = host['cpu']
        ram = host['ram']
        
        # Определяем тип хоста на основе соотношения CPU/RAM
        ratio = cpu / ram if ram > 0 else float('inf')
        
        if ratio > 1.5:
            return "cpu_optimized"
        elif ratio < 0.7:
            return "ram_optimized"
        else:
            return "balanced"

    def _log_decision(self, message, level=1):
        """
        Логирует решение планировщика.
        
        Args:
            message (str): Сообщение для логирования
            level (int): Уровень логирования (1 - информация, 2 - отладка, 3 - трассировка)
        """
        # Комментируем всю функцию логирования
        # if level <= self.log_level:
        #     print(f"[VMScheduler] {message}", file=sys.stderr)
        pass

def split_json_objects(line):
    """Split concatenated JSON objects into a list of individual JSON objects."""
    if not line:
        return []
    
    # Если line уже является объектом Python, возвращаем его как строку JSON
    if not isinstance(line, str):
        try:
            return [json.dumps(line)]
        except:
            return []
    
    # Пытаемся разделить JSON объекты
    objects = []
    start = 0
    depth = 0
    in_string = False
    escape = False
    
    for i, char in enumerate(line):
        if escape:
            escape = False
            continue
            
        if char == '\\':
            escape = True
            continue
            
        if char == '"' and not escape:
            in_string = not in_string
            continue
            
        if in_string:
            continue
            
        if char == '{':
            depth += 1
            if depth == 1:
                start = i
        elif char == '}':
            depth -= 1
            if depth == 0:
                try:
                    obj = line[start:i+1]
                    # Проверяем, что это валидный JSON
                    json.loads(obj)
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass
    
    return objects

def main():
    """Main function to process input and generate output."""
    scheduler = VMScheduler()
    
    # Обрабатываем входные данные построчно
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        try:
            # Пробуем загрузить JSON напрямую
            data = json.loads(line)
            response = scheduler.process_input(data)
            print(json.dumps(response))
            sys.stdout.flush()
        except json.JSONDecodeError:
            # Если не удалось, пробуем разделить на отдельные JSON объекты
            json_objects = split_json_objects(line)
            for json_str in json_objects:
                try:
                    data = json.loads(json_str)
                    response = scheduler.process_input(data)
                    print(json.dumps(response))
                    sys.stdout.flush()
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"Error processing round: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
        except Exception as e:
            print(f"Error processing line: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    main()