#!/usr/bin/env python3
import json
import sys
from typing import Dict, List, Tuple, Set, Any, Optional
import math
import copy
import time

# Константы для оптимизации
TARGET_UTILIZATION = 0.807197  # Оптимальная утилизация для максимального балла
UPPER_THRESHOLD = 0.85  # Верхний порог утилизации (уменьшен для предотвращения перегрузки)
LOWER_THRESHOLD = 0.25  # Нижний порог утилизации (уменьшен для более агрессивной консолидации)
MAX_MIGRATIONS = 3  # Максимальное число миграций за раунд
MIN_BENEFIT_FOR_MIGRATION = 10  # Минимальная выгода для оправдания миграции (уменьшена для более активных миграций)
CONSOLIDATION_THRESHOLD = 0.25  # Порог для консолидации VM с хостов с низкой утилизацией
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
        self.hosts = {}  # имя хоста -> {cpu, ram}
        self.vms = {}  # имя VM -> {cpu, ram}
        self.allocations = {}  # имя хоста -> список VM
        self.previous_allocations = {}  # имя хоста -> список VM из предыдущего раунда
        self.host_zero_utilization_count = {}  # отслеживание хостов с нулевой нагрузкой
        self.vm_to_host_map = {}  # Отображение VM -> Host из предыдущих аллокаций
        self.hosts_with_previous_vms = set()  # Множество хостов, на которых были VM
        self.migration_history = []  # История миграций
        self.current_round_migrations = set()  # Множество VM, которые уже мигрировали в текущем раунде
        self.allocation_failures = []  # Список VM, которые не удалось разместить
        self.round_counter = 0  # Счетчик раундов
        self.host_scores = {}  # Оценки хостов по раундам
        self.total_score = 0  # Общий счет
        self.capacity_cache = {}  # Кэш для результатов calculate_host_capacity
        self.hosts_to_shutdown = set()  # Хосты, которые планируется выключить
        self.performance_stats = {  # Статистика производительности
            "round_time": [], 
            "cache_hits": [],
            "cache_misses": [],
            "migrations": [],
            "successful_migrations": [],
            "cancelled_migrations": []
        }
        
    def _get_allocation_key(self, host_id: str, allocations: Dict[str, List[str]] = None) -> str:
        """Генерирует ключ для кэширования на основе хоста и размещений."""
        if allocations is None:
            allocations = self.previous_allocations
        
        # Создаем уникальный ключ, включающий:
        # - ID хоста
        # - Список VM на хосте (отсортированный для консистентности)
        # - Характеристики всех VM на хосте (для учета изменений в VM)
        host_vms = sorted(allocations.get(host_id, []))
        vm_specs = []
        for vm_id in host_vms:
            if vm_id in self.vms:
                vm = self.vms[vm_id]
                vm_specs.append(f"{vm_id}:{vm.get('cpu', 0)}:{vm.get('ram', 0)}")
        
        return f"{host_id}|{','.join(vm_specs)}"
        
    def calculate_host_capacity(self, host_id: str, allocations: Dict[str, List[str]] = None) -> Dict:
        """Вычисляет остаточные ресурсы хоста и его утилизацию на основе текущих размещений VM."""
        # Генерируем ключ для кэша
        cache_key = self._get_allocation_key(host_id, allocations)
        
        # Проверяем кэш
        if cache_key in self.capacity_cache:
            # Статистика кэша
            CACHE_STATS["hits"] += 1
            # Возвращаем результат из кэша без копирования
            return self.capacity_cache[cache_key]
        
        # Статистика кэша
        CACHE_STATS["misses"] += 1
        
        if host_id not in self.hosts:
            result = {"capacity": {}, "utilization": {}, "max_utilization": 0}
            self.capacity_cache[cache_key] = result
            return result
        
        host = self.hosts[host_id]
        capacity = {
            "cpu": host.get("cpu", 0),
            "ram": host.get("ram", 0)
        }
        
        # Используем предоставленные аллокации или текущие
        allocs = allocations if allocations is not None else self.previous_allocations
        
        # Вычитаем ресурсы, используемые виртуальными машинами
        allocated_vms = allocs.get(host_id, [])
        used_resources = {"cpu": 0, "ram": 0}
            
        for vm_id in allocated_vms:
            if vm_id in self.vms:
                vm = self.vms[vm_id]
                used_resources["cpu"] += vm.get("cpu", 0)
                used_resources["ram"] += vm.get("ram", 0)
        
        # Вычисляем остаточные ресурсы и утилизацию
        remaining_capacity = {}
        utilization = {}
        max_utilization = 0.0
        
        for resource in ["cpu", "ram"]:
            # Оставшиеся ресурсы
            remaining_capacity[resource] = capacity[resource] - used_resources[resource]
            
            # Утилизация (используемые / общие)
            util = used_resources[resource] / host[resource] if host[resource] > 0 else 0
            utilization[resource] = util
            
            # Максимальная утилизация среди всех ресурсов
            max_utilization = max(max_utilization, util)
        
        result = {
            "capacity": remaining_capacity,
            "utilization": utilization,
            "max_utilization": max_utilization
        }
        
        # Сохраняем результат в кэш
        self.capacity_cache[cache_key] = result
        
        return result
        
    def clear_capacity_cache(self):
        """Очищает кэш вычисленных емкостей."""
        self.capacity_cache = {}
        # Обновляем статистику кэша
        self.performance_stats["cache_hits"].append(CACHE_STATS["hits"])
        self.performance_stats["cache_misses"].append(CACHE_STATS["misses"])
        CACHE_STATS["hits"] = 0
        CACHE_STATS["misses"] = 0

    def estimate_required_hosts(self) -> int:
        """Оценивает необходимое количество хостов для размещения всех VM."""
        total_vm_cpu = sum(vm.get("cpu", 0) for vm in self.vms.values())
        total_vm_ram = sum(vm.get("ram", 0) for vm in self.vms.values())
        
        # Сортируем хосты по размеру (CPU + RAM)
        sorted_hosts = sorted(
            self.hosts.items(), 
            key=lambda x: x[1].get("cpu", 0) + x[1].get("ram", 0),
            reverse=True
        )
        
        # Оцениваем, сколько хостов нужно при целевой утилизации
        required_hosts = 0
        remaining_cpu = total_vm_cpu
        remaining_ram = total_vm_ram
        
        for host_id, host in sorted_hosts:
            host_cpu = host.get("cpu", 0)
            host_ram = host.get("ram", 0)
            
            # Сколько ресурсов можем разместить на этом хосте при целевой утилизации
            usable_cpu = host_cpu * TARGET_UTILIZATION
            usable_ram = host_ram * TARGET_UTILIZATION
            
            # Если хост может вместить оставшиеся ресурсы, добавляем его
            if remaining_cpu > 0 or remaining_ram > 0:
                required_hosts += 1
                remaining_cpu -= usable_cpu
                remaining_ram -= usable_ram
            else:
                break
        
        # Добавляем 1 хост для запаса
        return min(required_hosts + 1, len(self.hosts))

    def process_input(self):
        """Обрабатывает входные данные построчно."""
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # Засекаем время начала обработки раунда
                    start_time = time.time()
                    
                    # Очищаем кэш в начале каждого раунда
                    self.clear_capacity_cache()
                    
                    # Пытаемся разобрать JSON
                    data = json.loads(line)
                    print(f"Processing input round {self.round_counter + 1}: {line[:100]}...", file=sys.stderr)
                    
                    # Если успешно, обрабатываем данные
                    self.load_data(data)
                    
                    # Размещаем VM
                    new_allocations = self.place_vms()
                    
                    # Определяем миграции
                    migrations = self.get_migrations(new_allocations)
                    
                    # Формируем ответ
                    response = {
                        "allocations": new_allocations,
                        "migrations": migrations,
                        "allocation_failures": self.allocation_failures
                    }
                    
                    # Выводим ответ
                    print(json.dumps(response))
                    sys.stdout.flush()
                    
                    # Сохраняем текущие размещения как предыдущие для следующего раунда
                    # Используем более оптимальное копирование
                    self.previous_allocations = {host_id: list(vms) for host_id, vms in new_allocations.items()}
                    
                    # Сохраняем статистику миграций
                    self.performance_stats["migrations"].append(len(migrations))
                    
                    # Засекаем время окончания обработки раунда
                    end_time = time.time()
                    self.performance_stats["round_time"].append(end_time - start_time)
                    
                    # Выводим статистику производительности
                    print(f"Round {self.round_counter + 1} completed in {end_time - start_time:.4f} sec. "
                          f"Cache: {CACHE_STATS['hits']} hits, {CACHE_STATS['misses']} misses. "
                          f"Migrations: {len(migrations)}", file=sys.stderr)
                    
                    self.round_counter += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}", file=sys.stderr)
                    continue
                except Exception as e:
                    print(f"Error processing data: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    continue
                    
        except Exception as e:
            print(f"Error in process_input: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise

    def load_data(self, data: Dict) -> None:
        """Загружает входные данные."""
        try:
            print(f"Loading data for round {self.round_counter}...", file=sys.stderr)
            
            # Загружаем хосты (только в первом раунде)
            if self.round_counter == 0:
                self.hosts = data.get("hosts", {})
                print(f"Loaded hosts: {len(self.hosts)}", file=sys.stderr)
            
            # Загружаем виртуальные машины (без глубокого копирования)
            self.vms = data.get("virtual_machines", {})
            print(f"Loaded VMs: {len(self.vms)}", file=sys.stderr)
            
            # Обрабатываем изменения
            diff = data.get("diff", {})
            print(f"Processing diff: {diff}", file=sys.stderr)
            
            # Добавляем новые VM
            if "add" in diff:
                new_vms = diff["add"].get("virtual_machines", [])
                print(f"Added VMs: {new_vms}", file=sys.stderr)
                # Сбрасываем список неразмещенных VM при добавлении новых
                self.allocation_failures = []
            
            # Удаляем VM
            if "remove" in diff:
                removed_vms = diff["remove"].get("virtual_machines", [])
                print(f"Removed VMs: {removed_vms}", file=sys.stderr)
                
                # Удаляем VM из предыдущих размещений (оптимизировано)
                for host_id in self.previous_allocations:
                    vms = self.previous_allocations[host_id]
                    self.previous_allocations[host_id] = [vm for vm in vms if vm not in removed_vms]
            
            # Обновляем vm_to_host_map на основе предыдущих размещений (без лишних копирований)
            self.vm_to_host_map.clear()
            for host_id, vm_list in self.previous_allocations.items():
                for vm_id in vm_list:
                    if vm_id in self.vms:  # Проверяем, что VM все еще существует
                        self.vm_to_host_map[vm_id] = host_id
            print(f"Updated VM to host map: {len(self.vm_to_host_map)} mappings", file=sys.stderr)
            
            # Обновляем множество хостов с предыдущими VM
            self.hosts_with_previous_vms = {
                host_id for host_id, vm_list in self.previous_allocations.items()
                if vm_list and host_id in self.hosts
            }
            print(f"Updated hosts with previous VMs: {len(self.hosts_with_previous_vms)} hosts", file=sys.stderr)
            
            # Инициализируем счетчик хостов с нулевой утилизацией при необходимости
            if not self.host_zero_utilization_count:
                self.host_zero_utilization_count = {host_id: 0 for host_id in self.hosts}
                print(f"Initialized zero utilization counters for {len(self.hosts)} hosts", file=sys.stderr)
            
            # Обновляем счетчики хостов с нулевой утилизацией
            for host_id in self.hosts:
                host_vms = self.previous_allocations.get(host_id, [])
                if not host_vms:
                    self.host_zero_utilization_count[host_id] = self.host_zero_utilization_count.get(host_id, 0) + 1
                    print(f"Host {host_id} has zero utilization for {self.host_zero_utilization_count[host_id]} rounds", file=sys.stderr)
                else:
                    self.host_zero_utilization_count[host_id] = 0
            
            # Обновляем список хостов для выключения
            self.update_hosts_to_shutdown()
            
        except Exception as e:
            print(f"Error in load_data: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
            
    def update_hosts_to_shutdown(self):
        """Обновляет список хостов, которые следует выключить для получения бонусов."""
        # Оцениваем необходимое количество хостов
        required_hosts = self.estimate_required_hosts()
        print(f"Estimated required hosts: {required_hosts} out of {len(self.hosts)}", file=sys.stderr)
        
        # Хосты, которые уже близки к получению бонуса (4+ раунда с нулевой утилизацией)
        almost_bonus_hosts = {
            host_id for host_id, count in self.host_zero_utilization_count.items()
            if count >= 4 and host_id in self.hosts_with_previous_vms
        }
        print(f"Hosts close to bonus: {almost_bonus_hosts}", file=sys.stderr)
        
        # Хосты с низкой утилизацией, которые можно выключить
        low_utilization_hosts = []
        for host_id in self.hosts:
            if host_id in self.previous_allocations and self.previous_allocations[host_id]:
                state = self.calculate_host_capacity(host_id)
                if state["max_utilization"] < LOWER_THRESHOLD:
                    low_utilization_hosts.append((host_id, state["max_utilization"]))
        
        # Сортируем по утилизации (сначала самые низкие)
        low_utilization_hosts.sort(key=lambda x: x[1])
        
        # Определяем, сколько хостов можно выключить
        hosts_to_shutdown_count = max(0, len(self.hosts) - required_hosts)
        
        # Приоритет для выключения:
        # 1. Хосты, близкие к получению бонуса
        # 2. Хосты с самой низкой утилизацией
        self.hosts_to_shutdown = set()
        
        # Добавляем хосты, близкие к бонусу
        self.hosts_to_shutdown.update(almost_bonus_hosts)
        
        # Добавляем хосты с низкой утилизацией, если нужно больше
        for host_id, util in low_utilization_hosts:
            if len(self.hosts_to_shutdown) < hosts_to_shutdown_count and host_id not in self.hosts_to_shutdown:
                self.hosts_to_shutdown.add(host_id)
        
        print(f"Hosts to shutdown: {self.hosts_to_shutdown}", file=sys.stderr)

    def can_host_vm(self, host_id: str, vm_id: str, existing_capacity: Optional[Dict] = None) -> bool:
        """Проверяет, может ли хост разместить указанную VM."""
        if vm_id not in self.vms or host_id not in self.hosts:
            return False
        
        vm = self.vms[vm_id]
        
        # Получаем текущую емкость хоста
        if existing_capacity:
            # Используем переданную емкость
            capacity = existing_capacity
        else:
            # Вычисляем текущую емкость
            host_state = self.calculate_host_capacity(host_id)
            capacity = host_state["capacity"]
        
        # Проверяем, достаточно ли ресурсов
        return (capacity["cpu"] >= vm.get("cpu", 0) and 
                capacity["ram"] >= vm.get("ram", 0))

    def get_best_host_for_vm(self, vm_id: str, target_utilization: float = TARGET_UTILIZATION, avoid_hosts: Set[str] = None) -> Optional[str]:
        """Находит лучший хост для размещения VM с учетом целевой утилизации."""
        if vm_id not in self.vms:
            return None
        
        vm = self.vms[vm_id]
        best_host = None
        best_score = float('-inf')
        
        # Хосты, которых следует избегать (например, планируемые к выключению)
        if avoid_hosts is None:
            avoid_hosts = set()
        
        # Получаем текущее состояние всех хостов
        host_states = {
            host_id: self.calculate_host_capacity(host_id)
            for host_id in self.hosts
        }
        
        # Оцениваем каждый хост
        for host_id, state in host_states.items():
            # Пропускаем хосты, которых следует избегать
            if host_id in avoid_hosts:
                continue
                
            # Проверяем базовые ограничения ресурсов
            if not self.can_host_vm(host_id, vm_id, state["capacity"]):
                continue
            
            # Вычисляем новую утилизацию после размещения VM
            host = self.hosts[host_id]
            new_cpu_utilization = ((state["utilization"]["cpu"] * host["cpu"]) + vm.get("cpu", 0)) / host["cpu"]
            new_ram_utilization = ((state["utilization"]["ram"] * host["ram"]) + vm.get("ram", 0)) / host["ram"]
            new_utilization = max(new_cpu_utilization, new_ram_utilization)
            
            # Вычисляем оценку для данного хоста (насколько близко к целевой утилизации)
            score = -abs(new_utilization - target_utilization)
            
            # Учитываем соотношение CPU/RAM
            vm_ratio = vm.get("cpu", 0) / max(vm.get("ram", 1), 1)
            host_remaining_ratio = state["capacity"]["cpu"] / max(state["capacity"]["ram"], 1)
            ratio_similarity = -abs(vm_ratio - host_remaining_ratio)
            
            # Бонус для хостов, которые уже используются
            usage_bonus = 0
            if len(self.previous_allocations.get(host_id, [])) > 0:
                usage_bonus = 0.2
            
            # Штраф для хостов с высокой утилизацией
            utilization_penalty = 0
            if new_utilization > UPPER_THRESHOLD:
                utilization_penalty = (new_utilization - UPPER_THRESHOLD) * 5
            
            # Комбинируем оценки
            combined_score = score + 0.1 * ratio_similarity + usage_bonus - utilization_penalty
            
            # Если текущий хост лучше предыдущего лучшего, обновляем
            if combined_score > best_score:
                best_score = combined_score
                best_host = host_id
        
        return best_host

    def select_hosts_for_shutdown(self, allocations: Dict[str, List[str]]) -> None:
        """Выбирает хосты для выключения на основе их утилизации и истории."""
        # Очищаем предыдущий список хостов для выключения
        self.hosts_to_shutdown.clear()
        
        # Получаем текущую утилизацию всех хостов
        hosts_utilization = {
            host_id: self.calculate_host_capacity(host_id, allocations)["max_utilization"]
            for host_id in self.hosts
        }
        
        # Оцениваем необходимое количество хостов
        required_hosts = self.estimate_required_hosts()
        active_hosts = sum(1 for util in hosts_utilization.values() if util > 0)
        
        print(f"Selecting hosts for shutdown. Required: {required_hosts}, Active: {active_hosts}", file=sys.stderr)
        
        # Если активных хостов меньше необходимого, не выключаем хосты
        if active_hosts <= required_hosts:
            print("Not enough active hosts for shutdown", file=sys.stderr)
            return
        
        # Сортируем хосты по приоритету выключения
        host_shutdown_priority = []
        for host_id, utilization in hosts_utilization.items():
            if utilization == 0:
                continue  # Пропускаем уже выключенные хосты
            
            # Вычисляем приоритет выключения (больше = выше приоритет)
            priority_score = 0
            
            # Низкая утилизация увеличивает приоритет
            if utilization < CONSOLIDATION_THRESHOLD:
                priority_score += (CONSOLIDATION_THRESHOLD - utilization) * 15
            
            # Близость к получению бонуса увеличивает приоритет
            zero_count = self.host_zero_utilization_count.get(host_id, 0)
            if zero_count > 0:
                # Резко увеличиваем приоритет для хостов, близких к бонусу
                if zero_count >= BONUS_THRESHOLD - 1:
                    priority_score += 30
                else:
                    priority_score += zero_count * 2
            
            # Меньшее количество VM увеличивает приоритет
            vm_count = len(allocations.get(host_id, []))
            priority_score += (1.0 / (vm_count + 1)) * 10
            
            # Бонус для хостов, которые уже использовались и могут получить бонус
            if host_id in self.hosts_with_previous_vms:
                priority_score += 5
            
            host_shutdown_priority.append((host_id, priority_score, vm_count, utilization))
        
        # Сортируем по приоритету (больший приоритет = раньше выключаем)
        host_shutdown_priority.sort(key=lambda x: (x[1], 1.0/(x[2]+1)), reverse=True)
        
        # Выбираем хосты для выключения
        hosts_to_shutdown_count = min(
            active_hosts - required_hosts,  # Не выключаем больше, чем можем
            3  # Ограничиваем количеством возможных миграций
        )
        
        print(f"Host shutdown candidates (priority, vm_count, utilization):", file=sys.stderr)
        for host_id, priority, vm_count, utilization in host_shutdown_priority[:5]:
            print(f"  {host_id}: priority={priority:.2f}, vm_count={vm_count}, util={utilization:.2f}", file=sys.stderr)
        
        for host_id, priority, vm_count, utilization in host_shutdown_priority[:hosts_to_shutdown_count]:
            self.hosts_to_shutdown.add(host_id)
            print(f"Selected host {host_id} for shutdown (priority: {priority:.2f}, VMs: {vm_count}, util: {utilization:.2f})", file=sys.stderr)
        
        # Если есть хосты, близкие к получению бонуса, но они не попали в список,
        # добавляем их принудительно (если это не сильно превышает ограничения)
        for host_id, priority, vm_count, utilization in host_shutdown_priority:
            if host_id in self.hosts_to_shutdown:
                continue
                
            zero_count = self.host_zero_utilization_count.get(host_id, 0)
            if zero_count >= BONUS_THRESHOLD - 1 and host_id in self.hosts_with_previous_vms:
                self.hosts_to_shutdown.add(host_id)
                print(f"Adding host {host_id} to shutdown list because it's close to bonus (zero count: {zero_count})", file=sys.stderr)

    def place_vms(self) -> Dict[str, List[str]]:
        """Размещает VM на хостах."""
        print(f"\nStarting VM placement for round {self.round_counter}...", file=sys.stderr)
        
        # Инициализируем новые размещения на основе предыдущих (без deep copy)
        new_allocations = {host_id: list(vms) for host_id, vms in self.previous_allocations.items()}
        for host_id in self.hosts:
            if host_id not in new_allocations:
                new_allocations[host_id] = []
        
        # Получаем утилизацию хостов
        hosts_utilization = {
            host_id: self.calculate_host_capacity(host_id, new_allocations)["max_utilization"]
            for host_id in self.hosts
        }
        
        # Сортируем хосты по утилизации (для более равномерного распределения)
        sorted_hosts = sorted(
            [(host_id, hosts_utilization[host_id]) for host_id in self.hosts],
            key=lambda x: x[1]  # Сортировка по утилизации (по возрастанию)
        )
        
        # Выводим информацию о текущей утилизации
        print(f"Current host utilization: {hosts_utilization}", file=sys.stderr)
        
        # Оцениваем необходимое количество хостов
        required_hosts = self.estimate_required_hosts()
        print(f"Required hosts for placement: {required_hosts}", file=sys.stderr)
        
        # Сортируем VM по размеру и соотношению CPU/RAM
        vm_metrics = {}
        for vm_id, vm in self.vms.items():
            size = vm.get("cpu", 0) + vm.get("ram", 0)
            ratio = vm.get("cpu", 0) / max(vm.get("ram", 1), 1)
            vm_metrics[vm_id] = (size, ratio)
            
        # Группируем VM по размеру для более равномерного распределения
        size_groups = {}
        for vm_id in self.vms:
            size = vm_metrics[vm_id][0]
            group = size // 100  # Группируем по диапазонам по 100 единиц
            if group not in size_groups:
                size_groups[group] = []
            size_groups[group].append(vm_id)
            
        # Сортируем VM в каждой группе по соотношению CPU/RAM
        for group in size_groups.values():
            group.sort(key=lambda vm_id: vm_metrics[vm_id][1])
            
        # Собираем отсортированный список VM (сначала большие VM)
        sorted_vms = []
        for group in sorted(size_groups.keys(), reverse=True):
            sorted_vms.extend(size_groups[group])
        
        # Находим неразмещенные VM
        unplaced_vms = []
        
        # Сначала пытаемся разместить VM на их текущих хостах
        for vm_id in sorted_vms:
            current_host = self.vm_to_host_map.get(vm_id)
            
            # Если текущий хост планируется выключить, добавляем VM в список неразмещенных
            if current_host in self.hosts_to_shutdown:
                unplaced_vms.append(vm_id)
                continue
            
            if current_host and current_host in self.hosts:
                # Проверяем, можем ли оставить VM на текущем хосте
                if self.can_host_vm(current_host, vm_id):
                    if vm_id not in new_allocations[current_host]:
                        new_allocations[current_host].append(vm_id)
                else:
                    unplaced_vms.append(vm_id)
            else:
                unplaced_vms.append(vm_id)
                
        print(f"Unplaced VMs after initial placement: {len(unplaced_vms)}", file=sys.stderr)
        
        # Стратегия размещения неразмещенных VM:
        # 1. Рассчитываем целевое количество активных хостов
        # 2. Предпочитаем размещать на уже активных хостах
        # 3. Предпочитаем более крупные хосты для больших VM
        
        # Рассчитываем общий размер неразмещенных VM
        total_unplaced_size = sum(
            self.vms.get(vm_id, {}).get("cpu", 0) + self.vms.get(vm_id, {}).get("ram", 0)
            for vm_id in unplaced_vms
        )
        
        print(f"Total size of unplaced VMs: {total_unplaced_size}", file=sys.stderr)
        
        # Обновляем утилизацию хостов после первого размещения
        for host_id in self.hosts:
            hosts_utilization[host_id] = self.calculate_host_capacity(host_id, new_allocations)["max_utilization"]
        
        # Если у нас есть неразмещенные VM, попробуем сначала освободить место,
        # мигрируя существующие VM на другие хосты
        if unplaced_vms:  # Убираем ограничение на количество неразмещенных VM
            print(f"Attempting to make room for {len(unplaced_vms)} unplaced VMs by migrating existing ones", file=sys.stderr)
            
            # Группируем неразмещенные VM по размеру
            unplaced_by_size = sorted(
                [(vm_id, self.vms[vm_id].get("cpu", 0) + self.vms[vm_id].get("ram", 0)) for vm_id in unplaced_vms],
                key=lambda x: x[1],
                reverse=True  # Сортируем по убыванию размера
            )
            
            # Для каждой неразмещенной VM пытаемся освободить место
            successful_emergency_migrations = 0
            max_emergency_migrations = min(len(unplaced_vms) * 2, 10)  # Увеличиваем лимит экстренных миграций
            
            for vm_id, vm_size in unplaced_by_size:
                if successful_emergency_migrations >= max_emergency_migrations:
                    print(f"Reached maximum emergency migrations limit ({max_emergency_migrations})", file=sys.stderr)
                    break
                    
                # Получаем требования VM
                vm = self.vms[vm_id]
                required_cpu = vm.get("cpu", 0)
                required_ram = vm.get("ram", 0)
                
                # Находим хосты, которые почти могут разместить VM (нехватка менее 30%)
                potential_hosts = []
                for host_id in self.hosts:
                    if host_id in self.hosts_to_shutdown:
                        continue
                        
                    host_capacity = self.calculate_host_capacity(host_id, new_allocations)["capacity"]
                    cpu_deficit = max(0, required_cpu - host_capacity["cpu"])
                    ram_deficit = max(0, required_ram - host_capacity["ram"])
                    
                    # Проверяем, насколько не хватает ресурсов (относительно)
                    cpu_deficit_percent = cpu_deficit / required_cpu if required_cpu > 0 else 0
                    ram_deficit_percent = ram_deficit / required_ram if required_ram > 0 else 0
                    
                    # Если ресурсов почти хватает, добавляем хост в потенциальные
                    if cpu_deficit_percent <= 0.3 and ram_deficit_percent <= 0.3:  # Увеличиваем порог до 30%
                        # Вычисляем общий дефицит в процентах
                        total_deficit_percent = (cpu_deficit_percent + ram_deficit_percent) / 2
                        potential_hosts.append((host_id, cpu_deficit, ram_deficit, total_deficit_percent))
                
                # Сортируем хосты по дефициту ресурсов (сначала с наименьшим дефицитом)
                potential_hosts.sort(key=lambda x: x[3])
                
                # Пытаемся освободить место на потенциальных хостах через миграцию
                found_place = False
                
                for host_id, cpu_deficit, ram_deficit, _ in potential_hosts:
                    # Если дефицита нет, просто размещаем
                    if cpu_deficit == 0 and ram_deficit == 0:
                        if host_id not in new_allocations:
                            new_allocations[host_id] = []
                        new_allocations[host_id].append(vm_id)
                        found_place = True
                        print(f"Placed unplaced VM {vm_id} on host {host_id} (no deficit)", file=sys.stderr)
                        break
                    
                    # Если есть дефицит, ищем VM для миграции
                    host_vms = new_allocations.get(host_id, [])
                    
                    # Ищем VM, которые можно мигрировать, чтобы освободить место
                    migratable_vms = []
                    for existing_vm_id in host_vms:
                        # Пропускаем VM, которые уже мигрировали
                        if existing_vm_id in self.current_round_migrations:
                            continue
                            
                        existing_vm = self.vms.get(existing_vm_id, {})
                        existing_vm_cpu = existing_vm.get("cpu", 0)
                        existing_vm_ram = existing_vm.get("ram", 0)
                        
                        # Проверяем, поможет ли миграция этой VM решить проблему дефицита
                        if (existing_vm_cpu >= cpu_deficit * 0.8 or existing_vm_ram >= ram_deficit * 0.8):  # Снижаем требования
                            migratable_vms.append((
                                existing_vm_id, 
                                existing_vm_cpu, 
                                existing_vm_ram,
                                existing_vm_cpu + existing_vm_ram  # Общий размер VM
                            ))
                    
                    # Сортируем VM по размеру (сначала самые маленькие, которые решат проблему)
                    migratable_vms.sort(key=lambda x: x[3])
                    
                    # Пытаемся мигрировать VM, чтобы освободить место
                    for existing_vm_id, vm_cpu, vm_ram, _ in migratable_vms:
                        # Ищем новый хост для этой VM
                        new_host = None
                        for other_host_id in self.hosts:
                            if other_host_id == host_id or other_host_id in self.hosts_to_shutdown:
                                continue
                                
                            # Проверяем, может ли другой хост принять эту VM
                            if self.can_host_vm(other_host_id, existing_vm_id):
                                new_host = other_host_id
                                break
                        
                        if new_host:
                            # Мигрируем VM
                            new_allocations[host_id].remove(existing_vm_id)
                            if new_host not in new_allocations:
                                new_allocations[new_host] = []
                            new_allocations[new_host].append(existing_vm_id)
                            
                            # Добавляем в список миграций
                            self.current_round_migrations.add(existing_vm_id)
                            successful_emergency_migrations += 1
                            
                            print(f"Emergency migration: VM {existing_vm_id} from {host_id} to {new_host} to make room for {vm_id}", file=sys.stderr)
                            
                            # Проверяем, хватит ли теперь места для неразмещенной VM
                            if self.can_host_vm(host_id, vm_id, self.calculate_host_capacity(host_id, new_allocations)["capacity"]):
                                # Размещаем неразмещенную VM
                                new_allocations[host_id].append(vm_id)
                                found_place = True
                                print(f"Placed unplaced VM {vm_id} on host {host_id} after emergency migration", file=sys.stderr)
                                break
                    
                    if found_place:
                        break
                    
                    # Если не удалось освободить место через миграцию одной VM, 
                    # попробуем каскадную миграцию нескольких VM
                    if not found_place and len(migratable_vms) >= 2:
                        print(f"Attempting cascade migration for VM {vm_id} on host {host_id}", file=sys.stderr)
                        
                        # Пытаемся мигрировать несколько VM, чтобы освободить место
                        migrated_vms = []
                        total_freed_cpu = 0
                        total_freed_ram = 0
                        
                        for existing_vm_id, vm_cpu, vm_ram, _ in migratable_vms:
                            if total_freed_cpu >= cpu_deficit and total_freed_ram >= ram_deficit:
                                break
                                
                            # Ищем новый хост для этой VM
                            new_host = None
                            for other_host_id in self.hosts:
                                if other_host_id == host_id or other_host_id in self.hosts_to_shutdown:
                                    continue
                                    
                                # Проверяем, может ли другой хост принять эту VM
                                if self.can_host_vm(other_host_id, existing_vm_id):
                                    new_host = other_host_id
                                    break
                            
                            if new_host:
                                # Мигрируем VM
                                new_allocations[host_id].remove(existing_vm_id)
                                if new_host not in new_allocations:
                                    new_allocations[new_host] = []
                                new_allocations[new_host].append(existing_vm_id)
                                
                                # Добавляем в список миграций
                                self.current_round_migrations.add(existing_vm_id)
                                successful_emergency_migrations += 1
                                migrated_vms.append(existing_vm_id)
                                
                                # Учитываем освобожденные ресурсы
                                total_freed_cpu += vm_cpu
                                total_freed_ram += vm_ram
                                
                                print(f"Cascade migration: VM {existing_vm_id} from {host_id} to {new_host}", file=sys.stderr)
                        
                        # Проверяем, хватит ли теперь места для неразмещенной VM
                        if self.can_host_vm(host_id, vm_id, self.calculate_host_capacity(host_id, new_allocations)["capacity"]):
                            # Размещаем неразмещенную VM
                            new_allocations[host_id].append(vm_id)
                            found_place = True
                            print(f"Placed unplaced VM {vm_id} on host {host_id} after cascade migration of {len(migrated_vms)} VMs", file=sys.stderr)
                            break
                        else:
                            # Если все равно не хватает места, отменяем каскадные миграции
                            print(f"Cascade migration failed for VM {vm_id}, reverting {len(migrated_vms)} migrations", file=sys.stderr)
                            
                            # Отменяем миграции
                            for migrated_vm_id in migrated_vms:
                                # Находим текущий хост мигрированной VM
                                current_host = None
                                for h, vms in new_allocations.items():
                                    if migrated_vm_id in vms:
                                        current_host = h
                                        break
                                
                                if current_host:
                                    # Возвращаем VM на исходный хост
                                    new_allocations[current_host].remove(migrated_vm_id)
                                    new_allocations[host_id].append(migrated_vm_id)
                                    
                                    # Удаляем из списка миграций
                                    self.current_round_migrations.remove(migrated_vm_id)
                                    successful_emergency_migrations -= 1
                
                # Если удалось разместить VM, убираем ее из списка неразмещенных
                if found_place:
                    unplaced_vms.remove(vm_id)
            
            # Обновляем утилизацию хостов после экстренных миграций
            if successful_emergency_migrations > 0:
                for host_id in self.hosts:
                    hosts_utilization[host_id] = self.calculate_host_capacity(host_id, new_allocations)["max_utilization"]
                print(f"Performed {successful_emergency_migrations} emergency migrations to make room for unplaced VMs", file=sys.stderr)
                print(f"Remaining unplaced VMs after emergency migrations: {len(unplaced_vms)}", file=sys.stderr)
        
        # Сортируем хосты по потенциалу для размещения (предпочитаем активные хосты с низкой утилизацией)
        placement_hosts = []
        for host_id in self.hosts:
            utilization = hosts_utilization[host_id]
            is_active = utilization > 0
            is_shutdown = host_id in self.hosts_to_shutdown
            host_size = self.hosts[host_id].get("cpu", 0) + self.hosts[host_id].get("ram", 0)
            
            # Пропускаем хосты для выключения
            if is_shutdown:
                continue
                
            # Рассчитываем приоритет размещения (больше = выше приоритет)
            placement_priority = 0
            
            # Предпочитаем активные хосты
            if is_active:
                placement_priority += 10
                
            # Предпочитаем хосты с утилизацией ниже целевой
            if utilization < TARGET_UTILIZATION:
                placement_priority += (TARGET_UTILIZATION - utilization) * 5
            else:
                # Штраф для хостов с высокой утилизацией
                placement_priority -= (utilization - TARGET_UTILIZATION) * 10
            
            # Бонус за размер хоста
            placement_priority += (host_size / 1000)
            
            placement_hosts.append((host_id, placement_priority, utilization, host_size))
            
        # Сортируем хосты по приоритету размещения (по убыванию)
        placement_hosts.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Placement hosts (priority, utilization, size): {placement_hosts}", file=sys.stderr)
        
        # Пытаемся размещать VM одну за другой
        placement_failures = []
        
        for vm_id in unplaced_vms:
            vm = self.vms.get(vm_id, {})
            vm_size = vm.get("cpu", 0) + vm.get("ram", 0)
            vm_ratio = vm.get("cpu", 0) / max(vm.get("ram", 1), 1)
            
            best_host = None
            best_score = float('-inf')
            
            for host_id, priority, utilization, host_size in placement_hosts:
                # Проверяем, может ли хост разместить VM
                if not self.can_host_vm(host_id, vm_id):
                    continue
                    
                # Проверяем, не превысит ли утилизация верхний порог
                test_allocations = {h: list(vms) for h, vms in new_allocations.items() if h != host_id}
                if host_id in new_allocations:
                    test_allocations[host_id] = new_allocations[host_id] + [vm_id]
                else:
                    test_allocations[host_id] = [vm_id]
                
                new_utilization = self.calculate_host_capacity(host_id, test_allocations)["max_utilization"]
                
                # Базовая оценка: близость к целевой утилизации
                score = -abs(new_utilization - TARGET_UTILIZATION)
                
                # Штраф за высокую утилизацию
                if new_utilization > UPPER_THRESHOLD:
                    score -= (new_utilization - UPPER_THRESHOLD) * 20
                
                # Бонус за соответствие размера VM и хоста
                size_ratio = vm_size / host_size
                if 0.05 <= size_ratio <= 0.2:  # Оптимальный диапазон
                    score += 0.5
                
                # Бонус за соответствие соотношения CPU/RAM
                host_capacity = self.calculate_host_capacity(host_id, new_allocations)["capacity"]
                host_ratio = host_capacity.get("cpu", 0) / max(host_capacity.get("ram", 1), 1)
                ratio_score = -abs(vm_ratio - host_ratio) * 0.1
                score += ratio_score
                
                if score > best_score:
                    best_score = score
                    best_host = host_id
            
            if best_host:
                # Размещаем VM на лучшем хосте
                if best_host not in new_allocations:
                    new_allocations[best_host] = []
                new_allocations[best_host].append(vm_id)
                
                # Обновляем утилизацию хоста
                hosts_utilization[best_host] = self.calculate_host_capacity(best_host, new_allocations)["max_utilization"]
                
                print(f"Placed VM {vm_id} on host {best_host} (new utilization: {hosts_utilization[best_host]:.2f})", file=sys.stderr)
            else:
                # Не удалось разместить VM
                placement_failures.append(vm_id)
                print(f"Failed to place VM {vm_id}", file=sys.stderr)
        
        # Записываем неразмещенные VM
        self.allocation_failures = placement_failures
        
        # Выбираем хосты для выключения
        self.select_hosts_for_shutdown(new_allocations)
        
        # Пытаемся консолидировать VM
        allocations, migrations = self.consolidate_vms(new_allocations)
        
        # Вычисляем и сохраняем оценки хостов
        self.host_scores = {
            host_id: self.calculate_host_score(host_id, new_allocations)
            for host_id in self.hosts
        }
        
        # Вычисляем утилизацию хостов
        host_utilization = {
            host_id: self.calculate_host_capacity(host_id, new_allocations)["max_utilization"]
            for host_id in self.hosts
        }
        
        # Выводим статистику по утилизации
        active_hosts = sum(1 for host_id in self.hosts if new_allocations.get(host_id))
        avg_utilization = sum(host_utilization.values()) / len(self.hosts) if self.hosts else 0
        print(f"Placement complete. Active hosts: {active_hosts}/{len(self.hosts)}, Avg utilization: {avg_utilization:.2f}", file=sys.stderr)
        print(f"Host scores: {self.host_scores}", file=sys.stderr)
        
        return new_allocations

    def consolidate_vms(self, allocations: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], List[Dict[str, str]]]:
        """Пытается консолидировать VM, перемещая их с хостов с низкой утилизацией на лучшие хосты."""
        migrations = []
        
        # Определяем хосты для консолидации (с низкой утилизацией или в списке на выключение)
        hosts_utilization = {
            host_id: self.calculate_host_capacity(host_id, allocations)["max_utilization"]
            for host_id in self.hosts
        }
        
        # Добавляем хосты с низкой утилизацией к кандидатам на выключение
        for host_id, utilization in hosts_utilization.items():
            if utilization > 0 and utilization < CONSOLIDATION_THRESHOLD:
                if host_id not in self.hosts_to_shutdown:
                    self.hosts_to_shutdown.add(host_id)
                    print(f"Adding host {host_id} to shutdown list due to low utilization ({utilization:.2f})", file=sys.stderr)
        
        print(f"Starting consolidation. Hosts to shutdown: {self.hosts_to_shutdown}", file=sys.stderr)
        print(f"Current utilization: {hosts_utilization}", file=sys.stderr)
        
        # Выводим статистику по хостам с нулевой утилизацией
        zero_counts = {host_id: count for host_id, count in self.host_zero_utilization_count.items() if count > 0}
        if zero_counts:
            print(f"Hosts with zero utilization counts: {zero_counts}", file=sys.stderr)
        
        # Сортируем хосты для выключения по приоритету
        hosts_to_empty = []
        for host_id in self.hosts_to_shutdown:
            if not allocations.get(host_id):
                continue  # Хост уже пуст
                
            # Получаем количество VM и утилизацию
            vm_count = len(allocations.get(host_id, []))
            utilization = hosts_utilization.get(host_id, 0)
            
            # Получаем счетчик нулевой утилизации
            zero_count = self.host_zero_utilization_count.get(host_id, 0)
            
            # Бонус для хостов, близких к получению бонуса за выключение
            bonus_priority = 0
            if host_id in self.hosts_with_previous_vms and zero_count > 0:
                bonus_priority = (zero_count + 1) * 5
                if zero_count >= BONUS_THRESHOLD - 1:
                    bonus_priority = 50  # Большой приоритет для хостов, которые почти получили бонус
            
            hosts_to_empty.append((host_id, vm_count, utilization, bonus_priority))
        
        # Сортируем хосты для выключения
        hosts_to_empty.sort(key=lambda x: (
            -x[3],  # Сначала по бонусному приоритету (убывание)
            x[1],   # Затем по количеству VM (возрастание)
            x[2]    # Затем по утилизации (возрастание)
        ))
        
        # Выводим информацию о сортировке
        print(f"Hosts to empty (sorted by priority): {hosts_to_empty}", file=sys.stderr)
        
        # Оцениваем необходимое количество хостов
        required_hosts = self.estimate_required_hosts()
        active_hosts = sum(1 for util in hosts_utilization.values() if util > 0)
        
        print(f"Required hosts: {required_hosts}, Active hosts: {active_hosts}", file=sys.stderr)
        
        # Пытаемся переместить VM с хостов, которые планируется выключить
        for host_id, vm_count, utilization, bonus_priority in hosts_to_empty:
            print(f"Trying to consolidate VMs from host {host_id} ({vm_count} VMs, {utilization:.2f} util, bonus priority: {bonus_priority})", file=sys.stderr)
            
            # Если достигли минимального количества хостов, останавливаем консолидацию
            if active_hosts <= required_hosts:
                print(f"Reached minimum required hosts ({required_hosts})", file=sys.stderr)
                break
            
            # Сортируем VM по размеру и приоритету для миграции
            host_vms = sorted(
                allocations.get(host_id, []),
                key=lambda vm_id: (
                    -self.vms.get(vm_id, {}).get("cpu", 0) - self.vms.get(vm_id, {}).get("ram", 0),  # Сначала мигрируем большие VM (в обратном порядке)
                    -abs(TARGET_UTILIZATION - hosts_utilization.get(host_id, 0))  # Приоритет VM с хостов, далеких от целевой утилизации
                )
            )
            
            print(f"VMs on host {host_id} for migration: {host_vms}", file=sys.stderr)
            
            successful_migrations = 0
            for vm_id in host_vms:
                # Если достигли лимита миграций, останавливаем консолидацию
                if len(migrations) >= MAX_MIGRATIONS:
                    print(f"Reached maximum migrations limit ({MAX_MIGRATIONS})", file=sys.stderr)
                    break
                
                # Ищем лучший хост для VM, исключая хосты для выключения
                best_host = None
                best_score = float('-inf')
                
                # Сортируем хосты по утилизации (предпочитаем хосты с утилизацией ближе к целевой)
                sorted_hosts = sorted(
                    [h for h in self.hosts if h not in self.hosts_to_shutdown],
                    key=lambda h: abs(TARGET_UTILIZATION - hosts_utilization.get(h, 0))
                )
                
                for target_host in sorted_hosts:
                    # Проверяем, может ли хост разместить VM
                    if not self.can_host_vm(target_host, vm_id):
                        continue
                        
                    # Проверяем текущую утилизацию хоста
                    current_utilization = hosts_utilization.get(target_host, 0)
                    
                    # Если утилизация уже близка к верхнему порогу, пропускаем
                    if current_utilization > UPPER_THRESHOLD - 0.05:
                        continue
                        
                    # Создаем тестовое размещение
                    test_allocations = {h: list(vms) for h, vms in allocations.items() if h != target_host}
                    if target_host in allocations:
                        test_allocations[target_host] = allocations[target_host] + [vm_id]
                    else:
                        test_allocations[target_host] = [vm_id]
                    
                    # Вычисляем новую утилизацию
                    new_utilization = self.calculate_host_capacity(target_host, test_allocations)["max_utilization"]
                    
                    # Оценка: комбинация близости к целевой утилизации и эффективности использования ресурсов
                    score = -abs(new_utilization - TARGET_UTILIZATION)
                    
                    # Штраф за высокую утилизацию
                    if new_utilization > UPPER_THRESHOLD:
                        score -= (new_utilization - UPPER_THRESHOLD) * 20
                    
                    # Бонус за консолидацию на уже активных хостах
                    if current_utilization > 0:
                        score += 0.2
                    
                    # Бонус за оптимальное соотношение CPU/RAM
                    vm = self.vms.get(vm_id, {})
                    vm_ratio = vm.get("cpu", 0) / max(vm.get("ram", 1), 1)
                    host_capacity = self.calculate_host_capacity(target_host, allocations)["capacity"]
                    host_ratio = host_capacity.get("cpu", 0) / max(host_capacity.get("ram", 1), 1)
                    ratio_score = -abs(vm_ratio - host_ratio) * 0.1
                    score += ratio_score
                    
                    if score > best_score:
                        best_score = score
                        best_host = target_host
                
                if best_host:
                    # Перемещаем VM
                    allocations[host_id].remove(vm_id)
                    if best_host not in allocations:
                        allocations[best_host] = []
                    allocations[best_host].append(vm_id)
                    
                    migrations.append({
                        "vm": vm_id,
                        "source": host_id,
                        "destination": best_host
                    })
                    
                    # Обновляем утилизацию
                    hosts_utilization[host_id] = self.calculate_host_capacity(host_id, allocations)["max_utilization"]
                    hosts_utilization[best_host] = self.calculate_host_capacity(best_host, allocations)["max_utilization"]
                    
                    successful_migrations += 1
                    print(f"Consolidation: Migrated VM {vm_id} from {host_id} to {best_host}", file=sys.stderr)
                    
                    # Если освободили хост, уменьшаем счетчик активных хостов
                    if not allocations[host_id]:
                        active_hosts -= 1
                        print(f"Host {host_id} is now empty. Active hosts: {active_hosts}", file=sys.stderr)
                    
                    # Ограничиваем количество миграций
                    if len(migrations) >= MAX_MIGRATIONS:
                        print(f"Reached maximum migrations limit ({MAX_MIGRATIONS})", file=sys.stderr)
                        return allocations, migrations
            
            # Если не удалось переместить ни одну VM с хоста, убираем его из списка на выключение
            if successful_migrations == 0:
                self.hosts_to_shutdown.remove(host_id)
                print(f"Could not migrate any VMs from host {host_id}, removing from shutdown list", file=sys.stderr)
        
        # Выводим окончательную статистику
        empty_hosts = [host_id for host_id in self.hosts if not allocations.get(host_id, [])]
        print(f"After consolidation: {len(empty_hosts)} empty hosts, {len(migrations)} migrations", file=sys.stderr)
        
        return allocations, migrations

    def get_migrations(self, new_allocations: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Определяет необходимые миграции VM между хостами."""
        migrations = []
        self.current_round_migrations = set()
        
        # Создаем отображение VM -> Host для новых размещений (оптимизировано)
        new_vm_to_host = {}
        for host_id, vm_list in new_allocations.items():
            for vm_id in vm_list:
                new_vm_to_host[vm_id] = host_id
        
        # Сначала пытаемся консолидировать VM с хостов, которые планируется выключить
        consolidated_allocations, consolidation_migrations = self.consolidate_vms(new_allocations)
        
        # Счетчики для статистики
        successful_migrations = 0
        cancelled_migrations = 0
        
        # Если уже достигли лимита миграций, возвращаем результат
        if len(consolidation_migrations) >= MAX_MIGRATIONS:
            # Обновляем размещения и возвращаем миграции
            for host_id, vms in consolidated_allocations.items():
                new_allocations[host_id] = vms
            
            # Обновляем статистику
            successful_migrations += len(consolidation_migrations)
            self.performance_stats["successful_migrations"].append(successful_migrations)
            self.performance_stats["cancelled_migrations"].append(cancelled_migrations)
            
            return consolidation_migrations
            
        # Добавляем миграции от консолидации
        migrations.extend(consolidation_migrations)
        for migration in consolidation_migrations:
            self.current_round_migrations.add(migration["vm"])
        
        # Если уже есть миграции, ограничиваем количество дополнительных миграций
        remaining_migrations = MAX_MIGRATIONS - len(migrations)
        if remaining_migrations <= 0:
            # Обновляем статистику
            successful_migrations += len(migrations)
            self.performance_stats["successful_migrations"].append(successful_migrations)
            self.performance_stats["cancelled_migrations"].append(cancelled_migrations)
            
            return migrations
            
        print(f"After consolidation: {len(migrations)} migrations, {remaining_migrations} remaining", file=sys.stderr)
        
        # Быстрая проверка необходимости миграций
        potential_migrations = []
        for vm_id, new_host in new_vm_to_host.items():
            old_host = self.vm_to_host_map.get(vm_id)
            if old_host and old_host != new_host:
                # Пропускаем VM, которые уже мигрировали
                if vm_id in self.current_round_migrations:
                    continue
                    
                # Быстрая оценка выгоды от миграции
                old_host_state = self.calculate_host_capacity(old_host, self.previous_allocations)
                new_host_state = self.calculate_host_capacity(new_host, new_allocations)
                
                # Если утилизация обоих хостов близка к целевой, пропускаем
                if (abs(old_host_state["max_utilization"] - TARGET_UTILIZATION) < 0.1 and
                    abs(new_host_state["max_utilization"] - TARGET_UTILIZATION) < 0.1):
                    continue
                
                # Если утилизация нового хоста слишком высокая, пропускаем
                if new_host_state["max_utilization"] > UPPER_THRESHOLD:
                    continue
                
                # Добавляем в список потенциальных миграций
                potential_migrations.append((vm_id, old_host, new_host))
        
        # Сортируем потенциальные миграции по приоритету
        potential_migrations.sort(
            key=lambda x: (
                # Приоритет выше для VM на хостах с высокой утилизацией
                self.calculate_host_capacity(x[1], self.previous_allocations)["max_utilization"],
                # Приоритет выше для больших VM
                -(self.vms[x[0]].get("cpu", 0) + self.vms[x[0]].get("ram", 0))
            ),
            reverse=True
        )
        
        # Оцениваем только самые перспективные миграции
        for vm_id, old_host, new_host in potential_migrations[:remaining_migrations * 2]:
            if len(migrations) >= MAX_MIGRATIONS:
                break
                
            # Создаем тестовое размещение без миграции (оптимизировано)
            test_allocations = {host_id: list(vms) for host_id, vms in new_allocations.items()}
            if new_host in test_allocations and vm_id in test_allocations[new_host]:
                test_allocations[new_host].remove(vm_id)
            if old_host not in test_allocations:
                test_allocations[old_host] = []
            test_allocations[old_host].append(vm_id)
            
            # Оценка без миграции
            no_migration_score = self.calculate_total_score(test_allocations, migrations)
            
            # Оценка с миграцией
            migration_score = self.calculate_total_score(new_allocations, migrations + [{"vm": vm_id, "source": old_host, "destination": new_host}])
            
            # Вычисляем выгоду от миграции
            benefit = migration_score - no_migration_score
            
            # Проверяем, стоит ли выполнять миграцию
            if benefit > MIN_BENEFIT_FOR_MIGRATION:
                migrations.append({
                    "vm": vm_id,
                    "source": old_host,
                    "destination": new_host
                })
                self.current_round_migrations.add(vm_id)
                successful_migrations += 1
                print(f"Migration: VM {vm_id} from {old_host} to {new_host} (benefit: {benefit})", file=sys.stderr)
            else:
                # Отменяем миграцию, если выгода недостаточна
                if vm_id in new_allocations[new_host]:
                    new_allocations[new_host].remove(vm_id)
                if old_host in new_allocations:
                    if vm_id not in new_allocations[old_host]:
                        new_allocations[old_host].append(vm_id)
                cancelled_migrations += 1
                print(f"Cancelled migration of VM {vm_id} (benefit: {benefit} < {MIN_BENEFIT_FOR_MIGRATION})", file=sys.stderr)
        
        # Обновляем статистику
        self.performance_stats["successful_migrations"].append(successful_migrations)
        self.performance_stats["cancelled_migrations"].append(cancelled_migrations)
        
        return migrations

    def calculate_host_score(self, host_id: str, allocations: Dict[str, List[str]] = None) -> float:
        """Вычисляет оценку хоста на основе его утилизации."""
        state = self.calculate_host_capacity(host_id, allocations)
        utilization = state["max_utilization"]
        
        # Базовая оценка за утилизацию
        score = calculate_reward(utilization)
        
        # Бонус за выключенный хост
        if utilization == 0:
            # Проверяем, использовался ли хост ранее
            if host_id in self.hosts_with_previous_vms:
                zero_count = self.host_zero_utilization_count.get(host_id, 0)
                # Бонус начисляется после 5+ раундов простоя
                if zero_count >= BONUS_THRESHOLD:
                    score += 8  # Бонус за выключенный хост
                    print(f"Host {host_id} gets bonus for being off for {zero_count} rounds", file=sys.stderr)
                # Частичный бонус для хостов, близких к получению полного бонуса
                elif zero_count >= BONUS_THRESHOLD - 1:
                    score += 4  # Частичный бонус для стимулирования сохранения хоста выключенным
        
        # Штраф за утилизацию выше верхнего порога
        if utilization > UPPER_THRESHOLD:
            penalty = (utilization - UPPER_THRESHOLD) * 5
            score -= penalty
            
        return score
        
    def calculate_total_score(self, allocations: Dict[str, List[str]], migrations: List[Dict[str, str]]) -> float:
        """Вычисляет общий счет за раунд."""
        # Сумма оценок всех хостов
        host_scores = {
            host_id: self.calculate_host_score(host_id, allocations)
            for host_id in self.hosts
        }
        
        # Выводим детальную информацию о оценках хостов
        active_hosts = sum(1 for host_id, score in host_scores.items() if score > 0)
        bonus_hosts = sum(1 for host_id in self.hosts 
                         if self.calculate_host_capacity(host_id, allocations)["max_utilization"] == 0 
                         and self.host_zero_utilization_count.get(host_id, 0) >= BONUS_THRESHOLD
                         and host_id in self.hosts_with_previous_vms)
        
        print(f"Host scores: {host_scores}", file=sys.stderr)
        print(f"Active hosts: {active_hosts}/{len(self.hosts)}, Bonus hosts: {bonus_hosts}", file=sys.stderr)
        
        total_score = sum(host_scores.values())
        
        # Штраф за миграции
        migration_penalty = len(migrations) ** 2
        total_score -= migration_penalty
        
        # Штраф за неразмещенные VM
        allocation_failure_penalty = 0
        if self.allocation_failures:
            allocation_failure_penalty = 5 * len(self.hosts) * len(self.allocation_failures)
            total_score -= allocation_failure_penalty
            
        print(f"Total score: {total_score} (base: {sum(host_scores.values())}, migration penalty: {migration_penalty}, allocation failure penalty: {allocation_failure_penalty})", file=sys.stderr)
            
        return total_score

if __name__ == "__main__":
    scheduler = VMScheduler()
    scheduler.process_input() 