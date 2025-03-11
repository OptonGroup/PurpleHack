#!/usr/bin/env python3
import json
import sys
from typing import Dict, List, Tuple, Set, Any, Optional
import math

# Константы для оптимизации
TARGET_UTILIZATION = 0.807197  # Оптимальная утилизация для максимального балла
UPPER_THRESHOLD = 0.9  # Верхний порог утилизации
LOWER_THRESHOLD = 0.3  # Нижний порог утилизации
MAX_MIGRATIONS = 3  # Максимальное число миграций за раунд
MIN_BENEFIT_FOR_MIGRATION = 15  # Минимальная выгода для оправдания миграции

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
            return self.capacity_cache[cache_key].copy()
        
        if host_id not in self.hosts:
            result = {"capacity": {}, "utilization": {}, "max_utilization": 0}
            self.capacity_cache[cache_key] = result
            return result.copy()
        
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
        
        return result.copy()
        
    def clear_capacity_cache(self):
        """Очищает кэш вычисленных емкостей."""
        self.capacity_cache = {}

    def process_input(self):
        """Обрабатывает входные данные построчно."""
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                    
                try:
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
                    self.previous_allocations = {host_id: list(vms) for host_id, vms in new_allocations.items()}
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
            
            # Загружаем виртуальные машины
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
                
                # Удаляем VM из предыдущих размещений
                for host_id, vm_list in list(self.previous_allocations.items()):
                    self.previous_allocations[host_id] = [vm for vm in vm_list if vm not in removed_vms]
            
            # Обновляем vm_to_host_map на основе предыдущих размещений
            self.vm_to_host_map = {}
            for host_id, vm_list in self.previous_allocations.items():
                for vm_id in vm_list:
                    if vm_id in self.vms:  # Проверяем, что VM все еще существует
                        self.vm_to_host_map[vm_id] = host_id
            print(f"Updated VM to host map: {len(self.vm_to_host_map)} mappings", file=sys.stderr)
            
            # Обновляем множество хостов с предыдущими VM
            self.hosts_with_previous_vms = set(
                host_id for host_id, vm_list in self.previous_allocations.items()
                if vm_list and host_id in self.hosts
            )
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
            
        except Exception as e:
            print(f"Error in load_data: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise

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

    def get_best_host_for_vm(self, vm_id: str, target_utilization: float = TARGET_UTILIZATION) -> Optional[str]:
        """Находит лучший хост для размещения VM с учетом целевой утилизации."""
        if vm_id not in self.vms:
            return None
        
        vm = self.vms[vm_id]
        best_host = None
        best_score = float('-inf')
        
        # Получаем текущее состояние всех хостов
        host_states = {
            host_id: self.calculate_host_capacity(host_id)
            for host_id in self.hosts
        }
        
        # Оцениваем каждый хост
        for host_id, state in host_states.items():
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
            
            # Комбинируем оценки
            combined_score = score + 0.1 * ratio_similarity
            
            # Если текущий хост лучше предыдущего лучшего, обновляем
            if combined_score > best_score:
                best_score = combined_score
                best_host = host_id
        
        return best_host

    def place_vms(self) -> Dict[str, List[str]]:
        """Размещает VM на хостах с учетом оптимизации утилизации и минимизации миграций."""
        print(f"\nStarting VM placement for round {self.round_counter}...", file=sys.stderr)
        
        # Инициализируем новые размещения на основе предыдущих
        new_allocations = {host_id: list(vms) for host_id, vms in self.previous_allocations.items()}
        for host_id in self.hosts:
            if host_id not in new_allocations:
                new_allocations[host_id] = []
                
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
            
        # Собираем отсортированный список VM
        sorted_vms = []
        for group in sorted(size_groups.keys(), reverse=True):
            sorted_vms.extend(size_groups[group])
        
        # Сначала пытаемся разместить VM на их текущих хостах
        unplaced_vms = []
        for vm_id in sorted_vms:
            current_host = self.vm_to_host_map.get(vm_id)
            
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
        
        # Пытаемся разместить оставшиеся VM
        self.allocation_failures = []  # Сбрасываем список неразмещенных VM
        
        # Сортируем хосты по утилизации для более равномерного распределения
        host_utilization = {
            host_id: self.calculate_host_capacity(host_id, new_allocations)["max_utilization"]
            for host_id in self.hosts
        }
        sorted_hosts = sorted(self.hosts.keys(), key=lambda h: host_utilization[h])
        
        for vm_id in unplaced_vms:
            # Если VM уже мигрировала в этом раунде, пропускаем
            if vm_id in self.current_round_migrations:
                continue
                
            # Находим лучший хост для VM
            best_host = None
            best_score = float('-inf')
            vm_ratio = vm_metrics[vm_id][1]
            
            for host_id in sorted_hosts:
                # Проверяем базовые ограничения ресурсов
                if not self.can_host_vm(host_id, vm_id):
                    continue
                    
                # Создаем тестовое размещение
                test_allocations = {
                    h: list(vms) for h, vms in new_allocations.items()
                }
                test_allocations[host_id].append(vm_id)
                
                # Вычисляем оценку для данного размещения
                score = self.calculate_host_score(host_id, test_allocations)
                
                # Учитываем соотношение CPU/RAM
                host_state = self.calculate_host_capacity(host_id, new_allocations)
                host_ratio = host_state["capacity"]["cpu"] / max(host_state["capacity"]["ram"], 1)
                ratio_score = -abs(vm_ratio - host_ratio)
                
                # Штраф за высокую утилизацию
                utilization_penalty = 0
                if host_state["max_utilization"] > UPPER_THRESHOLD:
                    utilization_penalty = (host_state["max_utilization"] - UPPER_THRESHOLD) * 10
                elif host_state["max_utilization"] < LOWER_THRESHOLD:
                    utilization_penalty = (LOWER_THRESHOLD - host_state["max_utilization"]) * 5
                
                # Бонус за размещение на менее загруженном хосте
                utilization_bonus = (1 - host_state["max_utilization"]) * 2
                
                # Комбинируем оценки
                final_score = score + 0.3 * ratio_score - utilization_penalty + utilization_bonus
                
                if final_score > best_score:
                    best_score = final_score
                    best_host = host_id
            
            if best_host:
                new_allocations[best_host].append(vm_id)
                print(f"Placed VM {vm_id} on host {best_host} with score {best_score}", file=sys.stderr)
                
                # Обновляем утилизацию хоста
                host_utilization[best_host] = self.calculate_host_capacity(best_host, new_allocations)["max_utilization"]
                # Пересортируем хосты
                sorted_hosts = sorted(self.hosts.keys(), key=lambda h: host_utilization[h])
            else:
                self.allocation_failures.append(vm_id)
                print(f"Failed to place VM {vm_id}", file=sys.stderr)
        
        # Обновляем счетчики нулевой утилизации
        for host_id in self.hosts:
            if not new_allocations.get(host_id):
                self.host_zero_utilization_count[host_id] = self.host_zero_utilization_count.get(host_id, 0) + 1
            else:
                self.host_zero_utilization_count[host_id] = 0
                
        # Вычисляем и сохраняем оценки хостов
        self.host_scores = {
            host_id: self.calculate_host_score(host_id, new_allocations)
            for host_id in self.hosts
        }
        
        print(f"Placement complete. Scores: {self.host_scores}", file=sys.stderr)
        return new_allocations

    def get_migrations(self, new_allocations: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Определяет необходимые миграции VM между хостами."""
        migrations = []
        self.current_round_migrations = set()
        
        # Создаем отображение VM -> Host для новых размещений
        new_vm_to_host = {}
        for host_id, vm_list in new_allocations.items():
            for vm_id in vm_list:
                new_vm_to_host[vm_id] = host_id
        
        # Вычисляем метрики для всех хостов
        host_metrics = {}
        for host_id in self.hosts:
            state = self.calculate_host_capacity(host_id, new_allocations)
            host_metrics[host_id] = {
                "utilization": state["max_utilization"],
                "cpu_ratio": state["capacity"]["cpu"] / max(state["capacity"]["ram"], 1),
                "total_resources": state["capacity"]["cpu"] + state["capacity"]["ram"]
            }
        
        # Находим хосты с высокой и низкой утилизацией
        overloaded_hosts = [
            host_id for host_id, metrics in host_metrics.items()
            if metrics["utilization"] > UPPER_THRESHOLD
        ]
        underloaded_hosts = [
            host_id for host_id, metrics in host_metrics.items()
            if metrics["utilization"] < LOWER_THRESHOLD and len(new_allocations[host_id]) > 0
        ]
        
        # Сортируем хосты по утилизации
        overloaded_hosts.sort(key=lambda h: host_metrics[h]["utilization"], reverse=True)
        underloaded_hosts.sort(key=lambda h: host_metrics[h]["utilization"])
        
        # Пытаемся разгрузить перегруженные хосты
        for source_host in overloaded_hosts:
            if len(migrations) >= MAX_MIGRATIONS:
                break
                
            source_vms = sorted(
                new_allocations[source_host],
                key=lambda vm_id: (
                    # Приоритет VM с наибольшим несоответствием по CPU/RAM
                    abs(self.vms[vm_id].get("cpu", 0) / max(self.vms[vm_id].get("ram", 1), 1) - 
                        host_metrics[source_host]["cpu_ratio"]),
                    # Затем по размеру
                    -(self.vms[vm_id].get("cpu", 0) + self.vms[vm_id].get("ram", 0))
                )
            )
            
            for vm_id in source_vms:
                if vm_id in self.current_round_migrations:
                    continue
                    
                # Ищем лучший хост для миграции
                best_host = None
                best_score = float('-inf')
                best_benefit = 0
                
                # Сначала проверяем недогруженные хосты
                candidate_hosts = underloaded_hosts.copy()
                # Затем добавляем хосты с нормальной утилизацией
                candidate_hosts.extend([
                    host_id for host_id in self.hosts
                    if host_id not in overloaded_hosts and host_id not in underloaded_hosts
                ])
                
                for dest_host in candidate_hosts:
                    if not self.can_host_vm(dest_host, vm_id):
                        continue
                        
                    # Создаем тестовое размещение
                    test_allocations = {
                        h: [vm for vm in vms if vm != vm_id]
                        for h, vms in new_allocations.items()
                    }
                    test_allocations[dest_host].append(vm_id)
                    
                    # Оценка без миграции
                    no_migration_score = self.calculate_total_score(new_allocations, migrations)
                    
                    # Оценка с миграцией
                    migration = {"vm": vm_id, "source": source_host, "destination": dest_host}
                    migration_score = self.calculate_total_score(test_allocations, migrations + [migration])
                    
                    # Вычисляем выгоду
                    benefit = migration_score - no_migration_score
                    
                    # Учитываем дополнительные факторы
                    dest_state = self.calculate_host_capacity(dest_host, test_allocations)
                    
                    # Бонус за улучшение баланса утилизации
                    balance_bonus = 0
                    if (abs(dest_state["max_utilization"] - TARGET_UTILIZATION) <
                        abs(host_metrics[source_host]["utilization"] - TARGET_UTILIZATION)):
                        balance_bonus = 5
                    
                    # Бонус за консолидацию на недогруженных хостах
                    consolidation_bonus = 0
                    if dest_host in underloaded_hosts:
                        consolidation_bonus = 3
                    
                    total_score = benefit + balance_bonus + consolidation_bonus
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_host = dest_host
                        best_benefit = benefit
                
                if best_host and best_benefit > MIN_BENEFIT_FOR_MIGRATION:
                    migration = {
                        "vm": vm_id,
                        "source": source_host,
                        "destination": best_host
                    }
                    migrations.append(migration)
                    self.current_round_migrations.add(vm_id)
                    
                    # Обновляем размещения и метрики
                    new_allocations[source_host].remove(vm_id)
                    new_allocations[best_host].append(vm_id)
                    
                    source_state = self.calculate_host_capacity(source_host, new_allocations)
                    dest_state = self.calculate_host_capacity(best_host, new_allocations)
                    
                    host_metrics[source_host]["utilization"] = source_state["max_utilization"]
                    host_metrics[best_host]["utilization"] = dest_state["max_utilization"]
                    
                    print(f"Migration: VM {vm_id} from {source_host} to {best_host} "
                          f"(benefit: {best_benefit}, total score: {best_score})", file=sys.stderr)
                    
                    if len(migrations) >= MAX_MIGRATIONS:
                        break
                else:
                    print(f"Cancelled migration of VM {vm_id} "
                          f"(best benefit: {best_benefit} < {MIN_BENEFIT_FOR_MIGRATION})", file=sys.stderr)
        
        # Пытаемся консолидировать VM с недогруженных хостов
        if len(migrations) < MAX_MIGRATIONS:
            for source_host in underloaded_hosts:
                if len(migrations) >= MAX_MIGRATIONS:
                    break
                    
                if not new_allocations[source_host]:
                    continue
                    
                # Пытаемся переместить все VM с недогруженного хоста
                source_vms = sorted(
                    new_allocations[source_host],
                    key=lambda vm_id: self.vms[vm_id].get("cpu", 0) + self.vms[vm_id].get("ram", 0)
                )
                
                for vm_id in source_vms:
                    if vm_id in self.current_round_migrations:
                        continue
                        
                    # Ищем хост с подходящей утилизацией
                    best_host = None
                    best_score = float('-inf')
                    best_benefit = 0
                    
                    for dest_host in self.hosts:
                        if dest_host == source_host or not self.can_host_vm(dest_host, vm_id):
                            continue
                            
                        # Пропускаем перегруженные хосты
                        if host_metrics[dest_host]["utilization"] > UPPER_THRESHOLD:
                            continue
                            
                        # Создаем тестовое размещение
                        test_allocations = {
                            h: [vm for vm in vms if vm != vm_id]
                            for h, vms in new_allocations.items()
                        }
                        test_allocations[dest_host].append(vm_id)
                        
                        # Оценка без миграции
                        no_migration_score = self.calculate_total_score(new_allocations, migrations)
                        
                        # Оценка с миграцией
                        migration = {"vm": vm_id, "source": source_host, "destination": dest_host}
                        migration_score = self.calculate_total_score(test_allocations, migrations + [migration])
                        
                        benefit = migration_score - no_migration_score
                        
                        # Бонус за возможность выключения хоста
                        shutdown_bonus = 0
                        if len(new_allocations[source_host]) == 1:  # Последняя VM на хосте
                            shutdown_bonus = 10
                        
                        total_score = benefit + shutdown_bonus
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_host = dest_host
                            best_benefit = benefit
                    
                    if best_host and best_benefit > MIN_BENEFIT_FOR_MIGRATION:
                        migration = {
                            "vm": vm_id,
                            "source": source_host,
                            "destination": best_host
                        }
                        migrations.append(migration)
                        self.current_round_migrations.add(vm_id)
                        
                        # Обновляем размещения и метрики
                        new_allocations[source_host].remove(vm_id)
                        new_allocations[best_host].append(vm_id)
                        
                        source_state = self.calculate_host_capacity(source_host, new_allocations)
                        dest_state = self.calculate_host_capacity(best_host, new_allocations)
                        
                        host_metrics[source_host]["utilization"] = source_state["max_utilization"]
                        host_metrics[best_host]["utilization"] = dest_state["max_utilization"]
                        
                        print(f"Consolidation: VM {vm_id} from {source_host} to {best_host} "
                              f"(benefit: {best_benefit}, total score: {best_score})", file=sys.stderr)
                        
                        if len(migrations) >= MAX_MIGRATIONS:
                            break
                    else:
                        print(f"Cancelled consolidation of VM {vm_id} "
                              f"(best benefit: {best_benefit} < {MIN_BENEFIT_FOR_MIGRATION})", file=sys.stderr)
        
        return migrations

    def calculate_host_score(self, host_id: str, allocations: Dict[str, List[str]] = None) -> float:
        """Вычисляет оценку хоста на основе его утилизации."""
        state = self.calculate_host_capacity(host_id, allocations)
        utilization = state["max_utilization"]
        
        # Базовая оценка за утилизацию
        score = calculate_reward(utilization)
        
        # Бонус за выключенный хост
        if utilization == 0 and host_id in self.hosts_with_previous_vms:
            zero_count = self.host_zero_utilization_count.get(host_id, 0)
            if zero_count >= 5:
                score += 8  # Бонус за выключенный хост
                
        return score
        
    def calculate_total_score(self, allocations: Dict[str, List[str]], migrations: List[Dict[str, str]]) -> float:
        """Вычисляет общий счет за раунд."""
        # Сумма оценок всех хостов
        host_scores = {
            host_id: self.calculate_host_score(host_id, allocations)
            for host_id in self.hosts
        }
        
        total_score = sum(host_scores.values())
        
        # Штраф за миграции
        migration_penalty = len(migrations) ** 2
        total_score -= migration_penalty
        
        # Штраф за неразмещенные VM
        if self.allocation_failures:
            total_score -= 5 * len(self.hosts) * len(self.allocation_failures)
            
        return total_score

if __name__ == "__main__":
    scheduler = VMScheduler()
    scheduler.process_input() 