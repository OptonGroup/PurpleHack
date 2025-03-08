#!/usr/bin/env python3
import json
import sys
from typing import Dict, List, Tuple, Set, Any, Optional
import time

# Константы для оптимизации
TARGET_UTILIZATION = 0.8  # Целевая утилизация хостов (80%)
UPPER_THRESHOLD = 0.9  # Верхний порог утилизации
LOWER_THRESHOLD = 0.2  # Нижний порог утилизации
MAX_MIGRATIONS = 5  # Максимальное число миграций за раунд

class VMScheduler:
    def __init__(self):
        self.hosts = {}  # имя хоста -> {cpu, ram}
        self.vms = {}  # имя VM -> {cpu, ram}
        self.allocations = {}  # имя хоста -> список VM
        self.previous_allocations = {}  # имя хоста -> список VM из предыдущего раунда
        self.host_zero_utilization_count = {}  # отслеживание хостов с нулевой нагрузкой
        self.vm_to_host_map = {}  # Отображение VM -> Host из предыдущих аллокаций
        self.hosts_with_previous_vms = set()  # Множество хостов, на которых были VM
        
    def load_data(self, input_str: str) -> None:
        """Загружает входные данные из JSON строки."""
        try:
            data = json.loads(input_str)
            
            # Загружаем хосты
            self.hosts = data.get("hosts", {})
            
            # Загружаем виртуальные машины
            self.vms = data.get("virtual_machines", data.get("vms", {}))
            
            # Загружаем предыдущие размещения
            self.previous_allocations = data.get("allocations", {})
            
            # Обновляем vm_to_host_map на основе предыдущих размещений
            self.vm_to_host_map = {}
            for host_id, vm_list in self.previous_allocations.items():
                for vm_id in vm_list:
                    if vm_id in self.vms:
                        self.vm_to_host_map[vm_id] = host_id
            
            # Обновляем множество хостов с предыдущими VM
            self.hosts_with_previous_vms = {
                host_id for host_id, vm_list in self.previous_allocations.items()
                if vm_list and host_id in self.hosts
            }
            
            # Обрабатываем diff, если он есть
            if "diff" in data:
                diff = data["diff"]
                
                # Добавляем новые VM
                if "add" in diff:
                    new_vms = diff["add"].get("virtual_machines", diff["add"].get("vms", {}))
                    if isinstance(new_vms, list):
                        # Если new_vms - список, значит это список VM для добавления из основного словаря
                        for vm_id in new_vms:
                            if vm_id in self.vms:
                                continue
                            if vm_id in data.get("virtual_machines", {}):
                                self.vms[vm_id] = data["virtual_machines"][vm_id]
                            elif vm_id in data.get("vms", {}):
                                self.vms[vm_id] = data["vms"][vm_id]
                    else:
                        # Если new_vms - словарь, добавляем его напрямую
                        self.vms.update(new_vms)
                
                # Удаляем VM
                if "remove" in diff:
                    removed_vms = diff["remove"].get("virtual_machines", diff["remove"].get("vms", []))
                    if isinstance(removed_vms, dict):
                        removed_vms = list(removed_vms.keys())
                    for vm_id in removed_vms:
                        if vm_id in self.vms:
                            del self.vms[vm_id]
                        if vm_id in self.vm_to_host_map:
                            del self.vm_to_host_map[vm_id]
                
                # Обновляем предыдущие размещения, удаляя несуществующие VM
                for host_id in self.previous_allocations:
                    self.previous_allocations[host_id] = [
                        vm_id for vm_id in self.previous_allocations[host_id]
                        if vm_id in self.vms
                    ]
            
            # Инициализируем счетчики хостов с нулевой утилизацией
            self.host_zero_utilization_count = {host_id: 0 for host_id in self.hosts}
        
        except json.JSONDecodeError:
            print("Error: Invalid JSON input")
            self.hosts = {}
            self.vms = {}
            self.previous_allocations = {}
            self.vm_to_host_map = {}
            self.hosts_with_previous_vms = set()
            self.host_zero_utilization_count = {}
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            self.hosts = {}
            self.vms = {}
            self.previous_allocations = {}
            self.vm_to_host_map = {}
            self.hosts_with_previous_vms = set()
            self.host_zero_utilization_count = {}
            raise

    def calculate_host_capacity(self, host_id: str, allocations: Dict[str, List[str]] = None) -> Dict:
        """Вычисляет остаточные ресурсы хоста на основе текущих размещений VM."""
        if host_id not in self.hosts:
            return {}
        
        host = self.hosts[host_id]
        capacity = {
            "cpu": host.get("cpu", 0),
            "ram": host.get("ram", 0)
        }
        
        # Добавляем disk только если он есть в данных хоста
        if "disk" in host:
            capacity["disk"] = host["disk"]
        
        # Используем предоставленные аллокации или текущие
        allocs = allocations if allocations is not None else self.previous_allocations
        
        # Вычитаем ресурсы, используемые виртуальными машинами
        allocated_vms = allocs.get(host_id, [])
        for vm_id in allocated_vms:
            if vm_id in self.vms:
                vm = self.vms[vm_id]
                capacity["cpu"] -= vm.get("cpu", 0)
                capacity["ram"] -= vm.get("ram", 0)
                if "disk" in capacity and "disk" in vm:
                    capacity["disk"] -= vm["disk"]
        
        return capacity

    def can_host_vm(self, host_id: str, vm_id: str, existing_resources: Optional[Dict] = None) -> bool:
        """Проверяет, может ли хост разместить указанную VM."""
        if host_id not in self.hosts or vm_id not in self.vms:
            return False
        
        vm = self.vms[vm_id]
        
        # Используем предоставленные ресурсы или вычисляем их
        if existing_resources is None:
            resources = self.calculate_host_capacity(host_id)
        else:
            resources = existing_resources.copy()
        
        # Проверяем, достаточно ли ресурсов для размещения VM
        can_host = (resources.get("cpu", 0) >= vm.get("cpu", 0) and 
                   resources.get("ram", 0) >= vm.get("ram", 0))
        
        # Проверяем disk только если он есть и в ресурсах, и в VM
        if "disk" in resources and "disk" in vm:
            can_host = can_host and resources["disk"] >= vm["disk"]
        
        # Если базовая проверка пройдена, проверяем утилизацию
        if can_host:
            # Получаем текущие VM на хосте
            current_vms = self.previous_allocations.get(host_id, [])
            # Добавляем новую VM
            test_vms = current_vms + [vm_id]
            # Проверяем утилизацию
            utilization = self.calculate_host_utilization(host_id, test_vms)
            # Разрешаем размещение только если утилизация не превышает 1.0
            can_host = utilization <= 1.0
        
        return can_host

    def calculate_host_utilization(self, host_id: str, allocation: List[str] = None) -> float:
        """Вычисляет утилизацию хоста на основе размещенных VM."""
        if host_id not in self.hosts:
            return 0.0
        
        host = self.hosts[host_id]
        if not allocation:
            allocation = self.previous_allocations.get(host_id, [])
        
        # Инициализируем счетчики для каждого типа ресурсов
        used_resources = {"cpu": 0, "ram": 0}
        total_resources = {"cpu": host.get("cpu", 0), "ram": host.get("ram", 0)}
        
        # Добавляем disk только если он есть
        if "disk" in host:
            used_resources["disk"] = 0
            total_resources["disk"] = host["disk"]
        
        # Подсчитываем использованные ресурсы
        for vm_id in allocation:
            if vm_id in self.vms:
                vm = self.vms[vm_id]
                used_resources["cpu"] += vm.get("cpu", 0)
                used_resources["ram"] += vm.get("ram", 0)
                if "disk" in used_resources and "disk" in vm:
                    used_resources["disk"] += vm["disk"]
        
        # Вычисляем утилизацию для каждого типа ресурсов
        utilizations = []
        for resource in used_resources:
            if total_resources[resource] > 0:
                utilization = min(1.0, used_resources[resource] / total_resources[resource])
                utilizations.append(utilization)
        
        # Возвращаем среднее значение утилизации
        return sum(utilizations) / len(utilizations) if utilizations else 0.0

    def calculate_migration_cost(self, vm_id: str, source_host: str, target_host: str) -> float:
        """Вычисляет стоимость миграции VM с учетом различных факторов."""
        if vm_id not in self.vms or source_host not in self.hosts or target_host not in self.hosts:
            return float('inf')
        
        vm = self.vms[vm_id]
        vm_size = vm.get("cpu", 0) + vm.get("ram", 0) + vm.get("disk", 0)
        
        # Базовая стоимость зависит от размера VM
        cost = vm_size
        
        # Увеличиваем стоимость, если VM уже мигрировала
        if vm_id in self.vm_to_host_map and self.vm_to_host_map[vm_id] != source_host:
            cost *= 2
        
        # Учитываем утилизацию целевого хоста
        target_utilization = self.calculate_host_utilization(target_host)
        if target_utilization > UPPER_THRESHOLD:
            cost *= 1.5
        elif target_utilization < LOWER_THRESHOLD:
            cost *= 0.8
        
        return cost

    def find_best_migration_target(self, vm_id: str, current_host: str, host_resources: Dict[str, Dict]) -> Optional[str]:
        """Находит лучший хост для миграции VM."""
        if vm_id not in self.vms:
            return None
        
        vm = self.vms[vm_id]
        best_host = None
        min_cost = float('inf')
        
        for host_id in self.hosts:
            if host_id == current_host:
                continue
            
            # Проверяем, может ли хост принять VM
            if not self.can_host_vm(host_id, vm_id, host_resources[host_id]):
                continue
            
            # Вычисляем стоимость миграции
            cost = self.calculate_migration_cost(vm_id, current_host, host_id)
            
            # Учитываем текущую утилизацию хоста
            utilization = self.calculate_host_utilization(host_id)
            if utilization < TARGET_UTILIZATION:
                cost *= 0.9  # Поощряем миграцию на менее загруженные хосты
            
            if cost < min_cost:
                min_cost = cost
                best_host = host_id
        
        return best_host

    def optimize_migrations(self, allocations: Dict[str, List[str]], host_resources: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Оптимизирует размещение VM через миграции."""
        migrations_count = 0
        new_allocations = {k: v[:] for k, v in allocations.items()}
        
        # Инициализируем current_vm_locations из текущих размещений
        current_vm_locations = {}
        for host_id, vm_list in new_allocations.items():
            for vm_id in vm_list:
                if vm_id in self.vms:
                    current_vm_locations[vm_id] = host_id
        
        # Функция проверки возможности миграции
        def validate_migration(vm_id: str, from_host: str, to_host: str) -> bool:
            # Проверяем, что VM существует и находится на исходном хосте
            if vm_id not in self.vms or current_vm_locations.get(vm_id) != from_host:
                return False
            
            # Проверяем, что целевой хост может принять VM
            if not self.can_host_vm(to_host, vm_id, host_resources[to_host]):
                return False
            
            return True
        
        # Функция оценки эффективности миграции
        def evaluate_migration(vm_id: str, from_host: str, to_host: str) -> float:
            # Проверяем возможность миграции
            if not validate_migration(vm_id, from_host, to_host):
                return float('inf')
            
            # Если хост уже оптимально загружен, не трогаем его
            current_utilization = self.calculate_host_utilization(from_host, new_allocations[from_host])
            if abs(current_utilization - TARGET_UTILIZATION) < 0.1:
                return float('inf')
            
            # Оцениваем состояние до миграции
            before_from_util = current_utilization
            before_to_util = self.calculate_host_utilization(to_host, new_allocations[to_host])
            before_score = abs(before_from_util - TARGET_UTILIZATION) + abs(before_to_util - TARGET_UTILIZATION)
            
            # Симулируем миграцию
            test_allocations = {k: v[:] for k, v in new_allocations.items()}
            test_allocations[from_host].remove(vm_id)
            test_allocations[to_host].append(vm_id)
            
            # Оцениваем состояние после миграции
            after_from_util = self.calculate_host_utilization(from_host, test_allocations[from_host])
            after_to_util = self.calculate_host_utilization(to_host, test_allocations[to_host])
            after_score = abs(after_from_util - TARGET_UTILIZATION) + abs(after_to_util - TARGET_UTILIZATION)
            
            # Вычисляем выигрыш от миграции
            improvement = before_score - after_score
            
            # Учитываем стоимость миграции
            migration_cost = self.calculate_migration_cost(vm_id, from_host, to_host)
            
            # Если миграция не дает существенного улучшения, не выполняем её
            if improvement < 0.1:
                return float('inf')
            
            return migration_cost / improvement
        
        # Сортируем хосты по отклонению от целевой утилизации
        sorted_hosts = sorted(
            self.hosts.keys(),
            key=lambda h: abs(self.calculate_host_utilization(h, new_allocations[h]) - TARGET_UTILIZATION)
        )
        
        # Пытаемся оптимизировать размещение
        while migrations_count < MAX_MIGRATIONS:
            best_migration = None
            best_score = float('inf')
            
            # Ищем лучшую возможную миграцию
            for from_host in sorted_hosts:
                # Пропускаем пустые хосты
                if not new_allocations[from_host]:
                    continue
                
                # Сортируем VM по размеру (от больших к маленьким)
                vms_on_host = sorted(
                    new_allocations[from_host],
                    key=lambda vm: -(
                        self.vms[vm].get("cpu", 0) + 
                        self.vms[vm].get("ram", 0) + 
                        self.vms[vm].get("disk", 0)
                    )
                )
                
                for vm_id in vms_on_host:
                    # Проверяем, что VM действительно находится на этом хосте
                    if current_vm_locations.get(vm_id) != from_host:
                        continue
                    
                    # Ищем подходящий хост для миграции
                    for to_host in sorted_hosts:
                        if to_host == from_host:
                            continue
                        
                        # Оцениваем эффективность миграции
                        score = evaluate_migration(vm_id, from_host, to_host)
                        
                        if score < best_score:
                            best_score = score
                            best_migration = (vm_id, from_host, to_host)
            
            # Если нашли подходящую миграцию, выполняем её
            if best_migration and best_score < float('inf'):
                vm_id, from_host, to_host = best_migration
                
                # Выполняем миграцию
                new_allocations[from_host].remove(vm_id)
                new_allocations[to_host].append(vm_id)
                
                # Обновляем ресурсы хостов
                vm = self.vms[vm_id]
                for resource in ["cpu", "ram", "disk"]:
                    if resource in vm and resource in host_resources[from_host]:
                        host_resources[from_host][resource] += vm.get(resource, 0)
                    if resource in vm and resource in host_resources[to_host]:
                        host_resources[to_host][resource] -= vm.get(resource, 0)
                
                # Обновляем текущее местоположение VM
                current_vm_locations[vm_id] = to_host
                
                migrations_count += 1
            else:
                # Если не нашли подходящей миграции, прекращаем оптимизацию
                break
        
        return new_allocations

    def place_vms(self) -> Dict[str, List[str]]:
        """Размещает виртуальные машины на хостах."""
        # Инициализируем новые размещения и ресурсы хостов
        new_allocations = {host_id: [] for host_id in self.hosts}
        host_resources = {host_id: self.hosts[host_id].copy() for host_id in self.hosts}
        
        # Очищаем vm_to_host_map перед новым размещением
        self.vm_to_host_map = {}
        
        # Сначала пытаемся сохранить существующие размещения
        for host_id, vm_list in self.previous_allocations.items():
            if host_id not in self.hosts:
                continue
            
            # Проверяем каждую VM из предыдущего размещения
            for vm_id in vm_list:
                if vm_id not in self.vms:
                    continue
                
                vm = self.vms[vm_id]
                # Проверяем, можем ли оставить VM на текущем хосте
                if self.can_host_vm(host_id, vm_id, host_resources[host_id]):
                    new_allocations[host_id].append(vm_id)
                    self.vm_to_host_map[vm_id] = host_id
                    # Обновляем доступные ресурсы хоста
                    for resource in ["cpu", "ram", "disk"]:
                        if resource in vm and resource in host_resources[host_id]:
                            host_resources[host_id][resource] -= vm[resource]
        
        # Получаем список оставшихся VM для размещения
        placed_vms = set(vm_id for host_vms in new_allocations.values() for vm_id in host_vms)
        remaining_vms = sorted(
            [vm_id for vm_id in self.vms if vm_id not in placed_vms],
            key=lambda vm_id: -(
                self.vms[vm_id].get("cpu", 0) + 
                self.vms[vm_id].get("ram", 0) + 
                self.vms[vm_id].get("disk", 0)
            )
        )
        
        # Размещаем оставшиеся VM
        for vm_id in remaining_vms:
            vm = self.vms[vm_id]
            best_host = None
            best_utilization = float('inf')
            
            # Ищем хост с наименьшей утилизацией, который может принять VM
            for host_id in self.hosts:
                if not self.can_host_vm(host_id, vm_id, host_resources[host_id]):
                    continue
                
                # Вычисляем утилизацию после размещения VM
                test_allocation = new_allocations[host_id] + [vm_id]
                utilization = self.calculate_host_utilization(host_id, test_allocation)
                
                # Выбираем хост с утилизацией ближе к целевой
                if abs(utilization - TARGET_UTILIZATION) < abs(best_utilization - TARGET_UTILIZATION):
                    best_host = host_id
                    best_utilization = utilization
            
            # Если нашли подходящий хост, размещаем VM
            if best_host:
                new_allocations[best_host].append(vm_id)
                self.vm_to_host_map[vm_id] = best_host
                # Обновляем доступные ресурсы хоста
                for resource in ["cpu", "ram", "disk"]:
                    if resource in vm and resource in host_resources[best_host]:
                        host_resources[best_host][resource] -= vm[resource]
        
        # Оптимизируем размещения через миграции
        optimized_allocations = self.optimize_migrations(new_allocations, host_resources)
        
        # Обновляем vm_to_host_map на основе оптимизированных размещений
        self.vm_to_host_map = {}
        for host_id, vm_list in optimized_allocations.items():
            for vm_id in vm_list:
                self.vm_to_host_map[vm_id] = host_id
        
        return optimized_allocations

    def generate_output(self, new_allocations: Dict[str, List[str]]) -> str:
        """Генерирует выходные данные в формате JSON."""
        # Находим миграции, сравнивая новые и предыдущие аллокации
        migrations = {}  # Словарь vm_id -> {"from": old_host, "to": new_host}
        
        # Строим обратное отображение для быстрого поиска
        old_vm_locations = {}  # vm_id -> host_id
        for host_id, vm_list in self.previous_allocations.items():
            for vm_id in vm_list:
                if vm_id in self.vms:  # Проверяем существование VM
                    old_vm_locations[vm_id] = host_id
        
        new_vm_locations = {}  # vm_id -> host_id
        for host_id, vm_list in new_allocations.items():
            for vm_id in vm_list:
                new_vm_locations[vm_id] = host_id
        
        # Находим все миграции
        for vm_id in new_vm_locations:
            old_host = old_vm_locations.get(vm_id)
            new_host = new_vm_locations.get(vm_id)
            
            # Если VM перенесена на другой хост
            if old_host is not None and new_host is not None and old_host != new_host:
                migrations[vm_id] = {
                    "from": old_host,
                    "to": new_host
                }
        
        # Формируем выходной объект
        output = {
            "allocations": new_allocations,
            "migrations": migrations,
            "allocation_failures": []  # Добавляем пустой список для allocation_failures
        }
        
        return json.dumps(output)

    def solve(self, input_str: str) -> str:
        """Решает задачу размещения виртуальных машин."""
        try:
            # Загружаем входные данные
            self.load_data(input_str)
            
            # Размещаем виртуальные машины
            new_allocations = self.place_vms()
            
            # Генерируем выходные данные
            return self.generate_output(new_allocations)
        except json.JSONDecodeError:
            # В случае ошибки разбора JSON возвращаем пустой результат
            return json.dumps({
                "allocations": {},
                "migrations": {},
                "allocation_failures": []
            })
        except Exception as e:
            # В случае любой другой ошибки возвращаем пустой результат
            return json.dumps({
                "allocations": {},
                "migrations": {},
                "allocation_failures": []
            })

if __name__ == "__main__":
    try:
        # Читаем входные данные из stdin
        input_data = sys.stdin.read().strip()
        
        if not input_data:
            # Если входные данные пустые, возвращаем пустой результат
            sys.stdout.write(json.dumps({
                "allocations": {},
                "migrations": {},
                "allocation_failures": []
            }))
            sys.stdout.flush()
            sys.exit(0)
        
        # Создаем экземпляр планировщика
        scheduler = VMScheduler()
        
        # Решаем задачу
        result = scheduler.solve(input_data)
        
        # Выводим результат в stdout
        sys.stdout.write(result)
        sys.stdout.flush()
        sys.exit(0)
    except Exception as e:
        # В случае ошибки выводим пустой результат
        sys.stdout.write(json.dumps({
            "allocations": {},
            "migrations": {},
            "allocation_failures": []
        }))
        sys.stdout.flush()
        sys.exit(0) 