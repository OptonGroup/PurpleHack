#!/usr/bin/env python3
import json
import sys
from typing import Dict, List, Tuple, Set, Any, Optional
import time

# Константы для оптимизации
TARGET_UTILIZATION = 0.8  # Целевая утилизация хостов (80%)
UPPER_THRESHOLD = 0.9  # Верхний порог утилизации
LOWER_THRESHOLD = 0.2  # Нижний порог утилизации
MAX_MIGRATIONS = 3  # Максимальное число миграций за раунд (снижено для уменьшения штрафа)
MIGRATION_PENALTY_FACTOR = 1.0  # Коэффициент штрафа за миграцию

class VMScheduler:
    def __init__(self):
        self.hosts = {}  # имя хоста -> {cpu, ram}
        self.vms = {}  # имя VM -> {cpu, ram}
        self.allocations = {}  # имя хоста -> список VM
        self.previous_allocations = {}  # имя хоста -> список VM из предыдущего раунда
        self.host_zero_utilization_count = {}  # отслеживание хостов с нулевой нагрузкой
        self.vm_to_host_map = {}  # Отображение VM -> Host из предыдущих аллокаций
        self.log = []  # Логи для дебага
        
    def load_data(self, input_str: str) -> None:
        try:
            data = json.loads(input_str)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON input")

        # Получаем данные о хостах
        self.hosts = data.get("hosts", {})
        
        # Получаем данные о виртуальных машинах
        # Проверяем оба возможных ключа: 'vms' и 'virtual_machines'
        if "vms" in data:
            self.vms = data.get("vms", {})
        elif "virtual_machines" in data:
            self.vms = data.get("virtual_machines", {})
        
        # Получаем данные о предыдущих аллокациях
        self.previous_allocations = data.get("allocations", {})
        
        # Обрабатываем информацию о diff, если она есть
        if "diff" in data:
            diff = data.get("diff", {})
            
            # Обработка добавленных VM
            if "add" in diff:
                add_data = diff.get("add", {})
                if "virtual_machines" in add_data:
                    # Это список новых VM, которые нужно добавить в размещение
                    pass  # Они уже должны быть в self.vms
            
            # Обработка удаленных VM
            if "remove" in diff:
                remove_data = diff.get("remove", {})
                if "virtual_machines" in remove_data:
                    # Список VM для удаления - удаляем их из текущих распределений
                    vms_to_remove = remove_data.get("virtual_machines", [])
                    for vm_id in vms_to_remove:
                        # Удаляем VM из размещений, если она там есть
                        for host_id in self.previous_allocations:
                            if vm_id in self.previous_allocations[host_id]:
                                self.previous_allocations[host_id].remove(vm_id)
        
        # Инициализируем или обновляем счетчики хостов с нулевой утилизацией
        for host_id in self.hosts:
            if host_id not in self.host_zero_utilization_count:
                self.host_zero_utilization_count[host_id] = 0
        
        # Строим отображение VM -> Host из предыдущих аллокаций
        self.vm_to_host_map = {}
        for host_id, vm_ids in self.previous_allocations.items():
            for vm_id in vm_ids:
                if vm_id in self.vms:  # Проверяем существование VM
                    self.vm_to_host_map[vm_id] = host_id

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
        
        return can_host

    def calculate_host_utilization(self, host_id: str, allocation: List[str] = None) -> float:
        """Вычисляет утилизацию хоста на основе размещенных VM."""
        if host_id not in self.hosts:
            return 0.0
        
        host = self.hosts[host_id]
        if not allocation:
            allocation = self.previous_allocations.get(host_id, [])
        
        used_resources = {"cpu": 0, "ram": 0}
        total_resources = {"cpu": host.get("cpu", 0), "ram": host.get("ram", 0)}
        
        # Добавляем disk только если он есть
        if "disk" in host:
            used_resources["disk"] = 0
            total_resources["disk"] = host["disk"]
        
        for vm_id in allocation:
            if vm_id in self.vms:
                vm = self.vms[vm_id]
                used_resources["cpu"] += vm.get("cpu", 0)
                used_resources["ram"] += vm.get("ram", 0)
                if "disk" in used_resources and "disk" in vm:
                    used_resources["disk"] += vm["disk"]
        
        # Вычисляем показатель утилизации с приоритетом CPU и RAM
        # Увеличили вес CPU для лучшей балансировки
        cpu_util = used_resources["cpu"] / total_resources["cpu"] if total_resources["cpu"] > 0 else 0
        ram_util = used_resources["ram"] / total_resources["ram"] if total_resources["ram"] > 0 else 0
        
        if "disk" in used_resources and "disk" in total_resources and total_resources["disk"] > 0:
            disk_util = used_resources["disk"] / total_resources["disk"]
            # Среднее значение утилизации (с приоритетом CPU и RAM)
            return (cpu_util * 3 + ram_util * 2 + disk_util) / 6.0
        else:
            # Если нет disk, считаем только по cpu и ram с увеличенным весом CPU
            return (cpu_util * 3 + ram_util * 2) / 5.0

    def estimate_vm_utilization_impact(self, vm_id: str, host_id: str) -> float:
        """Оценивает, насколько VM повлияет на утилизацию хоста."""
        if vm_id not in self.vms or host_id not in self.hosts:
            return 0.0
        
        host = self.hosts[host_id]
        vm = self.vms[vm_id]
        
        # Расчет влияния VM на утилизацию хоста
        cpu_impact = vm.get("cpu", 0) / host.get("cpu", 1) if host.get("cpu", 0) > 0 else 0
        ram_impact = vm.get("ram", 0) / host.get("ram", 1) if host.get("ram", 0) > 0 else 0
        
        # Средневзвешенное с приоритетом CPU
        return (cpu_impact * 3 + ram_impact * 2) / 5.0

    def get_resource_efficiency(self, vm_id: str, host_id: str) -> float:
        """Оценивает эффективность размещения VM на хосте, учитывая баланс ресурсов."""
        if vm_id not in self.vms or host_id not in self.hosts:
            return 0.0
            
        vm = self.vms[vm_id]
        host = self.hosts[host_id]
        
        # Если хост не может вместить VM, возвращаем 0
        if (vm.get("cpu", 0) > host.get("cpu", 0) or 
            vm.get("ram", 0) > host.get("ram", 0)):
            return 0.0
            
        # Вычисляем, насколько эффективно VM использует ресурсы хоста
        # Чем ближе к целевой утилизации, тем лучше
        current_utilization = self.calculate_host_utilization(host_id)
        new_util = current_utilization + self.estimate_vm_utilization_impact(vm_id, host_id)
        
        # Штраф за чрезмерную утилизацию (выше TARGET_UTILIZATION)
        if new_util > TARGET_UTILIZATION:
            overutil_penalty = (new_util - TARGET_UTILIZATION) * 5
            return 1.0 - overutil_penalty
        
        # Бонус за приближение к целевой утилизации
        util_diff = abs(TARGET_UTILIZATION - new_util)
        return 1.0 - (util_diff / TARGET_UTILIZATION)

    def place_vms(self) -> Dict[str, List[str]]:
        """Размещает виртуальные машины на хостах, учитывая предыдущие размещения и миграции."""
        # Создаем новые распределения, начиная с предыдущих
        new_allocations = {host_id: [] for host_id in self.hosts}
        allocation_failures = []
        
        # Копируем текущие аллокации (только существующие VM)
        for host_id, vm_list in self.previous_allocations.items():
            # Пропускаем хосты, которых больше нет
            if host_id not in self.hosts:
                continue
            
            new_allocations[host_id] = [vm for vm in vm_list if vm in self.vms]
        
        # Обновляем отображение VM -> Host
        self.vm_to_host_map = {}
        for host_id, vm_ids in new_allocations.items():
            for vm_id in vm_ids:
                self.vm_to_host_map[vm_id] = host_id
        
        # Рассчитываем текущие ресурсы хостов
        host_resources = {}
        for host_id in self.hosts:
            host_resources[host_id] = self.calculate_host_capacity(host_id, new_allocations)
        
        # Сначала размещаем новые VM, которых не было раньше
        new_vms = [vm_id for vm_id in self.vms if vm_id not in self.vm_to_host_map]
        
        # Быстрая оценка для сортировки хостов
        host_utilization = {}
        for host_id in self.hosts:
            host_utilization[host_id] = self.calculate_host_utilization(host_id, new_allocations[host_id])
        
        # Сортируем хосты с улучшенной эвристикой
        sorted_hosts = sorted(
            self.hosts.keys(),
            key=lambda host_id: (
                # Сначала пытаемся использовать хосты близкие к целевой утилизации, но не превышающие её
                abs(TARGET_UTILIZATION - host_utilization[host_id]) if host_utilization[host_id] <= TARGET_UTILIZATION else 100,
                # Затем хосты с максимальной утилизацией, не превышающей целевую
                -host_utilization[host_id] if host_utilization[host_id] <= TARGET_UTILIZATION else 0,
                # Затем хосты с минимальным количеством нулевых утилизаций
                self.host_zero_utilization_count.get(host_id, 0)
            )
        )
        
        # Улучшенная сортировка VM - учитываем как размер, так и соотношение CPU/RAM
        sorted_vms = sorted(
            new_vms,
            key=lambda vm_id: (
                # Сначала по общему объему ресурсов (от большего к меньшему)
                -(self.vms[vm_id].get("cpu", 0) + self.vms[vm_id].get("ram", 0) / 10),
                # Затем по соотношению CPU/RAM (приоритет балансу)
                abs(self.vms[vm_id].get("cpu", 0) / max(1, self.vms[vm_id].get("ram", 1)) - 0.5)
            )
        )
        
        # Размещаем новые VM с улучшенной эвристикой
        for vm_id in sorted_vms:
            vm = self.vms[vm_id]
            placed = False
            
            # Сначала пытаемся найти оптимальное размещение
            best_host = None
            best_score = -1
            
            for host_id in sorted_hosts:
                if self.can_host_vm(host_id, vm_id, host_resources[host_id]):
                    # Вычисляем эффективность размещения
                    score = self.get_resource_efficiency(vm_id, host_id)
                    
                    if score > best_score:
                        best_score = score
                        best_host = host_id
            
            # Если нашли подходящий хост, размещаем VM
            if best_host:
                new_allocations[best_host].append(vm_id)
                self.vm_to_host_map[vm_id] = best_host
                
                # Обновляем доступные ресурсы хоста
                host_resources[best_host]["cpu"] -= vm.get("cpu", 0)
                host_resources[best_host]["ram"] -= vm.get("ram", 0)
                if "disk" in host_resources[best_host] and "disk" in vm:
                    host_resources[best_host]["disk"] -= vm["disk"]
                
                # Обновляем утилизацию хоста
                host_utilization[best_host] = self.calculate_host_utilization(best_host, new_allocations[best_host])
                
                placed = True
            
            # Если не удалось разместить обычным способом, пробуем найти место с миграцией
            if not placed:
                migration_successful = self.try_migration_placement(vm_id, new_allocations, host_resources, host_utilization)
                if not migration_successful:
                    # Не удалось разместить VM даже с миграцией
                    allocation_failures.append(vm_id)
                    self.log.append(f"Failed to allocate VM {vm_id} with resources CPU:{vm.get('cpu', 0)}, RAM:{vm.get('ram', 0)}")
        
        # Обновляем счетчики нулевой утилизации и анализируем загрузку
        for host_id in self.hosts:
            utilization = self.calculate_host_utilization(host_id, new_allocations[host_id])
            if utilization < 0.01:  # Почти нулевая утилизация
                self.host_zero_utilization_count[host_id] = self.host_zero_utilization_count.get(host_id, 0) + 1
            else:
                self.host_zero_utilization_count[host_id] = 0
        
        # Оптимизируем через перемещения для выключения неиспользуемых хостов
        self.optimize_allocations(new_allocations, host_resources)
        
        # Сохраняем информацию о неразмещенных VM
        self.allocation_failures = allocation_failures
        
        return new_allocations

    def try_migration_placement(self, vm_id: str, allocations: Dict[str, List[str]], 
                                host_resources: Dict[str, Dict], host_utilization: Dict[str, float]) -> bool:
        """Пытается разместить VM путем миграции других VM."""
        vm = self.vms[vm_id]
        
        # Сортируем хосты по утилизации (от наименьшей к наибольшей)
        candidate_hosts = sorted(
            [h for h in self.hosts if allocations[h]],  # Только хосты с VM
            key=lambda h: host_utilization[h]
        )
        
        for source_host in candidate_hosts:
            # Сортируем VM на хосте по размеру (от наименьших к наибольшим)
            source_vms = sorted(
                allocations[source_host],
                key=lambda vm: (
                    self.vms[vm].get("cpu", 0) + 
                    self.vms[vm].get("ram", 0)
                )
            )
            
            # Пробуем найти VM для миграции
            for move_vm in source_vms:
                vm_data = self.vms[move_vm]
                
                # Оцениваем, стоит ли пытаться мигрировать VM
                if (vm_data.get("cpu", 0) < vm.get("cpu", 0) and 
                    vm_data.get("ram", 0) < vm.get("ram", 0)):
                    # Мигрируемая VM меньше, чем VM, которую мы хотим разместить
                    # Это может быть хорошим кандидатом для миграции
                    
                    # Временно удаляем VM с исходного хоста
                    allocations[source_host].remove(move_vm)
                    
                    # Проверяем, появилось ли место для новой VM
                    current_capacity = self.calculate_host_capacity(source_host, allocations)
                    
                    # Проверяем, можем ли разместить новую VM
                    can_place = (
                        current_capacity.get("cpu", 0) >= vm.get("cpu", 0) and
                        current_capacity.get("ram", 0) >= vm.get("ram", 0)
                    )
                    
                    if "disk" in current_capacity and "disk" in vm:
                        can_place = can_place and current_capacity["disk"] >= vm["disk"]
                    
                    if can_place:
                        # Ищем хост для перемещаемой VM
                        # Сортируем хосты так, чтобы предпочитать те, которые ближе к целевой утилизации
                        target_hosts = sorted(
                            [h for h in self.hosts if h != source_host],
                            key=lambda h: abs(TARGET_UTILIZATION - (
                                host_utilization[h] + self.estimate_vm_utilization_impact(move_vm, h)
                            ))
                        )
                        
                        for target_host in target_hosts:
                            target_capacity = self.calculate_host_capacity(target_host, allocations)
                            
                            can_move = (
                                target_capacity.get("cpu", 0) >= vm_data.get("cpu", 0) and
                                target_capacity.get("ram", 0) >= vm_data.get("ram", 0)
                            )
                            
                            if "disk" in target_capacity and "disk" in vm_data:
                                can_move = can_move and target_capacity["disk"] >= vm_data["disk"]
                            
                            if can_move:
                                # Размещаем мигрируемую VM на целевом хосте
                                allocations[target_host].append(move_vm)
                                # Размещаем новую VM на исходном хосте
                                allocations[source_host].append(vm_id)
                                
                                # Обновляем ресурсы и утилизацию
                                for h in self.hosts:
                                    host_resources[h] = self.calculate_host_capacity(h, allocations)
                                    host_utilization[h] = self.calculate_host_utilization(h, allocations[h])
                                
                                self.log.append(f"Migration: moved {move_vm} from {source_host} to {target_host} to place {vm_id}")
                                return True
                    
                    # Если не нашли место для мигрируемой VM или не можем разместить новую VM,
                    # возвращаем мигрируемую VM на исходный хост
                    allocations[source_host].append(move_vm)
        
        return False

    def optimize_allocations(self, allocations: Dict[str, List[str]], host_resources: Dict[str, Dict]) -> None:
        """Оптимизирует размещение VM через миграции для выключения неиспользуемых хостов."""
        # Подсчитываем количество миграций
        migrations = 0
        
        # Определяем хосты с низкой и высокой утилизацией
        all_hosts = list(self.hosts.keys())
        host_utilization = {h: self.calculate_host_utilization(h, allocations[h]) for h in all_hosts}
        
        # Сортируем хосты по нагрузке (сначала хосты с низкой загрузкой)
        # Приоритизируем хосты с длительной нулевой утилизацией для ранней консолидации
        hosts_by_utilization = sorted(
            all_hosts,
            key=lambda h: (
                min(1, self.host_zero_utilization_count.get(h, 0) / 5),  # Приоритет хостам с нулевой утилизацией
                host_utilization[h]  # Затем сортировка по утилизации
            )
        )
        
        # Сначала пытаемся выключить хосты с нулевой утилизацией
        for source_idx, source_host in enumerate(hosts_by_utilization):
            if migrations >= MAX_MIGRATIONS:
                break
                
            # Пропускаем хосты с высокой утилизацией
            if host_utilization[source_host] > LOWER_THRESHOLD:
                continue
                
            # Если на хосте нет VM, пропускаем его
            if not allocations[source_host]:
                continue
            
            # Проверяем, стоит ли консолидировать VM с этого хоста
            should_consolidate = (
                host_utilization[source_host] < LOWER_THRESHOLD or
                self.host_zero_utilization_count.get(source_host, 0) >= 3  # После 3 раундов начинаем консолидацию
            )
            
            if should_consolidate:
                # Сортируем VM по размеру - сначала большие, чтобы не было фрагментации
                vms_to_migrate = sorted(
                    allocations[source_host],
                    key=lambda vm: -(
                        self.vms[vm].get("cpu", 0) + 
                        self.vms[vm].get("ram", 0)
                    )
                )
                
                # Для каждой VM ищем новый хост
                for vm_id in vms_to_migrate[:]:  # Копия списка для безопасного перебора
                    if migrations >= MAX_MIGRATIONS:
                        break
                        
                    vm = self.vms[vm_id]
                    
                    # Ищем подходящий хост, стремясь к целевой утилизации
                    candidate_hosts = [h for h in all_hosts if h != source_host]
                    # Сортировка для приоритета хостов с утилизацией близкой к целевой
                    candidate_hosts.sort(
                        key=lambda h: abs(
                            TARGET_UTILIZATION - (
                                host_utilization[h] + self.estimate_vm_utilization_impact(vm_id, h)
                            )
                        )
                    )
                    
                    for target_host in candidate_hosts:
                        # Проверяем, может ли хост принять VM
                        target_capacity = self.calculate_host_capacity(target_host, allocations)
                        
                        can_move = (
                            target_capacity.get("cpu", 0) >= vm.get("cpu", 0) and
                            target_capacity.get("ram", 0) >= vm.get("ram", 0)
                        )
                        
                        if "disk" in target_capacity and "disk" in vm:
                            can_move = can_move and target_capacity["disk"] >= vm["disk"]
                        
                        # Проверяем, не приведет ли миграция к перегрузке целевого хоста
                        target_utilization = host_utilization[target_host]
                        vm_impact = self.estimate_vm_utilization_impact(vm_id, target_host)
                        new_utilization = target_utilization + vm_impact
                        
                        # Не мигрируем, если утилизация превысит верхний порог
                        if can_move and new_utilization <= UPPER_THRESHOLD:
                            # Мигрируем VM только если она все еще на исходном хосте
                            if vm_id in allocations[source_host]:
                                allocations[source_host].remove(vm_id)
                                allocations[target_host].append(vm_id)
                                
                                # Обновляем ресурсы и утилизацию
                                host_resources[target_host] = self.calculate_host_capacity(target_host, allocations)
                                host_resources[source_host] = self.calculate_host_capacity(source_host, allocations)
                                host_utilization[target_host] = self.calculate_host_utilization(target_host, allocations[target_host])
                                host_utilization[source_host] = self.calculate_host_utilization(source_host, allocations[source_host])
                                
                                migrations += 1
                                self.log.append(f"Consolidation: moved {vm_id} from {source_host} to {target_host}")
                            break
                
                # Если все VM перенесены, хост будет выключен
                if not allocations[source_host]:
                    self.log.append(f"Host {source_host} is now empty and can be turned off")
        
        # Балансируем перегруженные хосты
        overutilized_hosts = [h for h in all_hosts if host_utilization[h] > UPPER_THRESHOLD]
        overutilized_hosts.sort(key=lambda h: -host_utilization[h])
        
        for source_host in overutilized_hosts:
            if migrations >= MAX_MIGRATIONS:
                break
                
            # Сортируем VM на хосте по размеру - сначала маленькие, чтобы минимизировать миграции
            vms_on_host = sorted(
                allocations[source_host],
                key=lambda vm: (
                    self.vms[vm].get("cpu", 0) + 
                    self.vms[vm].get("ram", 0)
                )
            )
            
            # Пробуем найти наименьшую VM для миграции
            for vm_id in vms_on_host[:]:  # Копия списка для безопасного перебора
                if migrations >= MAX_MIGRATIONS:
                    break
                    
                vm = self.vms[vm_id]
                
                # Выбираем хосты с низкой загрузкой
                target_hosts = [h for h in all_hosts if h != source_host and host_utilization[h] < TARGET_UTILIZATION]
                # Сортировка для приоритета хостов с утилизацией близкой к целевой
                target_hosts.sort(
                    key=lambda h: abs(
                        TARGET_UTILIZATION - (
                            host_utilization[h] + self.estimate_vm_utilization_impact(vm_id, h)
                        )
                    )
                )
                
                for target_host in target_hosts:
                    target_capacity = self.calculate_host_capacity(target_host, allocations)
                    
                    can_move = (
                        target_capacity.get("cpu", 0) >= vm.get("cpu", 0) and
                        target_capacity.get("ram", 0) >= vm.get("ram", 0)
                    )
                    
                    if "disk" in target_capacity and "disk" in vm:
                        can_move = can_move and target_capacity["disk"] >= vm["disk"]
                    
                    if can_move:
                        # Мигрируем VM только если она все еще на исходном хосте
                        if vm_id in allocations[source_host]:
                            allocations[source_host].remove(vm_id)
                            allocations[target_host].append(vm_id)
                            
                            # Обновляем ресурсы и утилизацию
                            host_resources[target_host] = self.calculate_host_capacity(target_host, allocations)
                            host_resources[source_host] = self.calculate_host_capacity(source_host, allocations)
                            host_utilization[target_host] = self.calculate_host_utilization(target_host, allocations[target_host])
                            host_utilization[source_host] = self.calculate_host_utilization(source_host, allocations[source_host])
                            
                            migrations += 1
                            self.log.append(f"Load balancing: moved {vm_id} from {source_host} to {target_host}")
                            
                            # Если утилизация теперь в приемлемом диапазоне, переходим к следующему хосту
                            if host_utilization[source_host] <= UPPER_THRESHOLD:
                                break
                
                if host_utilization[source_host] <= UPPER_THRESHOLD:
                    break

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
        for vm_id in self.vms:
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
            "allocation_failures": getattr(self, "allocation_failures", [])
        }
        
        return json.dumps(output)

    def solve(self, input_str: str) -> str:
        """Основной метод для решения задачи планирования VM."""
        start_time = time.time()
        
        # Сбрасываем логи
        self.log = []
        
        # Обрабатываем входные данные
        self.load_data(input_str)
        
        # Размещаем виртуальные машины
        new_allocations = self.place_vms()
        
        # Генерируем выходные данные
        output = self.generate_output(new_allocations)
        
        # Сохраняем новые аллокации для следующего раунда
        self.previous_allocations = new_allocations
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Записываем логи выполнения для отладки
        # print(f"Execution time: {execution_time:.4f} seconds", file=sys.stderr)
        # print(f"Logs: {len(self.log)} entries", file=sys.stderr)
        # for log_entry in self.log:
        #     print(f"  {log_entry}", file=sys.stderr)
        
        return output


if __name__ == "__main__":
    # Читаем входные данные
    input_str = sys.stdin.read()
    
    # Создаем экземпляр планировщика и решаем задачу
    scheduler = VMScheduler()
    output = scheduler.solve(input_str)
    
    # Выводим результат
    print(output) 