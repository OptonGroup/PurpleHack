# Алгоритм First-Fit-Decreasing для упаковки VM
for vm_id, vm_cpu, vm_ram in vm_sizes:
    placed = False
    
    for host_id, host in sorted_hosts:
        host_cpu = host.get("cpu", 0)
        host_ram = host.get("ram", 0)
        
        # Текущая утилизация хоста
        current_cpu_util = host_allocations[host_id]["cpu"] / host_cpu if host_cpu > 0 else 1
        current_ram_util = host_allocations[host_id]["ram"] / host_ram if host_ram > 0 else 1
        
        # Утилизация после добавления VM
        new_cpu_util = (host_allocations[host_id]["cpu"] + vm_cpu) / host_cpu if host_cpu > 0 else 1
        new_ram_util = (host_allocations[host_id]["ram"] + vm_ram) / host_ram if host_ram > 0 else 1
        
        # Проверяем, что VM поместится на хост и не превысит безопасную утилизацию
        if new_cpu_util <= safe_utilization and new_ram_util <= safe_utilization:
            # Размещаем VM на этом хосте
            host_allocations[host_id]["cpu"] += vm_cpu
            host_allocations[host_id]["ram"] += vm_ram
            host_allocations[host_id]["vms"].append(vm_id)
            placed = True
            break

# Обновляем утилизацию хостов
hosts_utilization = {
    host_id: self.calculate_host_capacity(host_id, new_allocations)["max_utilization"]
    for host_id in self.hosts
}

best_host = None
best_score = float('-inf')

for host_id, current_utilization in hosts_utilization.items():
    # Проверяем, может ли хост разместить VM
    if not self.can_host_vm(host_id, vm_id):
        continue
        
    # Оцениваем новую утилизацию после размещения VM
    host = self.hosts[host_id]
    vm = self.vms[vm_id]
    
    current_cpu_used = current_utilization * host.get("cpu", 0) if current_utilization > 0 else 0
    current_ram_used = current_utilization * host.get("ram", 0) if current_utilization > 0 else 0
    
    new_cpu_utilization = (current_cpu_used + vm.get("cpu", 0)) / host.get("cpu", 0)
    new_ram_utilization = (current_ram_used + vm.get("ram", 0)) / host.get("ram", 0)
    
    new_utilization = max(new_cpu_utilization, new_ram_utilization)
    
    # Оцениваем размещение с учетом TARGET_UTILIZATION
    # Стремимся именно к оптимальной утилизации (0.807197)
    utilization_score = -abs(new_utilization - TARGET_UTILIZATION)
    
    # Штраф за перегрузку хоста
    overload_penalty = 0
    if new_utilization > UPPER_THRESHOLD:
        overload_penalty = (new_utilization - UPPER_THRESHOLD) * 10
    
    # Бонус за консолидацию (предпочитаем хосты, которые уже используются)
    consolidation_bonus = 0
    if current_utilization > 0:
        consolidation_bonus = 0.2
    
    # Финальная оценка
    score = utilization_score - overload_penalty + consolidation_bonus
    
    if score > best_score:
        best_score = score
        best_host = host_id 

    def consolidate_vms(self, vm_to_host_map=None):
        """Консолидирует VM на минимальном количестве хостов для получения бонусов за выключенные хосты.
        
        Args:
            vm_to_host_map: Текущие размещения VM. Если None, используется self.vm_to_host_map
            
        Returns:
            Tuple[Dict[str, str], List[Dict[str, str]]]: Новые размещения и список миграций
        """
        # Используем текущий vm_to_host_map, если не предоставлен другой
        if vm_to_host_map is None:
            vm_to_host_map = self.vm_to_host_map
            
        # Вызываем оптимизацию выключения хостов для консолидации VM
        return self._optimize_host_shutdown(vm_to_host_map)

    def place_vms(self, vm_sizes, sorted_hosts, host_allocations, safe_utilization, TARGET_UTILIZATION, UPPER_THRESHOLD):
        """Размещает VM на хостах, используя алгоритм First-Fit-Decreasing.
        
        Args:
            vm_sizes: Список размеров VM
            sorted_hosts: Отсортированный список хостов
            host_allocations: Словарь с текущими размещениями VM на хостах
            safe_utilization: Безопасная утилизация хоста
            TARGET_UTILIZATION: Целевая утилизация
            UPPER_THRESHOLD: Верхняя граница утилизации для перегрузки
            
        Returns:
            Tuple[Dict[str, List[str]], List[Dict[str, str]]]: Новые размещения и список миграций
        """
        start_time = time.time()
        vm_to_host_map = {}
        migrations = []
        allocations = {host: [] for host in sorted_hosts}

        for vm_id, vm_cpu, vm_ram in vm_sizes:
            placed = False
            
            for host_id, host in sorted_hosts:
                host_cpu = host.get("cpu", 0)
                host_ram = host.get("ram", 0)
                
                # Текущая утилизация хоста
                current_cpu_util = host_allocations[host_id]["cpu"] / host_cpu if host_cpu > 0 else 1
                current_ram_util = host_allocations[host_id]["ram"] / host_ram if host_ram > 0 else 1
                
                # Утилизация после добавления VM
                new_cpu_util = (host_allocations[host_id]["cpu"] + vm_cpu) / host_cpu if host_cpu > 0 else 1
                new_ram_util = (host_allocations[host_id]["ram"] + vm_ram) / host_ram if host_ram > 0 else 1
                
                # Проверяем, что VM поместится на хост и не превысит безопасную утилизацию
                if new_cpu_util <= safe_utilization and new_ram_util <= safe_utilization:
                    # Размещаем VM на этом хосте
                    host_allocations[host_id]["cpu"] += vm_cpu
                    host_allocations[host_id]["ram"] += vm_ram
                    host_allocations[host_id]["vms"].append(vm_id)
                    vm_to_host_map[vm_id] = host_id
                    placed = True
                    break

            if not placed:
                # Если VM не поместилась на текущих хостах, ищем место на другом хосте
                for host_id, host in sorted_hosts:
                    host_cpu = host.get("cpu", 0)
                    host_ram = host.get("ram", 0)
                    
                    # Проверяем, что хост может принять VM
                    if self.can_host_vm(host_id, vm_id):
                        # Размещаем VM на этом хосте
                        host_allocations[host_id]["cpu"] += vm_cpu
                        host_allocations[host_id]["ram"] += vm_ram
                        host_allocations[host_id]["vms"].append(vm_id)
                        vm_to_host_map[vm_id] = host_id
                        placed = True
                        break

            if not placed:
                # Если VM не поместилась на текущих хостах и не может быть размещена на другом, создаем новую миграцию
                migrations.append({"vm_id": vm_id, "from_host_id": None, "to_host_id": None})

        # Обновляем утилизацию хостов
        hosts_utilization = {
            host_id: self.calculate_host_capacity(host_id, host_allocations)["max_utilization"]
            for host_id in self.hosts
        }

        best_host = None
        best_score = float('-inf')

        for host_id, current_utilization in hosts_utilization.items():
            # Проверяем, может ли хост разместить VM
            if not self.can_host_vm(host_id, vm_id):
                continue
            
            # Оцениваем новую утилизацию после размещения VM
            host = self.hosts[host_id]
            vm = self.vms[vm_id]
            
            current_cpu_used = current_utilization * host.get("cpu", 0) if current_utilization > 0 else 0
            current_ram_used = current_utilization * host.get("ram", 0) if current_utilization > 0 else 0
            
            new_cpu_utilization = (current_cpu_used + vm.get("cpu", 0)) / host.get("cpu", 0)
            new_ram_utilization = (current_ram_used + vm.get("ram", 0)) / host.get("ram", 0)
            
            new_utilization = max(new_cpu_utilization, new_ram_utilization)
            
            # Оцениваем размещение с учетом TARGET_UTILIZATION
            # Стремимся именно к оптимальной утилизации (0.807197)
            utilization_score = -abs(new_utilization - TARGET_UTILIZATION)
            
            # Штраф за перегрузку хоста
            overload_penalty = 0
            if new_utilization > UPPER_THRESHOLD:
                overload_penalty = (new_utilization - UPPER_THRESHOLD) * 10
            
            # Бонус за консолидацию (предпочитаем хосты, которые уже используются)
            consolidation_bonus = 0
            if current_utilization > 0:
                consolidation_bonus = 0.2
            
            # Финальная оценка
            score = utilization_score - overload_penalty + consolidation_bonus
            
            if score > best_score:
                best_score = score
                best_host = host_id 

        # Обновляем vm_to_host_map
        self.vm_to_host_map = vm_to_host_map
        
        # Консолидируем VM если есть такая возможность
        new_vm_to_host_map, migrations_from_consolidation = self.consolidate_vms(self.vm_to_host_map)
        
        # Если были миграции при консолидации, обновляем результат
        if migrations_from_consolidation:
            # Добавляем миграции от консолидации
            migrations.extend(migrations_from_consolidation)
            
            # Обновляем размещения на основе нового маппинга
            new_allocations = {host: [] for host in sorted_hosts}
            for vm, host in new_vm_to_host_map.items():
                if host in new_allocations:
                    new_allocations[host].append(vm)
            allocations = new_allocations
        
        end_time = time.time()
        # Комментируем вывод отладочной информации
        # print(f"Place VMs execution time: {end_time - start_time:.4f} seconds", file=sys.stderr)

        return allocations, migrations 