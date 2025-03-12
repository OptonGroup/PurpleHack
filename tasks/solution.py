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