from collections import defaultdict


def create_graph(dependencies, is_task) -> dict:
    """Create base graph only with tasks dependencies"""

    blockers = defaultdict(list)
    graph = defaultdict(list)
    for edge in dependencies:
        from_id = edge.get("from")
        to_id = edge.get("to")
        if is_task.get(from_id) and is_task.get(to_id):
            if from_id not in graph:
                graph[from_id] = [to_id]
            else:
                graph[from_id].append(to_id)
            blockers[to_id].append(from_id)

    return graph, blockers


def get_tasks(is_task) -> set:
    tasks = set()
    for task_id, is_task_flag in is_task.items():
        if is_task_flag:
            tasks.add(task_id)

    return tasks


def get_all_roots(dependencies, is_task) -> set:
    roots = get_tasks(is_task)
    for edge in dependencies:
        to_id = edge.get("to")
        if to_id in roots:
            roots.remove(to_id)

    return roots


def get_all_leaves(dependencies, is_task) -> set:
    leaves = get_tasks(is_task)
    for edge in dependencies:
        from_id = edge.get("from")
        if from_id in leaves:
            leaves.remove(from_id)

    return leaves


def create_roots_leaves_graph(task_tree, is_task, roots, leaves):
    """For each project create graph of roots and leaves"""

    roots_gr = defaultdict(list)
    leaves_gr = defaultdict(list)

    for r in roots:
        roots_gr[r] = [r]

    for l in leaves:
        leaves_gr[l] = [l]

    for project_id, task_list in task_tree.items():
        for task_id in task_list:
            if is_task.get(task_id):  # если таска, то добавляем в список проекта
                if task_id in roots:
                    if project_id not in roots_gr:
                        roots_gr[project_id] = [task_id]
                    else:
                        roots_gr[project_id].append(task_id)
                if task_id in leaves:
                    if project_id not in leaves_gr:
                        leaves_gr[project_id] = [task_id]
                    else:
                        leaves_gr[project_id].append(task_id)

    return roots_gr, leaves_gr


def projects_tree_dfs(project_id, par, task_tree, roots_gr, leaves_gr):
    """Dynamic programming that reculsively recalculates roots and leaves for every project and task"""

    for child_id in task_tree.get(project_id, []):
        if child_id != par:
            projects_tree_dfs(child_id, project_id, task_tree, roots_gr, leaves_gr)
            roots_gr[project_id].extend(roots_gr[child_id])
            leaves_gr[project_id].extend(leaves_gr[child_id])


def dp_roots_and_leaves(task_tree, is_task, roots, leaves):
    roots_gr, leaves_gr = create_roots_leaves_graph(task_tree, is_task, roots, leaves)
    root_id = task_tree.get("*")[0]
    projects_tree_dfs(root_id, "-1", task_tree, roots_gr, leaves_gr)
    return roots_gr, leaves_gr


def extend_graph(graph, dependencies, roots_gr, leaves_gr, blockers) -> dict:
    """Draws edges connecting blocker-project and blocked-project"""

    for edge in dependencies:
        from_id = edge.get("from")
        to_id = edge.get("to")
        if from_id == "7345363831820451841" and to_id == "7345357956305190913":
            x = 0
        for leaf in leaves_gr.get(from_id, []):
            for root in roots_gr.get(to_id, []):
                graph[leaf].append(root)
                blockers[root].append(leaf)


    return graph
