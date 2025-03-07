from typing import Dict, List, Any
import random

from .common import data as common_data


def create_project_role_ids(resources: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """For each project_role create a list of resourse_ids with this project_role"""

    resources_by_project_roles: Dict[str, List[str]] = {}
    for resource in resources:
        project_role = resource.get("project_role_id")
        if project_role not in resources_by_project_roles:
            resources_by_project_roles[project_role] = [resource.get("id")]
        else:
            resources_by_project_roles[project_role].append(resource.get("id"))

    return resources_by_project_roles


def count_role_ids(resources: List[Dict[str, Any]]) -> Dict[str, int]:
    resources_by_project_roles = create_project_role_ids(resources)
    role_ids_count = {}
    for role_id, resource_list in resources_by_project_roles.items():
        role_ids_count[role_id] = len(resource_list)

    return role_ids_count


def match_name_to_role_id(name: str):
    if "аналитика" in name.lower():
        return "analyst"
    if "разработка" in name.lower():
        return "developer"
    if "тестирование" in name.lower():
        return "tester"


def match_task_id_to_resourse_id(data: Dict[str, List[Dict[str, Any]]],
                                 resources_by_project_roles: Dict[str, List[str]],
                                 role_ids_local_count: Dict[str, List[str]], optimize_type="time") -> Dict[str, str]:
    """Randomly assign resourse_id to each task_id based on project_role"""

    for project_role_id, project_role_id_list in resources_by_project_roles.items():
        if optimize_type == "price" or optimize_type == "resource":
            role_resources_cnt = role_ids_local_count.get(project_role_id, 0)
            cheapest_resources = sorted([(x.get("price"), x.get("id")) for x in data.get("resources")
                                         if x.get("project_role_id") == project_role_id])[:role_resources_cnt]
            project_role_id_list = [x[1] for x in cheapest_resources]
        random.shuffle(project_role_id_list)
        resources_by_project_roles[project_role_id] = project_role_id_list

    last_ids = dict(zip(resources_by_project_roles.keys(), [0] * len(resources_by_project_roles)))
    tasks_to_resources = {}

    for task in data.get("tasks"):
        project_role_id = task.get("project_role_id")
        if project_role_id is None:
            project_role_id = match_name_to_role_id(task.get("name"))
            # project_role_id = random.choice(list(role_ids_local_count.keys()))
        last_id = last_ids.get(project_role_id)
        tasks_to_resources[task.get("id")] = resources_by_project_roles[project_role_id][last_id]
        last_ids[project_role_id] += 1

        if last_ids[project_role_id] == len(resources_by_project_roles.get(project_role_id)):
            random.shuffle(resources_by_project_roles[project_role_id])
            last_ids[project_role_id] = 0
    return tasks_to_resources
