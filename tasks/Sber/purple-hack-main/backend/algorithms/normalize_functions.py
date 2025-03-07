import json
from .tasks import assign_time
from .graph_dependencies.topological_sort import top_sort
from .common import raw_data


def get_base_data(is_dict=True) -> dict | list:
    return raw_data


def get_base_len(data: dict) -> int:
    return data.get("tasks").get("rows")[0].get("duration")


def base_cost_resource_fn(data_lists: dict) -> int:
    return len(data_lists.get("resources"))


def base_cost_len_fn():
    data = get_base_data()
    return get_base_len(data)


def get_base_assign_time(data):
    task_ids = []
    task_user = {}
    data = get_base_data()

    assignments = data.get("assignments", {}).get("rows", [])
    for assignment in assignments:
        task_user[assignment.get("event")] = assignment.get("resource")
        task_ids.append((assignment.get("startDate"), assignment.get("event")))

    base_assign_time = assign_time([task_id[1] for task_id in sorted(task_ids)], task_user, top_sort.blokers)
    return base_assign_time


if __name__ == "__main__":
    get_base_data()
