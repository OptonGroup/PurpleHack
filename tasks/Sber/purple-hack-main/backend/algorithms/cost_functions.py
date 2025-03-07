from datetime import timedelta, datetime
from typing import Callable, Union, List, Dict, Any
from .common import data, data_lists
from .normalize_functions import get_base_assign_time, base_cost_len_fn, base_cost_resource_fn
import json
from .common import get_output

Number = Union[float, int]


def calculate_normalized_values(base_assign_time: List[Dict[str, Any]]) -> list:
    normalized_values = [base_cost_len_fn(), base_cost_resource_fn(data_lists),
                         cost_fn(base_assign_time)]
    print(normalized_values)

    return normalized_values


def count_working_days(start_date: datetime, end_date: datetime) -> int:
    current_date = start_date
    working_days = 0

    while current_date <= end_date:
        if current_date.weekday() < 5:
            working_days += 1
        current_date += timedelta(days=1)

    return working_days


def cost_len_fn(infos: List[Dict[str, Any]]) -> int:
    total_cost = 0
    start_date = infos[0].get("start_date")
    end_date = infos[0].get("end_date")

    for info in infos:
        start_date = min(start_date, info.get("start_date"))
        end_date = max(end_date, info.get("end_date"))
    return (end_date - start_date).days


def cost_resource_fn(infos: List[Dict[str, Any]]) -> int:
    resourses = set()

    for info in infos:
        resourses.add(info["resource_id"])

    return len(resourses)


def cost_fn(infos: List[Dict[str, Any]]) -> Number:
    total_cost = 0
    print("info", infos)
    for task in infos:
        resource_id = task["resource_id"]
        total_cost += task["effort"] * data["resources"][resource_id]["price"]

    return total_cost


def combined_cost_fn(fn1: Callable[[List[Dict[str, Any]]], int],
                     fn2: Callable[[List[Dict[str, Any]]], int],
                     fn3: Callable[[List[Dict[str, Any]]], int],
                     k1: float,
                     k2: float,
                     k3: float,
                     ) -> Callable[[List[Dict[str, Any]]], Number]:
    def _fn(infos: List[Dict[str, Any]]) -> Number:
        return k1 * fn1(infos) / normalized_values[0] + k2 * fn2(infos) / normalized_values[1] + \
            k3 * fn3(infos) / normalized_values[2]

    return _fn


normalized_values = []


def init():
    normalized_values.clear()
    normalized_values.extend(calculate_normalized_values(get_base_assign_time(data)))
