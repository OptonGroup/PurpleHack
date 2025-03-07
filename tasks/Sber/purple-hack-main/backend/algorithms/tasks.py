from pprint import pprint

from .common import data, calendar
from datetime import datetime
from datetime import timedelta
from collections import defaultdict
from copy import deepcopy


def get_user_hour(user_id: str, date: str):
    return calendar[user_id][date]


def assign_time(task_ids: list, task_user: dict, graph: dict = {}) -> list[dict]:
    res = []
    parsed_data = deepcopy(data)
    init_date = datetime.strptime(parsed_data["start_date"], "%Y-%m-%d")
    step = timedelta(days=1)
    user_time = defaultdict(lambda: init_date)

    for task_id in task_ids:
        task: dict = parsed_data["tasks"][task_id]
        effort_time = task["effort"]
        user_id = task_user[task_id]

        duration = 0

        blocker_ids = graph.get(task_id, [])
        try:
            for blocker_id in blocker_ids:
                user_time[user_id] = max(user_time[user_id], parsed_data["tasks"][blocker_id]["end_date"])
        except Exception:
            print("!!!")
            print(task_ids)
            print(blocker_ids)
            print("task_id", task_id)
            print("blocker_id", blocker_id)
            pprint(graph)
        start_date = user_time[user_id]
        while effort_time > 0:
            if user_time[user_id].weekday() >= 5:
                user_time[user_id] += step
                continue
            hours = get_user_hour(task_id, user_time[user_id].strftime("%Y-%m-%d"))
            effort_time -= hours
            if hours:
                duration += 1
            user_time[user_id] += step
        end_date = user_time[user_id]
        task.update(
            {
                "start_date": start_date,
                "end_date": end_date,
                "resource_id": user_id,
                "resource_price": parsed_data["resources"][user_id]["price"],
                "duration": duration,
            }
        )
        parsed_data["tasks"][task_id] = task
        res.append(task)

    min_date = min(map(lambda x: x["start_date"], res))
    max_date = max(map(lambda x: x["end_date"], res))

    return res
