from collections import defaultdict
from datetime import datetime, timedelta
import json
from copy import deepcopy

from .parser import get_data


raw_data = dict()
data = dict()
data_lists = dict()
default_calendar = defaultdict(lambda: 8)
calendar = defaultdict(lambda: default_calendar)


def init(init_data: dict):
    raw_data.clear()
    data.clear()
    data_lists.clear()
    calendar.clear()
    raw_data.update(init_data)
    data.update(get_data(init_data))
    data_lists.update(get_data(init_data, is_dict=False))

    week_days = []
    for calendar_row in raw_data["calendars"]["rows"]:
        if calendar_row["id"] == "general":
            week_days = calendar_row["intervals"]

    for day in week_days:
        if day["startDate"] is None:
            continue
        date = datetime.strptime(day["startDate"], "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d")
        if default_calendar[date] in [0, 8]:
            if day["isWorking"]:
                default_calendar[date] = 7
            else:
                default_calendar[date] = 0


def get_output(tasks: list[dict], dep=None):
    tasks = {task["id"]: task for task in tasks}

    def change_tasks(tasks_list: list) -> (datetime, datetime):
        start_date, end_date = None, None
        for task in tasks_list:
            if 'children' in task:
                cur_start, cur_end = change_tasks(task["children"])
                task["startDate"] = cur_start.strftime("%Y-%m-%dT%H:%M:%S")
                task["endDate"] = cur_end.strftime("%Y-%m-%dT%H:%M:%S")
                duration = 0

                if start_date is None:
                    start_date = cur_start
                if end_date is None:
                    end_date = cur_end
                start_date = min(start_date, cur_start)
                end_date = max(end_date, cur_end)
                while cur_start < cur_end:
                    if default_calendar[cur_start.strftime("%Y-%m-%d")] > 0 and cur_start.weekday() < 5:
                        duration += 1
                    cur_start += timedelta(days=1)
                task["duration"] = duration
            else:
                new_task = tasks[task["id"]]
                if start_date is None:
                    start_date = new_task["start_date"]
                if end_date is None:
                    end_date = new_task["end_date"]
                start_date = min(start_date, new_task["start_date"])
                end_date = max(end_date, new_task["end_date"])
                task.update(
                    {
                        "startDate": new_task["start_date"].strftime("%Y-%m-%dT%H:%M:%S"),
                        "endDate": new_task["end_date"].strftime("%Y-%m-%dT%H:%M:%S"),
                        "duration": new_task["duration"],
                    }
                )

        return start_date, end_date

    result = deepcopy(raw_data)

    change_tasks(result["tasks"]["rows"])

    assignments = []

    for task in tasks.values():
        assignments.append(
            {
                "event": task["id"],
                "resource": task["resource_id"],
                "units": 100,
                "startDate": task["start_date"].strftime("%Y-%m-%dT%H:%M:%S"),
                "endDate": task["end_date"].strftime("%Y-%m-%dT%H:%M:%S"),
                "currentEffort": 144000000,
                "guid": "d6f73a0f-4d88-45cf-9d14-e0878bd0f466",
                "id": "7335019676944236545"
            }
        )

    result["assignments"]["rows"] = assignments

    # res_dep = []
    # for f, t in dep.items():
    #     for c in t:
    #         res_dep.append(
    #             {
    #                 "from": f,
    #                 "fromId": f,
    #                 "to": c,
    #                 "toId": c,
    #             }
    #         )
    #
    # result["dependencies"]["rows"] = res_dep

    return result
