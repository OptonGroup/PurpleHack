from collections import defaultdict
from datetime import datetime, timedelta
import json
from copy import deepcopy

from xml.etree import ElementTree as ET

from backend.algorithms.parser import get_data


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
    

def init_xml(xml_file: str):
    tree = ET.parse(xml_file)

    flag = False
    week_days = []
    for elem in tree.iter():
        if elem.tag.endswith("}Calendar"):
            for attr in list(elem):
                if "Name" in attr.tag:
                    if attr.text == "General":
                        flag = True
                        break
            
            if flag:
                for attr in list(elem):
                    if attr.tag.endswith("Exceptions"):
                        for exception in list(attr):
                            week_day = {
                                "recurrentStartDate": None,
                                "recurrentEndDate": None,
                                "isWorking": False,
                                "name": None,
                                # "startDate": "2021-01-01T00:00:00",
                                # "endDate": "2021-01-02T00:00:00",
                                "priority": 1000,
                                "recurrentWeekday": None,
                                "workException": "WEEKEND",
                                "overriddenWorkDayTimeInterval": None
                            }

                            for attttttr in list(exception):
                                if "TimePeriod" in attttttr.tag:
                                    for meow in list(attttttr):
                                        if "ToDate" in meow.tag:
                                            week_day["startDate"] = meow.text
                                        if "FromDate" in meow.tag:
                                            week_day["endDate"] = meow.text
                            week_days.append(week_day)
    
    print(week_days)


if __name__ == "__main__":
    init_xml("исходные данные.xml")
