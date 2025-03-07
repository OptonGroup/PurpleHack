from backend.app.utils.calculate import process_json
import json

duration = 1
price = 0
resource = 0

with open("тестовое задание.json") as input_file:
    res = process_json(json.load(input_file), duration, price, resource)

    with open("result.json", "w", encoding="utf-8") as output_file:
        json.dump(res, output_file)
