import xmltodict
import json
from pprint import pprint

# Sample XML data
xml_data = open("../../исходные данные.xml").read()

# Convert XML to Python dictionary
python_dict = xmltodict.parse(xml_data)

# Convert Python dictionary to JSON
json_data = json.dumps(python_dict, indent=4)

# Print JSON data
print(json_data)
