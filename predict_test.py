import requests
import json

url = 'http://localhost:9696/predict'

material = {
    "c": 0.29,
    "si": 0.26,
    "mn": 0.77,
    "p": 0.009,
    "s": 0.007,
    "ni": 0.046,
    "cr": 1.12,
    "mo": 1.2,
    "cu": 0.08,
    "v": 0.23,
    "al": 0.004,
    "n": 0.0095,
    "temperature_celcius": 27
    }

response = requests.post(url, json=material).json()

print(json.dumps(response, indent=1))
