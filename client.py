import requests
import random
import time
url = 'http://localhost:5000/predict'
while(True):
    myobj = {
          "precipitation": random.randrange(0, 55),
          "temp_max": random.randrange(-1, 35),
          "temp_min": random.randrange(-7, 18),
          "wind": random.randrange(1, 9)
        }
    
    x = requests.post(url, json = myobj)
    print(x.text)
    time.sleep(1)