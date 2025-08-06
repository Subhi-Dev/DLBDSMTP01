import requests
import random
import time
url = 'http://localhost:5000/predict'
while(True):
    myobj = {
          "temperature": random.randrange(0, 100),
          "humidity": random.randrange(1, 100),
          "sound_volume": random.randrange(20, 120)
        }
    
    x = requests.post(url, json = myobj)
    print(x.text)
    time.sleep(1)