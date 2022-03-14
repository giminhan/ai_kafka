from kafka import KafkaProducer
from json import dumps

import time
import numpy as np 
# Create your views here.

start_time = time.time()
def product():
    print("시작합니다")
    prod = KafkaProducer(acks=0, compression_type="gzip", api_version=(0, 11, 5),
                        bootstrap_servers=["localhost:9092"],
                        value_serializer = lambda x: dumps(x).encode("utf-8"))
    
    range_num = 1000
    rand_nums = np.random.rand(range_num)
    for i in range(10):
        data = {"str": "result"+str(rand_nums[i])}
        print(data)
        prod.send("data_delivery", value=data)
        prod.flush()
    print(f"data elapsed: {time.time() - start_time}")



product()