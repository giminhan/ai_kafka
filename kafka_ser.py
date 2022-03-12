from kafka import KafkaProducer
from kafka import KafkaConsumer
from json import dumps, load

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
        prod.send("test", value=data)
        prod.flush()
    print(f"data elapsed: {time.time() - start_time}")

"""
def con():
    consumer = KafkaConsumer("test",
                             bootstrap_servers=["localhost:9092"],
                             auto_offset_reset="earliest",
                             enable_auto_commit=True,
                             group_id="my-group",
                             value_deserializer=lambda x: load(x.decode("utf-8")),
                             consumer_timeout_ms=1000)
    print("get conumser lst")
    for message in consumer:
        print("Topic: %s, Partition: %d, Offset: %d, Key: %s, Value: %s" 
              % ( message.topic, message.partition, message.offset, message.key, message.value ))
    
    print("end")
"""

product()