# -*- coding: utf-8 -*-
"""
Description: Check if tensorflow works with a CPU or GPU.
"""
import tensorflow as tf
from datetime import datetime
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--device', default="cpu",
                    help='choose device: cpu or gpu')

parser.add_argument('-s', '--shape', default=1000, type=int,
                    help='choose number of rows/columns of quadratic matrix')

args = parser.parse_args()
device_name = args.device
shape = (args.shape,args.shape)

if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)

startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print('result:', result)

print("matrix shape:", shape, "device:", device_name)
print("running time:", datetime.now() - startTime)
