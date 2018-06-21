import tensorflow as tf
import numpy as np
import pylab as pl
import pandas as pd
from tensorflow.contrib.factorization.python.ops import clustering_ops
from mpl_toolkits.mplot3d import Axes3D
classes = 3
#read data
dt = pd.read_csv('DataFlow/output/caten', header=None)
fl = []
for index, row in dt.iterrows():
    z = []
    z. append(int(row[0][1:]))
    z. append(row[1])
    z. append(int(row[2][1:-1]))
    fl.append(z)
ip_data = fl
num_elems = len(ip_data)
num_recs = len(ip_data[0])

def train_data():
    ip_dt = tf.constant(ip_data, tf.float32)
    return (ip_dt, None)
def pred_fn():
    return np.array(ip_data, np.float32)

model = tf.contrib.learn.KMeansClustering(classes
                                          , distance_metric = clustering_ops.SQUARED_EUCLIDEAN_DISTANCE
                                          , initial_clusters=tf.contrib.learn.KMeansClustering.RANDOM_INIT
                                         )
model.fit(input_fn=train_data, steps=5000)
preds = model.predict(input_fn=pred_fn, as_iterable=True)

colors = ['yellow', 'blue', 'green']
f = pl.figure()
ax = f.add_subplot(111, projection='3d')
idx = 0
for i in preds:
    #print(ip_data[idx]," --> cluster_",i['cluster_idx'])
    ax.scatter(ip_data[idx][0], ip_data[idx][1], ip_data[idx][2], c=colors[i['cluster_idx']])
    idx  = idx + 1
pl.show()
