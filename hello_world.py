import tensorflow as tf
from tensorflow import keras
import numpy as np
xs = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],dtype=float)
ys = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5],dtype=float)
model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
model.fit(xs,ys,epochs=500)
print(model.predict([10.0]))