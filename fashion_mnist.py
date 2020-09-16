import tensorflow as tf
from tensorflow import keras
fashion_mnist=keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels)=fashion_mnist.load_data()
training_images  = training_images / 255.0
test_images = test_images / 255.0
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)