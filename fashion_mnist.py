import tensorflow as tf
import tensorflow.python.keras as keras
fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()