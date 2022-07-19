import tensornetwork as tn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
num_pixels = x_train.shape[1] * x_train.shape[2] # find size of one-dimensional vector

x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32') # flatten training images
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32') # flatten test images
x_train = x_train[:,:625]
x_test = x_test[:,:625]
num_classes=10
y_train = tf.keras.utils.to_categorical(y_train,num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

mpo_model = Sequential()
mpo_model.add(Dense(256, use_bias=True, activation='relu', input_shape=(625,)))
mpo_model.add(Dense(10, use_bias=True, activation='softmax'))
mpo_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
mpo_model.summary()
batch_size = 128
num_classes = 10
epochs = 20

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

history = mpo_model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=my_callbacks)
