import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., None]/255.0
x_test  = x_test[..., None]/255.0


model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

ckpt = callbacks.ModelCheckpoint('digit_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
model.fit(x_train, y_train, epochs=6, batch_size=128, validation_data=(x_test, y_test), callbacks=[ckpt])

print("✅ Saved: digit_model.h5")
