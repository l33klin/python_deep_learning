from tensorflow import keras

# Import data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Create model
model = keras.Sequential([
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Compile
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Pre-process data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# Train
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Prediction with first 10 data
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print("predictions[0]: {}".format(predictions[0].argmax()))
print("test_labels[0]: {}".format(test_labels[0]))

# prediction all test data and print accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")
