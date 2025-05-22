from tensorflow.keras import layers, models

model = models.Sequential()

# add convolutional layer
model.add(layers.Conv2D(
    filters=16,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(224, 224, 3)
))
# model.add(layers.LeakyReLU())

# add pooling layer
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# add another conv layer
model.add(layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
))
# model.add(layers.LeakyReLU())

# add pooling layer
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# add convolutional layer
model.add(layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
))
# model.add(layers.LeakyReLU())

# flatten as input to ANN
model.add(layers.Flatten())

# dropout to help prevent overfitting
model.add(layers.Dropout(0.5))

# hidden layer
model.add(layers.Dense(units=64, activation='relu'))
# model.add(layers.LeakyReLU())

# output layer
model.add(layers.Dense(101, activation='softmax'))

model.summary()
model.save('food101_cnn_pretrained.keras')