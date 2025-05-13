import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

dataset, info = tfds.load(
   'food101',
   split='train+validation',
   as_supervised=True,
   with_info=True
)

# Preprocess
def preprocess(image, label):
    image = tf.image.resize(image, [224, 224]) # Resize to compatible size
    image = tf.cast(image, tf.float32) / 255.0 # Normalize pixel values
    return image, label

# Apply preprocessing to each (image, label) pair
dataset = dataset.map(preprocess)
dataset = dataset.shuffle(buffer_size=10000)

# for i, (image, label) in enumerate(dataset.take(10)):
#     plt.figure()
#     plt.imshow(image)
#     plt.title(f"Label: {info.features['label'].int2str(label)}")
#     plt.axis('off')
#     plt.show()

dataset = dataset.batch(32) # Group data into batches of size 32
dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Pre-fetch next batch during training

model = models.Sequential()

# add convolutional layer
model.add(layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(224, 224, 3)
))

# add pooling layer
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# add another conv layer
model.add(layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation='relu'
))

# add pooling layer
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# add conv layer
# add convolutional layer
model.add(layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation='relu',
))

# flatten as input to ANN
model.add(layers.Flatten())

# hidden layer
model.add(layers.Dense(units=128, activation='relu'))

# output layer
model.add(layers.Dense(101, activation='softmax'))

model.summary()