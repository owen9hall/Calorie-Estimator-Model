import tensorflow as tf
import tensorflow_datasets as tfds

dataset, info = tfds.load(
   'food101',
   split='train+validation',
   as_supervised=True,
   with_info=True
)

# Preprocess: resize and normalize
def preprocess(image, label):
    image = tf.image.resize(image, [224, 224]) # Resize to compatible size
    image = tf.cast(image, tf.float32) / 255.0 # Normalize pixel values
    return image, label

# Apply preprocessing to each (image, label) pair
dataset = dataset.map(preprocess)
dataset = dataset.shuffle(buffer_size=10000)

dataset = dataset.batch(32) # Group data into batches of size 32
dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Pre-fetch next batch during training
