import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the Food-101 dataset
(ds, ds_info) = tfds.load('food101', split='train', as_supervised=True, with_info=True)

# Get the class names
class_names = ds_info.features['label'].names

# Display the first 10 samples
plt.figure(figsize=(15, 6))
for i, (image, label) in enumerate(ds.take(10)):
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.title(class_names[label.numpy()])
    plt.axis('off')
plt.tight_layout()
plt.show()
