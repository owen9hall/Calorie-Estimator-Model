import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot as plt

transfer_learning = False

(splits, info) = tfds.load(
   'food101',
   split=['train[:80%]', 'train[80%:90%]', 'train[90%:100%]'],
   as_supervised=True,
   with_info=True
)
train_ds1, val_ds1, test_ds1 = splits
train_ds2, val_ds2, test_ds2 = splits

# Preprocess
def preprocess(image, label):
   image = tf.image.resize(image, [224, 224]) # Resize to compatible size
   image = tf.cast(image, tf.float32) / 255.0 if not transfer_learning else preprocess_input(image) # Normalize pixel values
   return image, label

# Apply preprocessing to each (image, label) pair
# Group data into batches
# Pre-fetch next batch during training
batch_size = 32
train_ds1 = train_ds1.map(preprocess).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds1 = val_ds1.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds1 = test_ds1.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

transfer_learning = True
train_ds2 = train_ds2.map(preprocess).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds2 = val_ds2.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds2 = test_ds2.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=4,
                                                  restore_best_weights=True,
                                                  verbose=1)



# Train both models and plot their loss curves together

# Load both models
cnn_model = load_model('food101_cnn_pretrained.keras')
cnn_model.compile(
   optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
   loss='sparse_categorical_crossentropy',
   metrics=['accuracy']
)

inception_model = load_model('food101_inceptionv3.keras')
inception_model.compile(
   optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
   loss='sparse_categorical_crossentropy',
   metrics=['accuracy']
)

# Train both models
cnn_history = cnn_model.fit(
   train_ds1,
   validation_data=val_ds1,
   epochs=100,
   callbacks=[early_stopping],
   verbose=2
)

inception_history = inception_model.fit(
   train_ds2,
   validation_data=val_ds2,
   epochs=100,
   callbacks=[early_stopping],
   verbose=2
)

# Plot training & validation loss values for both models
plt.figure(figsize=(10, 6))
plt.plot(cnn_history.history['loss'], label='CNN Training Loss')
plt.plot(cnn_history.history['val_loss'], label='CNN Validation Loss')
plt.plot(inception_history.history['loss'], label='InceptionV3 Training Loss')
plt.plot(inception_history.history['val_loss'], label='InceptionV3 Validation Loss')
plt.title('Model Loss per Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluate both models
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(test_ds1)
inception_test_loss, inception_test_accuracy = inception_model.evaluate(test_ds2)
print(f"\nCNN Test Loss: {cnn_test_loss:.4f}")
print(f"CNN Test Accuracy: {cnn_test_accuracy:.2%}")
print(f"\nInceptionV3 Test Loss: {inception_test_loss:.4f}")
print(f"InceptionV3 Test Accuracy: {inception_test_accuracy:.2%}")