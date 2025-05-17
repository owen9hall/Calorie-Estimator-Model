import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model

transfer_learning = True

(splits, info) = tfds.load(
   'food101',
   split=['train[:80%]', 'train[80%:90%]', 'train[90%:100%]'],
   as_supervised=True,
   with_info=True
)
train_ds, val_ds, test_ds = splits

# Preprocess
def preprocess(image, label):
   image = tf.image.resize(image, [224, 224]) # Resize to compatible size
   image = tf.cast(image, tf.float32) / 255.0 # Normalize pixel values
   return image, label

# Apply preprocessing to each (image, label) pair
# Group data into batches of size 32
# Pre-fetch next batch during training
batch_size = 32
train_ds = train_ds.map(preprocess).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# load the specified model
model = load_model('food101_cnn_pretrained.keras') if not transfer_learning else load_model('food101_inceptionv3.keras')
model.compile(
   optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
   loss='sparse_categorical_crossentropy',
   metrics=['accuracy']
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=4,
                                                  restore_best_weights=True,
                                                  verbose=1)

model.fit(
   train_ds,
   validation_data=val_ds,
   epochs=100,
   callbacks=[early_stopping],
   verbose=2
)
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2%}")