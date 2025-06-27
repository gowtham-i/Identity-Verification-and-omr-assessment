import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pathlib
import os

# Create necessary directories
os.makedirs('reports', exist_ok=True)
os.makedirs('reports/visualizations', exist_ok=True)

data_dir = './dataset/train'
batch_size = 32
img_height = 224  
img_width = 224   

data_dir_path = pathlib.Path(data_dir)
class_names = sorted([item.name for item in data_dir_path.glob('*') if item.is_dir()])

print("Extracted class names:", class_names)
print("Total classes found:", len(class_names))

original_counts = {class_name: len(list((data_dir_path/class_name).glob('*'))) 
                  for class_name in class_names}
pd.DataFrame(original_counts, index=['count']).T.to_csv('reports/original_class_distribution.csv')

# Load dataset without validation
train_raw = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=None
)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

def preprocess(image, label, training=True):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0  # Normalize to [-1,1]
    if training:
        image = data_augmentation(image)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_raw.map(
    lambda x, y: preprocess(x, y, training=True),
    num_parallel_calls=AUTOTUNE
).shuffle(1000).batch(batch_size).cache().prefetch(AUTOTUNE)

# Display sample images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    denormalized = (images + 1.0) * 127.5
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(denormalized[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.savefig('reports/visualizations/augmented_sample_images.png')
plt.close()

# Define Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

tf.keras.utils.plot_model(
    model,
    to_file='reports/visualizations/model_architecture.png',
    show_shapes=True,
    show_layer_names=True
)
with open('reports/model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    save_best_only=True,
    monitor='accuracy',
    mode='max'
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=3,
    restore_best_weights=True
)

# Train model
history = model.fit(
    train_ds,
    epochs=15,
    callbacks=[checkpoint, early_stopping]
)

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('reports/training_history.csv', index=False)

# Generate evaluation metrics
y_true = []
y_pred_probs = []

for images, labels in train_ds.unbatch():
    y_true.append(labels.numpy())
    y_pred_probs.append(model.predict(tf.expand_dims(images, axis=0), verbose=0)[0])

y_pred = np.argmax(y_pred_probs, axis=1)

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
pd.DataFrame(report).transpose().to_csv('reports/classification_report.csv')

cm = confusion_matrix(y_true, y_pred)
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv('reports/confusion_matrix.csv')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('reports/visualizations/training_curves.png')
plt.close()

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('models/identity_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Training complete. Model saved.")