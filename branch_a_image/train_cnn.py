"""
train_cnn.py

Branch A: Genuine vs Tampered QR Image Classifier
MobileNetV2 transfer learning with stratified validation split
and evaluation metrics.

Run from project root:
    .\venv\Scripts\python.exe branch_a_image\train_cnn.py
"""

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==============================
# CONFIGURATION
# ==============================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS_HEAD = 8
EPOCHS_FINE_TUNE = 6

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "images"
)

MODEL_OUT = os.path.join(
    os.path.dirname(__file__),
    "qr_cnn_model.keras"
)


# ==============================
# BUILD DATASETS
# ==============================

def build_datasets():

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}"
        )

    class_names = ["genuine", "tampered"]

    file_paths = []
    labels = []

    # Collect images from both folders
    for label, class_name in enumerate(class_names):

        class_dir = os.path.join(DATA_DIR, class_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

        for filename in sorted(os.listdir(class_dir)):

            if filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp")
            ):

                file_paths.append(
                    os.path.join(class_dir, filename)
                )

                labels.append(label)

    # Check dataset
    if not file_paths:
        raise ValueError("No images found in dataset.")

    if len(set(labels)) != 2:
        raise ValueError(
            "Both genuine and tampered images are required."
        )

    print("\n========== DATASET INFORMATION ==========")

    print("Total images:", len(file_paths))
    print("Genuine images:", labels.count(0))
    print("Tampered images:", labels.count(1))

    # Stratified train-validation split
    train_paths, val_paths, train_labels, val_labels = (
        train_test_split(
            file_paths,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
    )

    print("\n========== DATASET SPLIT ==========")

    print("Training Genuine:", train_labels.count(0))
    print("Training Tampered:", train_labels.count(1))

    print("Validation Genuine:", val_labels.count(0))
    print("Validation Tampered:", val_labels.count(1))

    # Image loading and preprocessing
    def load_image(path, label):

        image = tf.io.read_file(path)

        image = tf.io.decode_image(
            image,
            channels=3,
            expand_animations=False
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(
            image,
            IMG_SIZE
        )

        image = tf.cast(image, tf.float32)

        # MobileNetV2 expects values between -1 and 1
        image = image / 127.5 - 1.0

        return image, tf.cast(label, tf.float32)

    # Training dataset
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_paths, train_labels)
    )

    train_ds = train_ds.shuffle(
        buffer_size=len(train_paths),
        seed=42
    )

    train_ds = train_ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # Validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_paths, val_labels)
    )

    val_ds = val_ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    print("\nClass mapping:")
    print("0 = Genuine")
    print("1 = Tampered")

    print("========================================")

    return train_ds, val_ds, class_names


# ==============================
# BUILD MOBILENETV2 MODEL
# ==============================

def build_model():

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers initially
    base.trainable = False

    inputs = tf.keras.Input(
        shape=IMG_SIZE + (3,)
    )

    x = base(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = models.Model(
        inputs,
        outputs
    )

    return model, base


# ==============================
# VALIDATION METRICS
# ==============================

def print_validation_metrics(model, val_ds):

    print("\n========== VALIDATION RESULTS ==========")

    y_true = []
    y_prob = []

    for images, labels in val_ds:

        predictions = model.predict(
            images,
            verbose=0
        ).flatten()

        y_true.extend(
            labels.numpy().flatten()
        )

        y_prob.extend(predictions)

    y_true = np.array(y_true).astype(int)

    y_prob = np.array(y_prob)

    # Threshold for binary classification
    y_pred = (y_prob >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    print(
        f"\nAccuracy  : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision : {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall    : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score  : {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Genuine", "Tampered"],
            zero_division=0
        )
    )

    print("========================================")

    print(
        "These are validation results, "
        "not independent test results."
    )


# ==============================
# MAIN TRAINING FUNCTION
# ==============================

def main():

    print("\nLoading QR image dataset...")

    train_ds, val_ds, class_names = build_datasets()

    print("\nBuilding MobileNetV2 model...")

    model, base = build_model()

    # Phase 1: Train classification head
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    print(
        "\n--- Phase 1: Training classification head ---"
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD
    )

    # Phase 2: Fine-tune top MobileNetV2 layers
    print(
        "\n--- Phase 2: Fine-tuning MobileNetV2 ---"
    )

    base.trainable = True

    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNE
    )

    # Final validation evaluation
    print_validation_metrics(
        model,
        val_ds
    )

    # Save trained model
    model.save(MODEL_OUT)

    print("\nCNN model saved successfully:")
    print(MODEL_OUT)

    print("\nClass mapping:")
    print("0 =", class_names[0])
    print("1 =", class_names[1])


if __name__ == "__main__":
    main()