
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# 1 DATA LOADING & PREPROCESSING


def load_data():
    """Load MNIST, normalise, reshape, and one-hot-encode labels."""
    print("\nLoading MNIST dataset ...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalise 
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32')  / 255.0

    X_train = X_train[..., np.newaxis]
    X_test  = X_test[...,  np.newaxis]

    y_train_ohe = to_categorical(y_train, 10)
    y_test_ohe  = to_categorical(y_test,  10)

    print(f"Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")
    return X_train, y_train, y_train_ohe, X_test, y_test, y_test_ohe


# 2 MODEL ARCHITECTURE


def build_cnn():
    """Build and compile the CNN model."""
    model = models.Sequential([

        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

  
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ], name="MNIST_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# 3 DATA AUGMENTATION


def make_augmentor():
    """Mild augmentation for training: slight shifts and rotation."""
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )


# 4 TRAINING


def train(model, X_train, y_train_ohe, X_test, y_test_ohe,
          epochs=20, batch_size=128, save_path="results"):

    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, "mnist_cnn.keras")

    cb = [
        callbacks.ModelCheckpoint(model_path, save_best_only=True,
                                  monitor='val_accuracy', verbose=0),
        callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=3, min_lr=1e-6, verbose=0)
    ]

    augmentor = make_augmentor()
    aug_gen   = augmentor.flow(X_train, y_train_ohe, batch_size=batch_size)
    steps     = len(X_train) // batch_size

    print(f"\nTraining CNN  (epochs={epochs}, batch_size={batch_size}) ...")
    history = model.fit(
        aug_gen,
        steps_per_epoch=steps,
        epochs=epochs,
        validation_data=(X_test, y_test_ohe),
        callbacks=cb,
        verbose=1
    )

    best_model = tf.keras.models.load_model(model_path)
    print(f"\nBest model saved -> {model_path}")
    return best_model, history


# 5 EVALUATION


def evaluate(model, X_test, y_test):
    """Print full classification report."""
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred,
                                 target_names=[str(i) for i in range(10)]))

    acc = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")
    return y_pred


# 6 VISUALISATIONS


def plot_training(history, save_path="results"):
    """Plot training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('MNIST CNN - Training History', fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(history.history['accuracy'],     label='Train Accuracy', linewidth=2)
    ax.plot(history.history['val_accuracy'], label='Val Accuracy',   linewidth=2)
    ax.set_title('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history.history['loss'],     label='Train Loss', linewidth=2)
    ax.plot(history.history['val_loss'], label='Val Loss',   linewidth=2)
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_path, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved -> {path}")


def plot_confusion_matrix(y_test, y_pred, save_path="results"):
    """Heatmap of the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10), ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    path = os.path.join(save_path, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved -> {path}")


def plot_sample_predictions(model, X_test, y_test, save_path="results", n=36):
    """Grid of sample predictions (green = correct, red = wrong)."""
    indices   = np.random.choice(len(X_test), n, replace=False)
    imgs      = X_test[indices]
    true_lbls = y_test[indices]
    pred_lbls = np.argmax(model.predict(imgs, verbose=0), axis=1)

    cols = 6
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2))
    fig.suptitle('Sample Predictions (green = correct, red = wrong)', fontsize=13, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i, :, :, 0], cmap='gray')
        color  = 'green' if pred_lbls[i] == true_lbls[i] else 'red'
        symbol = 'correct' if pred_lbls[i] == true_lbls[i] else 'wrong'
        ax.set_title(f"P:{pred_lbls[i]} T:{true_lbls[i]}",
                     color=color, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(save_path, "sample_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions saved -> {path}")


def plot_feature_maps(model, X_test, save_path="results"):
    """Visualise Conv layer 1 feature maps for one sample image."""
    sample = X_test[0:1]

    conv1_model = tf.keras.Model(inputs=model.inputs,
                                 outputs=model.layers[0].output)
    feat_maps   = conv1_model.predict(sample, verbose=0)[0]  # (28, 28, 32)

    n_maps = min(16, feat_maps.shape[-1])
    fig, axes = plt.subplots(4, n_maps // 4, figsize=(12, 6))
    fig.suptitle('Conv Layer 1 - Feature Maps', fontsize=13, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        ax.imshow(feat_maps[:, :, i], cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(save_path, "feature_maps.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature maps saved -> {path}")


def plot_misclassified(model, X_test, y_test, save_path="results", n=20):
    """Show examples the model got wrong."""
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    wrong  = np.where(y_pred != y_test)[0][:n]

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.2))
    fig.suptitle('Misclassified Samples', fontsize=13, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i < len(wrong):
            idx = wrong[i]
            ax.imshow(X_test[idx, :, :, 0], cmap='gray')
            ax.set_title(f"True:{y_test[idx]}  Pred:{y_pred[idx]}", color='red', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(save_path, "misclassified.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Misclassified examples saved -> {path}")



# 7 MAIN


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    SAVE_PATH = "results"

    X_train, y_train, y_train_ohe, X_test, y_test, y_test_ohe = load_data()

    model = build_cnn()
    print("\nModel Summary:")
    model.summary()

    best_model, history = train(
        model, X_train, y_train_ohe, X_test, y_test_ohe,
        epochs=20, batch_size=128, save_path=SAVE_PATH
    )

    y_pred = evaluate(best_model, X_test, y_test)

    plot_training(history, SAVE_PATH)
    plot_confusion_matrix(y_test, y_pred, SAVE_PATH)
    plot_sample_predictions(best_model, X_test, y_test, SAVE_PATH)
    plot_feature_maps(best_model, X_test, SAVE_PATH)
    plot_misclassified(best_model, X_test, y_test, SAVE_PATH)

    print(f"\nAll done. Results are in '{SAVE_PATH}/'")


if __name__ == "__main__":
    main()
