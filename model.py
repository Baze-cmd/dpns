from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import kagglehub
import os
from tensorflow import keras

os.environ['KAGGLEHUB_CACHE'] = os.getcwd()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


def load_images_from(path, limit=None):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = []

    for filename in os.listdir(path):
        if not filename.lower().endswith(valid_extensions):
            continue
        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue
        if limit:
            return img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = cv2.resize(img, (256, 256))
        images.append(img)

    if len(images) == 0:
        raise ValueError(f"No valid images found in {path}")

    return np.array(images)


def save_training_history(history, name):
    loss = history.history['loss']
    val_loss = history.history.get('val_loss')
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, label='Training Loss', color='blue')

    if val_loss is not None:
        plt.plot(epochs, val_loss, label='Validation Loss', color='orange')

    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    save_dir = os.path.join(os.getcwd(), "history")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{name}.png'))

    plt.close()

    history_data = {"loss": loss, "val_loss": val_loss} if val_loss is not None else {"loss": loss}
    json_path = os.path.join(save_dir, f'{name}.json')

    with open(json_path, 'w') as f:
        json.dump(history_data, f)


class SuperResolutionConvolutionalNeuralNetwork:
    def __init__(self, name="SRCNN", epochs=64, batch_size=32, learning_rate=0.0001):
        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.name += f"-epochs{self.epochs}-bs{self.batch_size}-lr{self.learning_rate}.keras"
        self.model = None
        self.base_path = None
        self.high_res = None
        self.low_res = None

    def build(self):
        self.model = models.Sequential(
            name=self.name,
            layers=[
                layers.Input(shape=(None, None, 3)),
                layers.Conv2D(64, kernel_size=9, padding='same', activation='relu'),
                layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
                layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
                layers.Conv2D(32, kernel_size=1, padding='same', activation='relu'),
                layers.Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')
            ]
        )

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae'])

        return self

    def download_dataset(self, kagglehub_dataset="adityachandrasekhar/image-super-resolution/versions/2"):
        print("Downloading dataset...")
        self.base_path = kagglehub.dataset_download(kagglehub_dataset)
        print("Path to dataset files:", self.base_path)

        self.high_res = load_images_from(os.path.join(self.base_path, "dataset", "Raw Data", "high_res"))
        self.low_res = load_images_from(os.path.join(self.base_path, "dataset", "Raw Data", "low_res"))

        print("Number of images:", len(self.high_res))

        if len(self.high_res) != len(self.low_res):
            raise ValueError(f"Number of images does not match")

        return self.base_path

    def train(self):
        history = self.model.fit(
            self.low_res,
            self.high_res,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=1
        )
        print(history.history)
        save_training_history(history, self.name)

        return history

    def enhance_image(self, image, scale_factor):
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
        scaled_image = np.expand_dims(scaled_image, axis=0)
        scaled_image = scaled_image.astype(np.float32) / 255.0

        enhanced_image = self.model.predict(scaled_image)

        enhanced_image = enhanced_image.squeeze()
        enhanced_image = (enhanced_image * 255.0).astype(np.uint8)
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
        return enhanced_image

    def save_model(self):
        model_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, self.name)

        self.model.save(path)
        print(f"Model saved to {path}")

    def load_models(self):
        model_dir = os.path.join(os.getcwd(), "models")

        keras_models = [f for f in os.listdir(model_dir) if f.endswith('.keras')]

        if not keras_models:
            return None

        for i, model_file in enumerate(keras_models, 1):
            print(f"{i}. {model_file}")

        while True:
            try:
                selection = int(input("Enter the number of the model you want to load(0 to build a new model): "))
                if selection == 0:
                    return None

                if 1 <= selection < len(keras_models) + 1:
                    file_name = keras_models[selection - 1]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

        path = os.path.join(model_dir, file_name)
        print(f"Loading model from {path}")
        self.model = load_model(path, compile=False)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse',
                           metrics=['mae'])
        print(f"Model loaded from {path}")
        return self
