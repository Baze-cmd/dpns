from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import kagglehub
import os
import cv2
import numpy as np


class SuperResolutionConvolutionalNetwork:
    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor
        self.model = None
        self.base_path = None
        self.training_high_res = None
        self.training_low_res = None

    def build(self):
        self.model = models.Sequential([
            layers.Input(shape=(None, None, 3)),
            layers.Conv2D(64, kernel_size=9, padding='same', activation='relu'),
            layers.Conv2D(32, kernel_size=1, padding='same', activation='relu'),
            layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
            layers.UpSampling2D(size=self.scale_factor),
            layers.Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')
        ])
        return self

    def download_dataset(self):
        print("Downloading dataset...")
        self.base_path = kagglehub.dataset_download("adityachandrasekhar/image-super-resolution")
        print("Path to dataset files:", self.base_path)
        return self.base_path

    def load_dataset(self):
        self.training_high_res = self.load_images_from(os.path.join(self.base_path, "dataset", "train", "high_res"))
        print("Number of training images:", len(self.training_high_res))
        downsampled_images = []
        for img in self.training_high_res:
            height, width, _ = img.shape
            new_dim = (int(width / self.scale_factor), int(height / self.scale_factor))
            downsampled_img = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
            downsampled_images.append(downsampled_img)
        self.training_low_res = downsampled_images
        self.training_high_res = np.array(self.training_high_res, dtype=np.float32)
        self.training_low_res = np.array(self.training_low_res, dtype=np.float32)

        #eval_high_res = self.load_images_from(os.path.join(self.base_path, "dataset", "val", "high_res"))
        #eval_low_res = self.load_images_from(os.path.join(self.base_path, "dataset", "val", "low_res"))
        #print("Number of evaluate high resolution images:", len(eval_high_res))
        #print("Number of evaluate low resolution images:", len(eval_low_res))

    def load_images_from(self, path):
        images = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        for filename in os.listdir(path):
            if not filename.lower().endswith(valid_extensions):
                continue
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                images.append(img)

        if len(images) == 0:
            raise ValueError(f"No valid images found in {path}")

        return images

    def train(self, epochs=100):
        if len(self.training_low_res.shape) != 4 or len(self.training_high_res.shape) != 4:
            raise ValueError(f"Incorrect input shape: {self.training_low_res.shape}, {self.training_high_res.shape}")

        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        history = self.model.fit(
            self.training_low_res,
            self.training_high_res,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )

        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final training MAE: {history.history['mae'][-1]:.4f}")

        return history

    def enhance_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32) / 255.0

        enhanced_image = self.model.predict(image)

        enhanced_image = enhanced_image.squeeze()
        enhanced_image = (enhanced_image * 255.0).astype(np.uint8)
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
        return enhanced_image

    def save_model(self, file_name="super_resolution_model.h5"):
        cwd = os.getcwd()
        path = os.path.join(cwd, file_name)
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, file_name="super_resolution_model.h5"):
        cwd = os.getcwd()
        path = os.path.join(cwd, file_name)
        if not os.path.exists(path):
            print(f"No saved model found at {path}")
            return None
        model = load_model(path)
        print(f"Model loaded from {path}")
        return model
