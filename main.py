import os
import cv2
from datetime import datetime
from model import SuperResolutionConvolutionalNeuralNetwork, load_images_from


def load_image():
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    files = os.listdir(os.getcwd())
    images = []
    for file in files:
        if file.lower().endswith(extensions):
            images.append(file)
    return cv2.imread(images[0])


def SRCNN(image, scale_factor):
    SRCNN = SuperResolutionConvolutionalNeuralNetwork().load_models()
    if SRCNN is None:
        print('No model found.')
        print('Creating new model and training.')
        SRCNN = SuperResolutionConvolutionalNeuralNetwork().build()
        SRCNN.download_dataset(kagglehub_dataset="adityachandrasekhar/image-super-resolution/versions/2")
        SRCNN.train()
        SRCNN.save_model()
    print(SRCNN.model.summary())
    scaled_image = SRCNN.enhance_image(image, scale_factor=scale_factor)
    return scaled_image


def save_image(image, algorithm):
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    name = f'{algorithm}:{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.jpg'
    output_path = os.path.join(output_dir, name)
    if cv2.imwrite(output_path, image):
        print(f"Image saved at: {output_path}")
    else:
        print("Failed to save the image.")


def main():
    image = load_images_from(path=os.getcwd(), limit=1)
    print('1. Nearest neighbor')
    print('2. Bilinear interpolation')
    print('3. Bicubic interpolation')
    print('4. super-resolution convolutional neural network')
    algorithm = int(input('Enter a number:'))
    scale_factor = int(input('Enter scaling factor:'))
    scaled_image = None
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)

    interpolation_methods = {
        1: cv2.INTER_NEAREST,
        2: cv2.INTER_LINEAR,
        3: cv2.INTER_CUBIC
    }

    if algorithm in interpolation_methods:
        scaled_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation_methods[algorithm])
    elif algorithm == 4:
        scaled_image = SRCNN(image, scale_factor=scale_factor)
    else:
        raise ValueError("Invalid algorithm choice")

    if algorithm == 1:
        algorithm = "NN"
    elif algorithm == 2:
        algorithm = "BILINEAR"
    elif algorithm == 3:
        algorithm = "BICUBIC"
    elif algorithm == 4:
        algorithm = "SRCNN"

    save_image(scaled_image, algorithm)
    cv2.imshow('Original', image)
    cv2.imshow('Scaled', scaled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
