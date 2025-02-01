import os
import cv2
from datetime import datetime
from model import SuperResolutionConvolutionalNetwork


def load_image():
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    files = os.listdir(os.getcwd())
    images = []
    for file in files:
        if file.lower().endswith(extensions):
            images.append(file)
    return cv2.imread(images[0])


def SRCNN(image, scale_factor=2):
    model = SuperResolutionConvolutionalNetwork().load_model()
    if model is None:
        print('No model found.')
        print('Creating new model and training.')
        model = SuperResolutionConvolutionalNetwork(scale_factor=scale_factor).build()
        model.download_dataset()
        model.load_dataset()
        model.train(epochs=100)
        model.save_model()
    scaled_image = model.enhance_image(image)
    return scaled_image


def save_image(image):
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    name = f'{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.jpg'
    output_path = os.path.join(output_dir, name)
    if cv2.imwrite(output_path, image):
        print(f"Image saved at: {output_path}")
    else:
        print("Failed to save the image.")


def main():
    image = load_image()
    print('1. Nearest neighbor')
    print('2. Bilinear interpolation')
    print('3. Bicubic interpolation')
    print('4. super-resolution convolutional neural network')
    algorithm = int(input('Enter a number:'))
    scale_factor = int(input('Enter scaling factor:'))
    scaled_image = None
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    match algorithm:
        case 1:
            scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        case 2:
            scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        case 3:
            scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        case 4:
            scaled_image = SRCNN(image, scale_factor)
    cv2.imshow('Original', image)
    cv2.imshow('Scaled', scaled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    save_image(scaled_image)


if __name__ == "__main__":
    main()
