![image](https://github.com/user-attachments/assets/e975f03d-89eb-41cb-9f50-fef06c95ae0b)


# Fashion MNIST Image Classifier

This project implements a multi-class image classifier for the Fashion MNIST dataset using a Convolutional Neural Network (CNN) and Streamlit for the web interface. Users can upload images to classify them into one of ten fashion categories.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Image Preprocessing](#image-preprocessing)
- [License](#license)

## Requirements

To run this project, you'll need the following Python libraries:

- `streamlit`
- `torch`
- `torchvision`
- `torchmetrics`
- `Pillow`
- `numpy`

You can install the required packages using pip:

```bash
pip install streamlit torch torchvision torchmetrics Pillow numpy
```

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:

   ```bash
   cd <project-directory>
   ```

3. Install the required libraries as mentioned above.

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the file uploader to upload an image of clothing. The model will predict the class of the uploaded image.

## Model Architecture

The CNN architecture consists of the following layers:

- **Convolutional Layer:** Applies 16 filters of size 3x3 with ReLU activation and padding.
- **Max Pooling Layer:** Reduces dimensionality with a kernel size of 2x2.
- **Fully Connected Layer:** Outputs class predictions for the 10 categories in Fashion MNIST.

## Image Preprocessing

Uploaded images are preprocessed using the following steps:

1. Convert the image to grayscale.
2. Resize the image to 28x28 pixels.
3. Normalize the pixel values to be between -1 and 1.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
