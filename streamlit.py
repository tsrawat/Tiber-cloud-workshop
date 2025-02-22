import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchmetrics import Accuracy, Precision, Recall
import numpy as np

# Define CNN for Fashion MNIST (as given in your code)
class MultiClassImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Single channel (grayscale)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 14 * 14, num_classes)  # Adjust for the 28x28 input shape

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Function to preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
        transforms.Resize((28, 28)),                 # Resize image to 28x28 for FashionMNIST model
        transforms.ToTensor(),                       # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,))         # Normalize as expected by the model
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to load model
def load_model():
    # Load the model structure and trained weights
    model = MultiClassImageClassifier(num_classes=10)  # 10 classes for FashionMNIST
    # Placeholder: Load saved weights (if available)
    # model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model

# Streamlit UI setup
st.title("Fashion MNIST Image Classifier")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image to classify", type=["jpg", "png", "jpeg"])

# Load FashionMNIST dataset to get class labels
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
classes = train_data.classes

# If the user uploads an image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image_tensor = preprocess_image(image)

    # Load the trained model
    model = load_model()

    # Perform prediction
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1).item()

    # Display prediction result
    st.write(f"Predicted class: {classes[prediction]}")

    # Optionally, display metrics or information on the model (pre-trained accuracy etc.)
    # Example: Accuracy of the trained model, but this would need to be precomputed
    # st.write(f"Model Accuracy: {precomputed_accuracy}")
