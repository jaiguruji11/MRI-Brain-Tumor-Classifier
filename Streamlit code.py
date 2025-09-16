import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Define the same model architecture as in your training script
class MRINet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(pretrained=False) # No need to download pretrained weights again
        self.base.fc = nn.Linear(self.base.fc.in_features, 4)

    def forward(self, x):
        return self.base(x)

# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    # Instantiate the model and load the saved state_dict
    model = MRINet()
    model.load_state_dict(torch.load('mri_classifier.pth'))
    model.eval()
    return model

# Load the trained model
model = load_model()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Define class labels
classes = ['glioma', 'meningioma', 'notumor', 'pituitary'] # Make sure this matches your dataset

st.title("MRI Brain Tumor Classifier")
st.write("Upload an MRI scan to get a prediction.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan', width='stretch')

    # Add the classify button
    if st.button('Classify'):
        st.write("Classifying...")
        
        # Preprocess the image
        img_tensor = transform(image)
        img_tensor = img_tensor.unsqueeze(0) # Add a batch dimension

        # Make a prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()
        
        # Display the result
        st.success(f"Prediction: {classes[predicted_class_idx]} with confidence: {confidence:.2f}")