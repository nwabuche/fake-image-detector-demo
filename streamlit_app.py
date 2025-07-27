import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from timm import create_model

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load CNN model (resnet18)
model_cnn = create_model("resnet18", pretrained=False, num_classes=2)
model_cnn.load_state_dict(torch.load("cnn_middle_dataset.pth", map_location=device))
model_cnn.eval().to(device)

# Load ViT model (vit_small_patch16_224, embed_dim=384)
model_vit = create_model("vit_small_patch16_224", pretrained=False, num_classes=2)
model_vit.load_state_dict(torch.load("vit_middle.pth", map_location=device))
model_vit.eval().to(device)

# Define class names
class_names = ["Fake", "Real"]

st.title("Fake Image Detection")
st.write("Upload a face image to detect if it's **Real** or **Fake**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        cnn_pred = model_cnn(img_tensor)
        vit_pred = model_vit(img_tensor)

    cnn_label = class_names[torch.argmax(cnn_pred, dim=1).item()]
    vit_label = class_names[torch.argmax(vit_pred, dim=1).item()]

    st.markdown("### Predictions")
    st.write(f"**CNN Prediction:** {cnn_label}")
    st.write(f"**ViT Prediction:** {vit_label}")
