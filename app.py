import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import timm
from PIL import Image
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Model Predictor", page_icon="🔍")

# --- 1. UTILS: LOAD CLASS NAMES ---
@st.cache_data
def load_class_names(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        # Read lines and strip whitespace
        classes = [line.strip() for line in f.readlines()]
    return classes

# --- 2. CORE: MODEL LOADER (Matches your predict.py) ---
@st.cache_resource
def load_model(model_name, num_classes, model_path, device):
    try:
        # A. Initialize Architecture
        if model_name == "coat_lite_mini":
            # Matches: timm.create_model(model_name, pretrained=False, num_classes=...)
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
            
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
        elif model_name == "resnet18":
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            
        else:
            # Fallback for other timm models
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

        # B. Load Weights
        # Matches: torch.load(..., map_location=device, weights_only=True)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        
        # Safety: Remove 'module.' prefix if trained on multi-GPU
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(clean_state_dict)
        model.to(device)
        model.eval()
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 3. PREPROCESSING (Replaces create_submission_dataloader) ---
def process_image(image):
    """
    Standard preprocessing for CoaT/ImageNet models.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Mandatory for coat_lite_mini
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

# --- 4. MAIN UI ---
def main():
    st.title("Web Classification App")
    st.write("Upload an image to test your `coat_lite_mini` model.")

    # -- Sidebar Setup --
    st.sidebar.header("Settings")
    
    # Model Selection
    model_name = st.sidebar.selectbox(
        "Model Architecture", 
        ["coat_lite_mini", "efficientnet_b0", "resnet18"]
    )
    
    # Weight Selection
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir) # Create it if it doesn't exist
        
    weight_files = [f for f in os.listdir(weights_dir) if f.endswith(".pth")]
    
    if weight_files:
        selected_weight = st.sidebar.selectbox("Select Weights", weight_files)
        model_path = os.path.join(weights_dir, selected_weight)
    else:
        st.sidebar.error("No .pth files found in 'weights/' folder.")
        model_path = None

    # Load Classes
    classes = load_class_names("class_names.txt")
    if not classes:
        st.sidebar.warning("No class_names.txt found. Using numbers.")
        num_classes = 10 # Default fallback
    else:
        num_classes = len(classes)
        st.sidebar.success(f"Loaded {num_classes} classes.")

    # -- Prediction Logic --
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_file and model_path:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        if st.button("Predict"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            with st.spinner("Analyzing..."):
                # 1. Load
                model = load_model(model_name, num_classes, model_path, device)
                
                if model:
                    # 2. Process
                    input_tensor = process_image(image).to(device)
                    
                    # 3. Predict
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # 4. Show Results
                    top_prob, top_class = torch.topk(probs, 1)
                    idx = top_class.item()
                    score = top_prob.item()
                    
                    if classes:
                        result_name = classes[idx]
                    else:
                        result_name = f"Class {idx}"

                    st.success(f"Prediction: **{result_name}**")
                    st.info(f"Confidence: **{score*100:.2f}%**")

if __name__ == "__main__":
    main()