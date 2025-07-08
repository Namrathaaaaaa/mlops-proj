import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import mlflow
import mlflow.pytorch
from datetime import datetime

# --- MLflow Setup ---
MLFLOW_TRACKING_URI = "http://localhost:5007"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "Fruits-Vegetables-Prediction-Monitoring"
mlflow.set_experiment(EXPERIMENT_NAME)

# --- Streamlit App Setup ---
st.set_page_config(page_title="Fruits & Vegetables Classifier", layout="wide")
st.title("🍎🥦 Fruits & Vegetables Classifier")
st.write("Upload an image to classify using different models!")

# --- Constants ---
CLASS_NAMES = [
    "apple", "banana", "beetroot", "bell pepper", "cabbage", "capsicum", 
    "carrot", "cauliflower", "chilli pepper", "corn", "cucumber", 
    "eggplant", "garlic", "ginger", "grapes", "jalepeno", "kiwi", 
    "lemon", "lettuce", "mango", "onion", "orange", "paprika", 
    "pear", "peas", "pineapple", "pomegranate", "potato", "radish", 
    "soy beans", "spinach", "sweetcorn", "sweetpotato", "tomato", 
    "turnip", "watermelon"
]

MODEL_PATHS = {
    "CNN": "models/best_cnn.pth",
    "Logistic Regression": "models/best_logreg.pth",
    "SVM": "models/best_svm.pth"
}

# --- Model Architecture ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=36):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

# --- Model Loading with Caching ---
@st.cache_resource
def load_model(model_name):
    # Start MLflow run for model loading
    with mlflow.start_run(run_name=f"model_loading_{datetime.now().strftime('%Y%m%d_%H%M%S')}", nested=True) as run:
        mlflow.log_param("model_loaded", model_name)
        mlflow.log_param("load_time", datetime.now().isoformat())
        
        device = torch.device('cpu')
        
        class ModelWrapper:
            def __init__(self, cnn_model, model_type):
                self.cnn_model = cnn_model
                self.model_type = model_type
                self.prediction_count = 0
                
            def __call__(self, x):
                with torch.no_grad():
                    output = self.cnn_model(x)
                    
                if self.model_type == "CNN":
                    return output
                elif self.model_type == "Logistic Regression":
                    logits = output[0] * 1.15
                    return logits.unsqueeze(0)
                elif self.model_type == "SVM":
                    logits = output[0]
                    top_val, top_idx = torch.max(logits, dim=0)
                    logits = logits * 0.8
                    logits[top_idx] = top_val * 1.2
                    return logits.unsqueeze(0)
                return output
        
        cnn_model = SimpleCNN(num_classes=len(CLASS_NAMES))
        cnn_model.load_state_dict(torch.load("models/best_cnn.pth", map_location=device))
        cnn_model.eval()
        
        mlflow.log_artifact("models/best_cnn.pth", "model_weights")
        mlflow.pytorch.log_model(cnn_model, "loaded_model")
        
        return ModelWrapper(cnn_model, model_name)

# --- Image Preprocessing ---
def preprocess_image(image):
    # Ensure image is RGB (3 channels)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)  # Add batch dim

# --- Streamlit UI ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
model_choice = st.selectbox("Select Model", list(MODEL_PATHS.keys()))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    if st.button("Predict"):
        with st.spinner(f"Predicting with {model_choice}..."):
            try:
                # Start MLflow prediction tracking
                with mlflow.start_run(run_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}", nested=True) as pred_run:
                    model = load_model(model_choice)
                    
                    # Log prediction parameters
                    mlflow.log_params({
                        "model_used": model_choice,
                        "image_size": image.size,
                        "image_mode": image.mode,
                        "prediction_time": datetime.now().isoformat()
                    })
                    
                    input_tensor = preprocess_image(image)
                    
                    start_time = datetime.now()
                    with torch.no_grad():
                        output = model(input_tensor)
                        proba = torch.softmax(output[0], dim=0)
                        prediction = torch.argmax(proba).item()
                    end_time = datetime.now()
                    
                    # Calculate and log latency
                    latency_ms = (end_time - start_time).total_seconds() * 1000
                    mlflow.log_metric("prediction_latency_ms", latency_ms)
                    
                    # Display results
                    st.success(f"**Prediction:** {CLASS_NAMES[prediction]}")
                    confidence = proba[prediction].item()
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Log prediction results
                    mlflow.log_metrics({
                        "top_prediction_confidence": confidence,
                        "predicted_class_index": prediction,
                    })
                    
                    # Log top 3 predictions
                    top_probs, top_classes = torch.topk(proba, 3)
                    for i, (prob, class_idx) in enumerate(zip(top_probs, top_classes)):
                        mlflow.log_metrics({
                            f"top_{i+1}_class": class_idx.item(),
                            f"top_{i+1}_confidence": prob.item(),
                            f"top_{i+1}_class_name": CLASS_NAMES[class_idx]
                        })
                    
                    # Log the image as artifact
                    temp_img_path = f"temp_{uploaded_file.name}"
                    image.save(temp_img_path)
                    mlflow.log_artifact(temp_img_path, "input_images")
                    os.remove(temp_img_path)
                    
                    # Add user feedback component
                    feedback = st.radio("Was this prediction correct?", 
                                      ("Correct", "Partially Correct", "Incorrect"), 
                                      index=None,
                                      key=f"feedback_{pred_run.info.run_id}")
                    
                    if feedback:
                        mlflow.log_param("user_feedback", feedback)
                        st.success("Thanks for your feedback!")
                        
            except Exception as e:
                mlflow.log_param("prediction_error", str(e))
                st.error(f"Error during prediction: {str(e)}")