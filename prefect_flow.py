# from prefect import flow, task
# import subprocess

# @task
# def run_streamlit_app():
#     # This will launch your Streamlit app
#     subprocess.run(["streamlit", "run", "app.py"], check=True)

# @flow
# def ml_pipeline():
#     run_streamlit_app()

# if __name__ == "__main__":
#     ml_pipeline()


from prefect import flow, task

# flow_pipeline.py
from prefect import flow, task
from datetime import datetime
import torch
import mlflow
from PIL import Image
import torchvision.transforms as transforms

@task
def load_model(model_path):
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    return model

@task
def preprocess(img_path):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0)

@task
def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        return torch.argmax(torch.softmax(output[0], dim=0)).item()

@flow(name="🍎 ML Prediction Flow")
def fruit_predict_pipeline(model_path: str, image_path: str):
    mlflow.set_tracking_uri("http://localhost:5007")
    mlflow.set_experiment("Fruits-Vegetables-Classifier")
    with mlflow.start_run(run_name="Prefect_CLI_Run"):
        model = load_model(model_path)
        tensor = preprocess(image_path)
        prediction = predict(model, tensor)
        mlflow.log_param("predicted_class_index", prediction)
        print(f"✅ Prediction Index: {prediction}")
        from prefect import flow, task

@task
def preprocess_data():
    print("Preprocessing data...")
    # Return some dummy data for illustration
    return "processed_data"

@task
def train_model(data):
    print(f"Training model on {data}...")
    # Return a dummy model
    return "trained_model"

@task
def evaluate_model(model):
    print(f"Evaluating {model}...")
    return "evaluation_results"

@flow
def ml_pipeline():
    data = preprocess_data()
    model = train_model(data)
    results = evaluate_model(model)
    return results

if __name__ == "__main__":
    ml_pipeline()


