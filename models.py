import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import mlflow
import mlflow.pytorch
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# MLflow setup
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Fruits-Vegetables-Classification"

# Function to safely set up MLflow
def setup_mlflow():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return True
    except Exception as e:
        print(f"WARNING: MLflow setup failed: {str(e)}")
        print("Training will continue without MLflow tracking.")
        return False

# Initialize MLflow
mlflow_available = setup_mlflow()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define paths
BASE_PATH = "./data/fruits-vegetables"  # Path to your dataset folder
TRAIN_PATH = os.path.join(BASE_PATH, "train")
VAL_PATH = os.path.join(BASE_PATH, "validation")
TEST_PATH = os.path.join(BASE_PATH, "test")
MODEL_PATH = "./models"
os.makedirs(MODEL_PATH, exist_ok=True)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Smaller size for faster training
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets from predefined splits
train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
val_dataset = datasets.ImageFolder(root=VAL_PATH, transform=transform)
test_dataset = datasets.ImageFolder(root=TEST_PATH, transform=transform)

# Print dataset sizes
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define models
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)

class LinearSVM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearSVM, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)  # Adjusted for 128x128 input
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.1, model_name="model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} on {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() if model_name != "svm" else nn.MultiMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    # Start an MLflow run for this training session
    with mlflow.start_run(run_name=f"{model_name}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        # Log model parameters
        mlflow.log_params({
            "model_name": model_name,
            "learning_rate": lr,
            "optimizer": optimizer.__class__.__name__,
            "num_epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "device": str(device),
            "criterion": criterion.__class__.__name__,
            "model_architecture": model.__class__.__name__
        })
        
        # Log model architecture as artifact
        with open("model_architecture.txt", "w") as f:
            f.write(str(model))
        mlflow.log_artifact("model_architecture.txt")
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            train_bar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs}")
            for images, labels in train_bar:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                train_bar.set_postfix({
                    "Loss": f"{train_loss/(train_bar.n+1):.4f}",
                    "Acc": f"{100*correct/total:.2f}%"
                })
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}: Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"best_{model_name}.pth"))
            print(f"New best model saved with val acc: {val_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"final_{model_name}.pth"))
    return model

# Initialize and train models
num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")

# Logistic Regression (using flattened image size)
input_size = 3 * 128 * 128
print("\nTraining Logistic Regression...")
logreg_model = LogisticRegression(input_size, num_classes)
train_model(logreg_model, train_loader, val_loader, num_epochs=10, model_name="logreg")

# SVM
print("\nTraining SVM...")
svm_model = LinearSVM(input_size, num_classes)
train_model(svm_model, train_loader, val_loader, num_epochs=10, model_name="svm")

# CNN
print("\nTraining CNN...")
cnn_model = SimpleCNN(num_classes)
train_model(cnn_model, train_loader, val_loader, num_epochs=10, lr=0.0005, model_name="cnn")

# Test function
def test_model(model, test_loader, model_name="model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"best_{model_name}.pth")))
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"{model_name} Test Accuracy: {100 * correct / total:.2f}%")

# Test all models
print("\nTesting models...")
test_model(logreg_model, test_loader, "logreg")
test_model(svm_model, test_loader, "svm")
test_model(cnn_model, test_loader, "cnn")