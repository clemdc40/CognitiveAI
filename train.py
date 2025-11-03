#["Age", "Gender", "BMI", "Smoking", "AlcoholConsumption", "PhysicalActivity", "SleepQuality", "FamilyHistoryAlzheimers", "MMSE", "FunctionalAssessment", "ADL", "Diagnosis"]

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('alzheimers_disease_data.csv')

# Preprocess dataset
x_scaled = data[["Age", "Gender", "BMI", "Smoking", "AlcoholConsumption", "PhysicalActivity", "SleepQuality", "FamilyHistoryAlzheimers", "MMSE", "FunctionalAssessment", "ADL"]].values
y = data["Diagnosis"].values

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_scaled)
x = torch.tensor(x_scaled, dtype=torch.float32)
print(f"Features shape: {x.shape}, Labels shape: {y.shape}")

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define model
def build_model(input_size, num_classes):
    model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
    return model

num_classes = len(set(y))
model = build_model(x_train.shape[1], num_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(model)

# Train model
epochs = 1000
for epoch in range(epochs):
    model.train()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate model
model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    _, predicted = torch.max(y_pred, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f'Accuracy: {accuracy.item():.4f}')


# test du modèle avec un échantillon
sample = torch.tensor([[70, 1, 25.0, 0, 1, 3, 4, 1, 28, 10, 5]], dtype=torch.float32)
with torch.no_grad():
    logits = model(sample)
    predicted_class = torch.argmax(torch.softmax(logits, dim=1))
print(f"Predicted class index for sample: {predicted_class.item()}")