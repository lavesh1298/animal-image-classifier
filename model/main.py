# main.py
import os
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
data_dir = r"D:\New folder\animal image classifier\data\dataset"

model_path = "animal_model.pth"
class_index_path = "class_names.txt"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset and loader
dataset = datasets.ImageFolder(data_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 15)  # 15 classes
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):  # You can increase this
    running_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# Save model and class names
torch.save(model.state_dict(), model_path)
with open(class_index_path, "w") as f:
    for class_name in dataset.classes:
        f.write(class_name + "\n")

print("Training complete.")

