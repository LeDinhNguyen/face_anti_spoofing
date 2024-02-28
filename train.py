import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import MyMobileNetV2
from datasets import FaceAntiSpoofingDataset
from torch.utils.data import DataLoader, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define transforms and other training configurations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the dataset
face_antispoofing_dataset = FaceAntiSpoofingDataset(root_dir='./datasets', celeba_split='train', lccfasd_split='training', transform=transform)

# Define the sizes of train and validation sets
train_size = int(0.8 * len(face_antispoofing_dataset))
val_size = len(face_antispoofing_dataset) - train_size

# Split the dataset
train_set, val_set = random_split(face_antispoofing_dataset, [train_size, val_size])

# Define training parameters
batch_size = 32
epochs = 10
learning_rate = 0.001

# Create data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Initialize model and optimizer
model = MyMobileNetV2().to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    print("Start train")
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    # Validation loop
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs_val, labels_val in val_loader:
            inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)

            outputs_val = model(inputs_val)
            _, predicted_val = torch.max(outputs_val.data, 1)

            total_val += labels_val.size(0)
            correct_val += (predicted_val == labels_val).sum().item()

        val_accuracy = correct_val / total_val
        print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'face_anti_spoofing_model.pth')