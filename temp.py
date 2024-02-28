import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import FaceAntiSpoofingModel
from datasets import FaceAntiSpoofingDataset
from torch.utils.data import DataLoader, random_split
from facenet_pytorch import MTCNN, InceptionResnetV1

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

# Initialize the model, MTCNN, and other components
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
face_antispoofing_model = FaceAntiSpoofingModel()  # Replace with your model class

# Define the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(face_antispoofing_model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    face_antispoofing_model.train()
    for inputs, labels in train_loader:
        # Assuming MTCNN is used for face detection and facenet_model for feature extraction
        print(inputs)
        faces = mtcnn(inputs)
        embeddings = facenet_model(faces)
        
        # Your Face Anti-Spoofing model forward pass
        outputs = face_antispoofing_model(embeddings)
        
        # Calculate loss and perform backpropagation
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loop
    face_antispoofing_model.eval()
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Similar steps for face detection and feature extraction
            faces = mtcnn(inputs)
            embeddings = facenet_model(faces)
            
            # Your Face Anti-Spoofing model forward pass for validation
            outputs = face_antispoofing_model(embeddings)
            
            # Calculate validation accuracy
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
    # Calculate validation accuracy
    validation_accuracy = correct_predictions / total_samples
    # Print or log training/validation metrics
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Validation Accuracy: {validation_accuracy}")
