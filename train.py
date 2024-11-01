import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights


def get_train_transform():
    # Define transformations for the training set
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
            transforms.Lambda(
                lambda x: x * torch.tensor([1.5, 0.9, 0.9]).view(3, 1, 1)
            ),  # Adjust color channels
        ]
    )
    return transform


def get_val_transform():
    # Define transformations for the validation set (no augmentation)
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
            transforms.Lambda(
                lambda x: x * torch.tensor([1.5, 0.9, 0.9]).view(3, 1, 1)
            ),  # Adjust color channels
        ]
    )
    return transform


data_dir = "train_data"
train_transform = get_train_transform()
val_transform = get_val_transform()

# Load the dataset from the specified directory
full_dataset = datasets.ImageFolder(root=data_dir)

# Split the dataset into training and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Assign transforms to the datasets
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["cherry", "tomato", "strawberry"]
num_classes = len(classes)


class SimpleMLP(nn.Module):
    def __init__(self, num_classes):
        super(SimpleMLP, self).__init__()
        # Define a simple MLP with a single fully connected layer
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten the input tensor
            nn.Linear(128 * 128 * 3, num_classes)  # Fully connected layer
        )

    def forward(self, x):
        x = self.fc_layers(x)
        return x


class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaselineCNN, self).__init__()
        # Define convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # First conv layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Second conv layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Third conv layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Define fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = self.classifier(x)
        return x


class EnsembleCNN(nn.Module):
    def __init__(self, num_classes):
        super(EnsembleCNN, self).__init__()
        # Initialize BaselineCNN
        self.baseline_cnn = BaselineCNN(num_classes)
        # Load pre-trained ResNet18
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace the final layer of ResNet18 with a new layer
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, num_classes)

        # Final classification layer combining both models
        self.fc = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x):
        # Get outputs from both models
        x1 = self.baseline_cnn(x)
        x2 = self.resnet18(x)
        # Concatenate outputs
        x_combined = torch.cat((x1, x2), dim=1)
        # Final classification
        out = self.fc(x_combined)
        return out


# Initialize models and move them to the device
mlp_model = SimpleMLP(num_classes=num_classes).to(device)
baseline_cnn_model = BaselineCNN(num_classes=num_classes).to(device)
ensemble_cnn_model = EnsembleCNN(num_classes=num_classes).to(device)

# Define loss functions and optimizers for each model
mlp_criterion = nn.CrossEntropyLoss()
mlp_optimiser = optim.Adam(mlp_model.parameters(), lr=0.001)

baseline_cnn_criterion = nn.CrossEntropyLoss()
baseline_cnn_optimiser = optim.Adam(
    baseline_cnn_model.parameters(), lr=0.001
)

ensemble_cnn_criterion = nn.CrossEntropyLoss()
ensemble_cnn_optimiser = optim.Adam(
    ensemble_cnn_model.parameters(), lr=0.0001  # Lower LR for fine-tuning
)

# Set number of epochs for training
mlp_num_epochs = 10
baseline_cnn_num_epochs = 10
ensemble_cnn_num_epochs = 10


def train_model(
    model, train_loader, val_loader, num_epochs, model_name, criterion, optimiser
):
    # Lists to store training and validation metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    times = []

    start_time = time.time()
    cumulative_time = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()  # Set model to training mode
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Training loop
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimiser.step()  # Update parameters

            running_loss += loss.item() * images.size(0)  # Accumulate loss
            _, predicted = torch.max(outputs.data, 1)  # Get predictions
            total_correct += (predicted == labels).sum().item()  # Count correct predictions
            total_samples += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / total_samples
        epoch_acc = total_correct / total_samples
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        model.eval()  # Set model to evaluation mode
        val_running_loss = 0.0
        val_total_correct = 0
        val_total_samples = 0

        # Validation loop
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item() * val_images.size(0)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total_correct += (val_predicted == val_labels).sum().item()
                val_total_samples += val_labels.size(0)

        # Calculate average validation loss and accuracy for the epoch
        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = val_total_correct / val_total_samples
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        cumulative_time += epoch_duration
        times.append(cumulative_time)

        # Print training and validation statistics
        print(
            f"{model_name} Epoch [{epoch+1}/{num_epochs}] "
            f"Time: {cumulative_time:.2f}s "
            f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f} "
            f"Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.4f}"
        )

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"{model_name} Total Training Time: {total_training_time:.2f} seconds")

    # Save the trained model
    if model_name == "EnsembleCNN":
        torch.save(model.state_dict(), "model.pth")
        print(f"{model_name} Model saved as 'model.pth'")
    else:
        torch.save(model.state_dict(), f"{model_name}_model.pth")
        print(f"{model_name} Model saved as '{model_name}_model.pth'")

    # Save metrics to CSV
    with open(f"{model_name}_metrics.csv", "w", newline="") as csvfile:
        fieldnames = [
            "epoch",
            "time",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(num_epochs):
            writer.writerow(
                {
                    "epoch": i + 1,
                    "time": times[i],
                    "train_loss": train_losses[i],
                    "train_accuracy": train_accuracies[i],
                    "val_loss": val_losses[i],
                    "val_accuracy": val_accuracies[i],
                }
            )
    print(f"{model_name} Metrics saved as '{model_name}_metrics.csv'")

    return total_training_time


# Train the SimpleMLP model
mlp_training_time = train_model(
    mlp_model,
    train_loader,
    val_loader,
    mlp_num_epochs,
    "SimpleMLP",
    criterion=mlp_criterion,
    optimiser=mlp_optimiser,
)

# Train the BaselineCNN model
baseline_cnn_training_time = train_model(
    baseline_cnn_model,
    train_loader,
    val_loader,
    baseline_cnn_num_epochs,
    "BaselineCNN",
    criterion=baseline_cnn_criterion,
    optimiser=baseline_cnn_optimiser,
)

# Train the EnsembleCNN model
ensemble_cnn_training_time = train_model(
    ensemble_cnn_model,
    train_loader,
    val_loader,
    ensemble_cnn_num_epochs,
    "EnsembleCNN",
    criterion=ensemble_cnn_criterion,
    optimiser=ensemble_cnn_optimiser,
)
