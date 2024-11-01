import os
import time
import csv
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dir = "testdata"
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"The directory '{test_dir}' does not exist.")

classes = ['cherry', 'tomato', 'strawberry']
num_classes = len(classes)


def get_transform():
    # Define transformations for the test set (no augmentation)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x * torch.tensor(
            [1.5, 0.9, 0.9]).view(3, 1, 1))  # Adjust color channels
    ])
    return transform


transform = get_transform()

# Load the test dataset
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


class SimpleMLP(nn.Module):
    def __init__(self, num_classes):
        super(SimpleMLP, self).__init__()
        # Define a simple MLP with a single fully connected layer
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 3, num_classes)
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
        # Replace the final layer of ResNet18
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


test_results = []

# List of models with their corresponding file names and classes
models_info = [
    {
        'model_name': 'SimpleMLP',
        'file_name': 'SimpleMLP_model.pth',
        'class': SimpleMLP
    },
    {
        'model_name': 'BaselineCNN',
        'file_name': 'BaselineCNN_model.pth',
        'class': BaselineCNN
    },
    {
        'model_name': 'EnsembleCNN',
        'file_name': 'model.pth',  # EnsembleCNN is saved as 'model.pth'
        'class': EnsembleCNN
    }
]

# Loop over each model and evaluate if the .pth file exists
for model_info in models_info:
    model_name = model_info['model_name']
    file_name = model_info['file_name']
    model_class = model_info['class']

    if os.path.exists(file_name):
        print(f"\nEvaluating model: {file_name}")

        # Load the model
        model = model_class(num_classes)
        model.load_state_dict(torch.load(file_name, map_location=device))
        model = model.to(device)
        model.eval()

        start_time = time.time()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        end_time = time.time()
        test_time = end_time - start_time

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy * 100:.2f}%')
        print(f'Test Time: {test_time:.2f} seconds')

        # Generate classification report
        report = classification_report(
            all_labels, all_predictions, target_names=classes, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(
            all_labels, all_predictions, target_names=classes))
        print("################################################")

        test_results.append({
            'model_name': model_name,
            'accuracy': accuracy,
            'test_time': test_time,
            'report': report
        })

    else:
        print(f"\n{file_name} not found, skipping {model_name} model.")

# Check if any models were evaluated
if test_results:
    # Save test results to CSV
    with open('test_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['model_name', 'accuracy', 'test_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in test_results:
            writer.writerow({
                'model_name': result['model_name'],
                'accuracy': result['accuracy'],
                'test_time': result['test_time']
            })
    print("Test results saved as 'test_results.csv'")

    # Save classification reports to CSV files
    for result in test_results:
        model_name = result['model_name']
        report = result['report']
        report_filename = f'{model_name}_classification_report.csv'
        with open(report_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['class', 'precision', 'recall', 'f1-score', 'support']
            writer.writerow(headers)
            for cls in classes:
                row = [
                    cls,
                    report[cls]['precision'],
                    report[cls]['recall'],
                    report[cls]['f1-score'],
                    report[cls]['support']
                ]
                writer.writerow(row)
        print(f"Classification report saved as '{report_filename}'")
else:
    print("No models were evaluated. Please ensure the .pth files exist in the directory.")
