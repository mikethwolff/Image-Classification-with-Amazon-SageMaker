import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from PIL import ImageFile
import boto3

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants for training
TRAIN = 'train'
LOGGING_THRESHOLD = 200  # Log every 200 batches

# Function to download the pretrained model from S3
def download_resnet50_from_s3(s3_bucket, s3_key, local_path):
    print(f"Downloading ResNet-50 model from s3://{s3_bucket}/{s3_key} to {local_path}")
    s3 = boto3.client('s3')
    s3.download_file(s3_bucket, s3_key, local_path)

# Define the training loop
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    running_samples = 0
    total_samples = len(train_loader.dataset)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        running_samples += len(inputs)

        # Log progress
        if (batch_idx + 1) % LOGGING_THRESHOLD == 0:
            print(
                f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}, "
                f"Accuracy: {running_corrects / running_samples:.4f}"
            )

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    # Log training metrics for SageMaker
    print(f"train:epoch_loss={epoch_loss}, train:epoch_accuracy={epoch_acc}")

    return epoch_loss, epoch_acc

# Define the data loader creation
def create_data_loaders(train_data_dir, batch_size):
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = torchvision.datasets.ImageFolder(train_data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return train_loader, len(train_dataset.classes)

# Define the model
def net(num_classes, local_model_path):
    # Load the pretrained model from the local path
    model = models.resnet50(pretrained=False)
    print(f"Loading ResNet-50 weights from {local_model_path}")
    model.load_state_dict(torch.load(local_model_path))

    # Freeze all layers except the final one
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last layer for classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model

# Save the trained model
def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    print(f"Saving the model to {path}")
    torch.save(model.state_dict(), path)

# Main function for training
def main(args):
    print(f"Arguments received: {args}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download ResNet-50 from S3
    local_model_path = "/tmp/resnet50.pth"
    download_resnet50_from_s3(
        s3_bucket="sagemaker-us-east-1-861747698849",
        s3_key="resnet50/resnet50-19c8e357.pth",
        local_path=local_model_path
    )

    # Create data loaders
    train_loader, num_classes = create_data_loaders(args.data_dir_train, args.batch_size)

    # Initialize the model
    model = net(num_classes, local_model_path).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, Accuracy={train_acc:.4f}")

    # Save the trained model
    save_model(model, args.model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning with SageMaker")

    # Add arguments for hyperparameters and environment variables
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"), help="Directory to save the model")
    parser.add_argument("--data-dir-train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "./data/train"), help="Directory for training data")

    args = parser.parse_args()
    main(args)
