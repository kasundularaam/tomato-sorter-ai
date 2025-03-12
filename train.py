import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

# Constants
DATA_DIR = 'dataset'
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
IMG_SIZE = 224
TRAIN_RATIO = 0.8
RESULTS_DIR = 'results'

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class TomatoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['ripe', 'damaged']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []

        # Load all images and their corresponding labels
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: {class_dir} is not a directory")
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append(
                        (img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def create_data_loaders():
    # Define data transforms for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Create a full dataset
    full_dataset = TomatoDataset(root_dir=DATA_DIR, transform=train_transform)

    # Check dataset size
    dataset_size = len(full_dataset)
    if dataset_size == 0:
        raise ValueError(f"No images found in {DATA_DIR}")

    # Split into training and validation sets
    train_size = int(TRAIN_RATIO * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])

    # Update the transform for the validation set
    val_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(
        f"Dataset split: {train_size} training samples, {val_size} validation samples")

    return train_loader, val_loader


def create_model(device):
    # Load a pre-trained ResNet-50 model
    model = models.resnet50(weights='IMAGENET1K_V2')

    # Freeze all layers in the network
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)  # 2 classes: ripe and damaged
    )

    # Move model to the specified device
    model = model.to(device)

    return model


def train_model(model, train_loader, val_loader, device):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Initialize history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # Best validation accuracy (for model saving)
    best_val_acc = 0.0

    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Progress bar for training
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")

        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item()})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")

        # No gradient computation for validation
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar
                val_pbar.set_postfix({'loss': loss.item()})

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects.double() / len(val_loader.dataset)

        # Update learning rate
        scheduler.step(epoch_val_loss)

        # Print epoch results
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['val_acc'].append(epoch_val_acc.item())

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_tomato_model.pth')
            print(
                f"Saved new best model with validation accuracy: {best_val_acc:.4f}")

    return model, history


def evaluate_model(model, val_loader, device):
    # Set model to evaluation mode
    model.eval()

    # Initialize lists to store predictions and ground truth
    all_preds = []
    all_labels = []

    # No gradient computation for evaluation
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Append batch predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=['ripe', 'damaged'])

    return cm, report


def plot_results(history, cm, report):
    # Create directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Plot training and validation loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
                'ripe', 'damaged'], yticklabels=['ripe', 'damaged'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))

    # Save classification report
    with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

    print(f"Results saved to {RESULTS_DIR}")


def main():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders()

    # Show class distribution
    class_counts = {cls: 0 for cls in ['ripe', 'damaged']}
    for _, (_, labels) in enumerate(train_loader):
        for label in labels:
            class_name = 'ripe' if label.item() == 0 else 'damaged'
            class_counts[class_name] += 1

    print("Class distribution in training set:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} images")

    # Create and train model
    model = create_model(device)
    model, history = train_model(model, train_loader, val_loader, device)

    # Evaluate model
    cm, report = evaluate_model(model, val_loader, device)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Plot and save results
    plot_results(history, cm, report)

    print("Training complete!")


if __name__ == '__main__':
    main()
