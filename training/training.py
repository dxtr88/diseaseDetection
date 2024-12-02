import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from augmentation import RobustTraining

class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the plant disease images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Verify directory exists
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory not found: {root_dir}")
            
        # Get all subdirectories (classes)
        self.classes = [d for d in sorted(os.listdir(root_dir)) 
                       if os.path.isdir(os.path.join(root_dir, d))]
        
        if not self.classes:
            raise ValueError(f"No class directories found in {root_dir}")
            
        print(f"Found {len(self.classes)} classes: {self.classes}")
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load all image paths and labels
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                class_idx = self.class_to_idx[class_name]
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(class_idx)
        
        if not self.images:
            raise ValueError(f"No valid images found in {root_dir}")
            
        print(f"Loaded {len(self.images)} images total")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    """Enhanced data transformations for better generalization"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Larger initial size for random cropping
        transforms.RandomCrop(224),     # Random crop for position invariance
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.3, 
            contrast=0.3, 
            saturation=0.3, 
            hue=0.1
        ),
        transforms.RandomGrayscale(p=0.1),  # Simulate different color conditions
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # Perspective changes
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # More realistic variations
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)  # Simulate occlusions
    ])

    # Keep validation transform simple
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def create_model(num_classes):
    """Create and configure the model"""
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda'):
    """Train the model and return training history"""
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Initialize robust training
    robust_trainer = RobustTraining(model, device)
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        print("Training phase:")
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            total_samples += inputs.size(0)

            # Use robust training step
            loss, outputs = robust_trainer.train_step(
                inputs, 
                labels, 
                optimizer, 
                criterion,
                mixup_alpha=0.2,
                adv_epsilon=0.007,
                return_outputs=True  # Make sure RobustTraining class returns outputs
            )
            
            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            running_loss += loss * inputs.size(0)
            
            # Print batch progress
            if (batch_idx + 1) % 10 == 0:
                avg_loss = running_loss / total_samples
                avg_acc = running_corrects.double() / total_samples
                print(f'Batch [{batch_idx + 1}/{len(train_loader)}] - '
                      f'Loss: {avg_loss:.4f} - Acc: {avg_acc:.4f}')

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        print(f'\nTraining Epoch Summary:')
        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        print("\nValidation phase:")
        total_val_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Print batch progress
                if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
                    print(f'Validation Batch [{batch_idx + 1}/{total_val_batches}] - '
                          f'Running Loss: {running_loss / ((batch_idx + 1) * inputs.size(0)):.4f}')

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())

        print(f'\nValidation Epoch Summary:')
        print(f'Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'\nNew best validation accuracy: {best_val_acc:.4f}')
            print('Saving model checkpoint...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history,
            }, 'best_plant_disease_model.pth')

        print('-' * 50)

    return model, history

def main():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up data transforms
    train_transform, val_transform = get_transforms()

    # Create datasets
    try:
        # Use only the correct path
        data_dir = "data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
        
        if not os.path.exists(data_dir):
            raise ValueError(f"Dataset directory not found at: {data_dir}")
        
        print(f"Loading dataset from: {data_dir}")
        full_dataset = PlantDiseaseDataset(data_dir, transform=train_transform)
        
        # Split dataset
        train_size = 0.8
        train_length = int(train_size * len(full_dataset))
        val_length = len(full_dataset) - train_length
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_length, val_length]
        )
        
        print(f"\nDataset split:")
        print(f"Total images: {len(full_dataset)}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Update validation transform
        val_dataset.dataset.transform = val_transform

        # Create data loaders
        print("\nCreating data loaders...")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Create model
        num_classes = len(full_dataset.classes)
        print(f"\nInitializing model with {num_classes} classes...")
        model = create_model(num_classes)
        model = model.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("Using Adam optimizer with learning rate: 0.001")

        print("\nStarting training...")
        # Train model
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=25,
            device=device
        )

        print("\nTraining completed!")
        return model, history
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
