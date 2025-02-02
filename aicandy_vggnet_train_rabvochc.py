"""

@author:  AIcandy 
@website: aicandy.vn

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from aicandy_model_src_cmxakmhr.aicandy_vggnet_model_nirksxki import VGGNet


# python aicandy_vggnet_train_rabvochc.py --train_dir ../dataset --num_epochs 10 --batch_size 32 --model_path aicandy_model_out_gxbruthb/aicandy_model_pth_aootldae.pth

def train(train_dir, num_epochs, batch_size, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training with: ", device)
    
    # Define data transformations
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = transform_val
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Save class names to label.txt
    with open('label.txt', 'w') as f:
        for idx, class_name in enumerate(dataset.classes):
            f.write(f"{idx}: {class_name}\n")
    
    # Initialize model, criterion, and optimizer
    model = VGGNet(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Model saved with accuracy: {best_accuracy:.2f}%")

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to save the best model')

    args = parser.parse_args()
    train(args.train_dir, args.num_epochs, args.batch_size, args.model_path)
