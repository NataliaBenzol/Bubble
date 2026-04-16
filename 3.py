import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time

class BubbleDataset(Dataset):
    
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_model(num_classes=7, pretrained=True):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=50, device='cuda', save_path='best_model.pth'):
    best_acc = 0.0
    best_gap = float('inf')
    best_combined_score = -float('inf')
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [], 'gap': []}
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        start_time = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = 100. * correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        gap = abs(train_acc - val_acc)
        history['gap'].append(gap)
        #combined_score = val_acc - 0.5 * max(0, gap)
        epoch_time = time.time() - start_time
        print(f'Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Gap: {gap:.2f}% | Time: {epoch_time:.1f}s')
        if val_acc > best_acc and gap < best_gap:
            best_acc = val_acc
            best_gap = gap
            #best_combined_score = combined_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'gap': gap,
                'accuracy': best_acc,
            }, save_path)
            print(f'Saved best model: Val Acc={val_acc:.2f}%, Gap={gap:.2f}%')
        #elif combined_score > best_combined_score:
            #best_combined_score = combined_score
            #torch.save({
                #'epoch': epoch,
                #'model_state_dict': model.state_dict(),
                #'optimizer_state_dict': optimizer.state_dict(),
                #'train_acc': train_acc,
                #'val_acc': val_acc,
                #'gap': gap,
                #'accuracy': val_acc,
            #}, save_path)
            #print(f'Saved best model: Val Acc={val_acc:.2f}%, Gap={gap:.2f}%')
        scheduler.step()
    return history

def evaluate_model(model, test_loader, class_names, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    print(f'\n{"="*50}')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1-Score: {f1:.4f}')
    print(f'{"="*50}')

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    cm = confusion_matrix(all_labels, all_preds)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'report': report,
        'confusion_matrix': cm
    }

def save_results_to_csv(metrics, model_name, output_file='results.csv'):
    results = []
    results.append({
        'Model': model_name,
        'Class': 'overall',
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1_score'],
        'Support': '-'
    })
    for class_name, class_metrics in metrics['report'].items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            results.append({
                'Model': model_name,
                'Class': class_name,
                'Precision': class_metrics['precision'],
                'Recall': class_metrics['recall'],
                'F1-Score': class_metrics['f1-score'],
                'Support': class_metrics['support']
            })
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f'\n Results saved to {output_file}')
    return df

def plot_training_history(history, output_file='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax1.plot(history['val_acc'], label='Val Acc', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training & Validation Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax2.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training & Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f'Training history plot saved to {output_file}')
    plt.close()

def plot_confusion_matrix(cm, class_names, output_file='confusion_matrix.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f'Confusion matrix saved to {output_file}')
    plt.close()

def main():
    DATA_DIR = 'filtered_frames'
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = 7
    MODEL_NAME = 'ResNet50_Transfer_Learning'
    RANDOM_STATE = 42
    
    CLASS_MAPPING = {
        'solution_0pct': 0,
        'solution_5pct': 1,
        'solution_12.5pct': 2,
        'solution_25pct': 3,
        'solution_50pct': 4,
        'solution_75pct': 5,
        'solution_100pct': 6
    }
    
    CLASS_NAMES = list(CLASS_MAPPING.keys())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('\n' + '='*50)
    print('Loading data...')
    print('='*50)
    
    all_samples = []
    
    for class_name, class_id in CLASS_MAPPING.items():
        class_dir = Path(DATA_DIR) / class_name
        if class_dir.exists():
            for img_path in class_dir.glob('*.jpg'):
                all_samples.append((str(img_path), class_id))
            print(f'{class_name}: {len(list(class_dir.glob("*.jpg")))} images')
        else:
            print(f' Warning: {class_dir} not found!')
    
    if len(all_samples) == 0:
        print(' Error: No images found! Check DATA_DIR path.')
        return
    
    print(f'\nTotal images: {len(all_samples)}')
    
    print('\n' + '='*50)
    print('Splitting data (70% train, 15% val, 15% test)...')
    print('='*50)
    
    labels = [sample[1] for sample in all_samples]
    
    train_samples, temp_samples = train_test_split(
        all_samples, 
        train_size=0.70, 
        random_state=RANDOM_STATE,
        stratify=labels
    )
    
    temp_labels = [sample[1] for sample in temp_samples]
    
    val_samples, test_samples = train_test_split(
        temp_samples, 
        test_size=0.50, 
        random_state=RANDOM_STATE,
        stratify=temp_labels
    )
    
    print(f'Train samples: {len(train_samples)}')
    print(f'Val samples: {len(val_samples)}')
    print(f'Test samples: {len(test_samples)}')
    
    print('\n' + '='*50)
    print('Setting up data transforms...')
    print('='*50)
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    }
    
    train_dataset = BubbleDataset(train_samples, transform=data_transforms['train'])
    val_dataset = BubbleDataset(val_samples, transform=data_transforms['val'])
    test_dataset = BubbleDataset(test_samples, transform=data_transforms['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print('\n' + '='*50)
    print('Creating model (ResNet50 with Transfer Learning)...')
    print('='*50)
    
    model = create_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    print('\n' + '='*50)
    print('Starting training...')
    print('='*50)
    
    start_time = time.time()
    
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        num_epochs=NUM_EPOCHS,
        device=device,
        save_path='best_bubble_model.pth'
    )
    
    training_time = time.time() - start_time
    print(f'\n Total training time: {training_time/60:.1f} minutes')
    
    print('\n' + '='*50)
    print('Loading best model...')
    print('='*50)
    
    checkpoint = torch.load('best_bubble_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded best model with validation accuracy: {checkpoint["accuracy"]:.2f}%')
    
    print('\n' + '='*50)
    print('Evaluating on test set...')
    print('='*50)
    
    metrics = evaluate_model(model, test_loader, CLASS_NAMES, device)
    
    print('\n' + '='*50)
    print('Saving results...')
    print('='*50)
    
    df_results = save_results_to_csv(metrics, MODEL_NAME, 'results.csv')
    
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print('Training history saved to training_history.json')
    
    plot_training_history(history, 'training_history.png')
    plot_confusion_matrix(metrics['confusion_matrix'], CLASS_NAMES, 'confusion_matrix.png')
    
    print('\n' + '='*50)
    print('FINAL REPORT')
    print('='*50)
    print(f'Model: {MODEL_NAME}')
    print(f'Training time: {training_time/60:.1f} minutes')
    print(f'Total images: {len(all_samples)}')
    print(f'Train/Val/Test split: {len(train_samples)}/{len(val_samples)}/{len(test_samples)}')
    print(f'Test Accuracy: {metrics["accuracy"]:.4f} ({metrics["accuracy"]*100:.2f}%)')
    print(f'Test Precision: {metrics["precision"]:.4f}')
    print(f'Test Recall: {metrics["recall"]:.4f}')
    print(f'Test F1-Score: {metrics["f1_score"]:.4f}')
    
    if metrics['accuracy'] >= 0.80:
        print('\n Succes, result > 80%')
    else:
        print('\nAccuracy below 80%. Consider:')
        print('   - More training epochs')
        print('   - More data augmentation')
        print('   - Fine-tuning more layers')
        print('   - Different learning rate')
    print('\nOutput files:')
    print('   - results.csv')
    print('   - best_bubble_model.pth')
    print('   - training_history.json')
    print('   - training_history.png')
    print('   - confusion_matrix.png')
    print('='*50)
    return model, metrics

if __name__ == "__main__":
    model, metrics = main()