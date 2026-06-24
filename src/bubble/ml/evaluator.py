import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from typing import List, Dict, Any

def evaluate_model(
    model: torch.nn.Module, 
    test_loader: torch.utils.data.DataLoader, 
    class_names: List[str], 
    device: torch.device
) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Evaluating on test set...")
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
    print(f'Test Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall:    {recall:.4f}')
    print(f'Test F1-Score:  {f1:.4f}')
    print(f'{"="*50}\n')
    
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'report': report,
        'confusion_matrix': cm
    }