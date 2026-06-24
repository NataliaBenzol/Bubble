import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import numpy as np

class Plotter:
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], output_file: str) -> None:
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

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_file: str) -> None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f'Confusion matrix saved to {output_file}')
        plt.close()