import time
import torch
import torch.nn as nn
from typing import Dict, List
from .checkpoint import CheckpointManager

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        checkpoint_manager: CheckpointManager,
        device: torch.device,
        patience: int = 10
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.device = device
        self.patience = patience
        self.epochs_without_improvement = 0

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        history = {
            'train_acc': [], 
            'val_acc': [], 
            'train_loss': [], 
            'val_loss': [], 
            'gap': []
        }
        
        print(f"Starting training for up to {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            self.model.train()
            train_loss, correct, total = 0.0, 0, 0
            
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            train_loss /= len(self.train_loader.dataset)
            train_acc = 100.0 * correct / total
            
            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                     
            val_loss /= len(self.val_loader.dataset)
            val_acc = 100.0 * val_correct / val_total
            gap = abs(train_acc - val_acc)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['gap'].append(gap)
            
            is_best = self.checkpoint_manager.check_and_save(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                val_acc=val_acc,
                gap=gap
            )
            
            if is_best:
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Gap: {gap:.2f}% | Time: {epoch_time:.1f}s | Patience: {self.epochs_without_improvement}/{self.patience}')
            
            self.scheduler.step()
            
            if self.epochs_without_improvement >= self.patience:
                print(f'Early stopping triggered at epoch {epoch+1}. No improvement in validation accuracy for {self.patience} epochs.')
                break
                
        return history