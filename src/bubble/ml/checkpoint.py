import torch
from dataclasses import dataclass
from typing import Protocol
from pathlib import Path

@dataclass
class CheckpointState:
    epoch: int
    model_state_dict: dict
    optimizer_state_dict: dict
    val_acc: float
    gap: float

class CheckpointSelector(Protocol):
    def should_save(self, current_val_acc: float, best_val_acc: float) -> bool:
        ...

class BestValAccuracySelector(CheckpointSelector):
    def should_save(self, current_val_acc: float, best_val_acc: float) -> bool:
        return current_val_acc > best_val_acc

class CheckpointManager:
    def __init__(self, save_path: str, selector: CheckpointSelector):
        self.save_path = Path(save_path)
        self.selector = selector
        self.best_val_acc = -1.0

    def check_and_save(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                       val_acc: float, gap: float) -> bool:
        if self.selector.should_save(val_acc, self.best_val_acc):
            self.best_val_acc = val_acc
            state = CheckpointState(
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                val_acc=val_acc,
                gap=gap
            )
            torch.save(state.__dict__, self.save_path)
            print(f'Saved best model at epoch {epoch+1}: Val Acc={val_acc:.2f}%')
            return True
        return False

    def load_best(self, model: torch.nn.Module) -> dict:
        checkpoint = torch.load(self.save_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint