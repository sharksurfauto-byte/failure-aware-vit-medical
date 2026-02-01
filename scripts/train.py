"""
Training script for CNN and ViT baselines.

Training configuration (identical for both):
- Optimizer: AdamW
- Learning rate: 3e-4
- Weight decay: 0.01
- Batch size: 64
- Epochs: 20 (max)
- Early stopping: 5 epochs patience
- LR scheduler: CosineAnnealingLR
"""

import sys
import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project root (path-agnostic)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MalariaDataset, load_normalization_stats
from src.models.cnn_baseline import CNNBaseline, count_parameters
from src.models.vit_baseline import ViTBaseline


def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN or ViT baseline')
    parser.add_argument('--model', type=str, choices=['cnn', 'vit'], required=True,
                        help='Model type: cnn or vit')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Maximum training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (default: 5)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    return parser.parse_args()


def get_device(device_arg):
    """Determine device to use"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU")
    else:
        device = torch.device(device_arg)
    return device


def create_model(model_type, device):
    """Create model based on type"""
    if model_type == 'cnn':
        model = CNNBaseline(num_classes=2, dropout=0.3)
    else:
        model = ViTBaseline(
            img_size=224,
            patch_size=16,
            num_classes=2,
            embed_dim=384,
            depth=6,
            num_heads=6,
            mlp_ratio=4,
            dropout=0.1
        )
    
    model = model.to(device)
    print(f"Model: {model_type.upper()}")
    print(f"Parameters: {count_parameters(model):,}")
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels, _) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train(args):
    """Main training loop"""
    print("=" * 60)
    print("TRAINING BASELINE MODEL")
    print("=" * 60)
    
    # Device
    device = get_device(args.device)
    
    # Load normalization stats
    stats_path = PROJECT_ROOT / 'data' / 'normalization_stats.json'
    load_normalization_stats(str(stats_path))
    
    # Create datasets
    print("\nLoading datasets...")
    splits_path = PROJECT_ROOT / 'data' / 'splits.json'
    train_dataset = MalariaDataset('train', splits_path=str(splits_path))
    val_dataset = MalariaDataset('val', splits_path=str(splits_path))
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,  # Parallel data loading
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(args.model, device)
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Checkpointing
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Epoch time
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / f'{args.model}_best.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  -> New best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val accuracy: {best_val_acc:.2f}%")
    
    # Save history
    history_path = checkpoint_dir / f'{args.model}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    return best_val_acc


if __name__ == "__main__":
    args = parse_args()
    train(args)
