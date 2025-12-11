"""
Fine-tuning ResNet-50 on CUB-200-2011 for Bird Species Classification

This script implements the training procedure described in the paper:
- ResNet-50 backbone with 2048→200 classifier head
- AdamW optimizer with weight decay
- Gradually reduced learning rate using cosine annealing
- Image standardization and modern stochastic transforms
- Softmax scores output over 200 bird species
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import pandas as pd

# Import helpers
from helpers import (
    create_dataloaders,
    create_resnet50_model,
    train_one_epoch,
    evaluate_model,
    plot_training_curves
)


def train_model(model, train_loader, test_loader, optimizer, scheduler,
                criterion, device, num_epochs, save_dir):
    """
    Complete training loop with learning rate scheduling

    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to train on
        num_epochs: Number of training epochs
        save_dir: Directory to save checkpoints
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': []
    }

    best_acc = 0.0

    print(f"\nStarting training for {num_epochs} epochs on {device}")
    print("=" * 70)

    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Evaluate
        test_loss, test_acc = evaluate_model(
            model, test_loader, criterion, device, epoch
        )

        # Update learning rate
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc*100:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_acc': train_acc,
                'test_acc': test_acc,
                'train_loss': train_loss,
                'test_loss': test_loss,
            }
            save_path = save_dir / 'best_model.pth'
            torch.save(checkpoint, save_path)
            print(f"  ✓ Saved new best model (test_acc: {test_acc*100:.2f}%)")

        print("=" * 70)

    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_acc*100:.2f}%")

    # Save final model
    final_checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
    }
    save_path = save_dir / 'final_model.pth'
    torch.save(final_checkpoint, save_path)
    print(f"Saved final model to {save_path}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_csv_path = save_dir / 'training_history.csv'
    history_df.to_csv(history_csv_path, index=False)
    print(f"Saved training history to {history_csv_path}")

    # Plot and save training curves to results/
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history_df, str(results_dir))

    return history


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune ResNet-50 on CUB-200-2011'
    )

    # Data parameters
    parser.add_argument('--data-dir', type=str, default='./CUB_200_2011',
                        help='Path to CUB-200-2011 dataset directory')
    parser.add_argument('--use-bbox', action='store_true',
                        help='Crop images to bounding boxes')
    parser.add_argument('--image-size', type=int, default=448,
                        help='Image size for training (default: 448)')

    # Training parameters (as specified in paper)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate (default: 5e-4)')
    parser.add_argument('--lr-min', type=float, default=5e-5,
                        help='Minimum learning rate for scheduler (default: 5e-5)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for AdamW (default: 1e-4)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    # Model parameters
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet pretrained weights')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone and only train classifier head')

    # Output parameters
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========================================================================
    # Data Preparation
    # ========================================================================

    print("\n" + "=" * 70)
    print("Data Preparation")
    print("=" * 70)

    # ImageNet normalization (ResNet-50 was pretrained on ImageNet)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Training transforms with TrivialAugmentWide (as specified in paper)
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # Create data loaders using helper function
    print(f"\nLoading CUB-200-2011 from: {args.data_dir}")
    train_loader, test_loader = create_dataloaders(
        root_dir=args.data_dir,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=args.batch_size,
        use_bbox=args.use_bbox,
        train_loader_kwargs={'num_workers': args.num_workers, 'pin_memory': True if device.type == 'cuda' else False},
        test_loader_kwargs={'num_workers': args.num_workers, 'pin_memory': True if device.type == 'cuda' else False}
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {len(train_loader.dataset.class_names)}")
    print(f"Using bounding boxes: {args.use_bbox}")

    # ========================================================================
    # Model Setup
    # ========================================================================

    print("\n" + "=" * 70)
    print("Model Setup")
    print("=" * 70)

    model = create_resnet50_model(
        num_classes=200,
        pretrained=args.pretrained
    )

    # Optionally freeze backbone
    if args.freeze_backbone:
        print("\nFreezing backbone layers...")
        for name, param in model.named_parameters():
            if 'fc' not in name:  # Don't freeze the final classifier
                param.requires_grad = False
        print("Only the classifier head will be trained")
    else:
        print("\nTraining full model (all parameters)")

    model = model.to(device)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")

    # ========================================================================
    # Training Setup
    # ========================================================================

    print("\n" + "=" * 70)
    print("Training Setup")
    print("=" * 70)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Cosine annealing learning rate scheduler for gradual reduction
    # Decays from args.lr (5e-4) to args.lr_min (5e-5) as specified in paper
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr_min
    )

    print(f"\nOptimizer: AdamW")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Weight decay: {args.weight_decay}")
    print(f"\nLR Scheduler: CosineAnnealingLR")
    print(f"  - T_max: {args.num_epochs}")
    print(f"  - Min LR: {args.lr_min}")

    # ========================================================================
    # Training
    # ========================================================================

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=save_dir
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {save_dir}")
    print(f"  - best_model.pth: Best model based on test accuracy")
    print(f"  - final_model.pth: Final model after all epochs")
    print(f"  - training_history.csv: Training metrics")


if __name__ == '__main__':
    main()
