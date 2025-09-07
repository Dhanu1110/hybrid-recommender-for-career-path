#!/usr/bin/env python3
"""
Training Script for BERT4Rec Career Path Model

This script handles training of the BERT4Rec model for career path prediction
using masked language modeling on career sequences.
"""

import os
import sys
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.bert4rec import BERT4Rec, BERT4RecConfig, create_bert4rec_model

logger = logging.getLogger(__name__)

class CareerSequenceDataset(Dataset):
    """Dataset for career sequences with masked language modeling."""
    
    def __init__(self, 
                 sequences: List[List[int]], 
                 vocab_size: int,
                 max_seq_len: int = 50,
                 mask_prob: float = 0.15,
                 pad_token_id: int = 0,
                 mask_token_id: int = 1):
        """
        Initialize dataset.
        
        Args:
            sequences: List of career sequences (job ID lists)
            vocab_size: Size of job vocabulary
            max_seq_len: Maximum sequence length
            mask_prob: Probability of masking tokens
            pad_token_id: ID for padding token
            mask_token_id: ID for mask token
        """
        self.sequences = sequences
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        
        # Filter sequences by length
        self.sequences = [seq for seq in sequences if len(seq) > 1]
        
        logger.info(f"Dataset initialized with {len(self.sequences)} sequences")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample with masked tokens."""
        sequence = self.sequences[idx].copy()
        
        # Truncate if too long
        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
        
        # Create input and target sequences
        input_ids = sequence.copy()
        target_ids = sequence.copy()
        
        # Apply masking
        for i in range(len(input_ids)):
            if random.random() < self.mask_prob:
                # 80% of the time, replace with mask token
                if random.random() < 0.8:
                    input_ids[i] = self.mask_token_id
                # 10% of the time, replace with random token
                elif random.random() < 0.5:
                    input_ids[i] = random.randint(2, self.vocab_size - 1)
                # 10% of the time, keep original token
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'length': len(input_ids)
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]], 
               pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    
    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=pad_token_id)
    
    # Create attention mask
    attention_mask = (input_ids != pad_token_id).float()
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'attention_mask': attention_mask
    }

class BERT4RecTrainer:
    """Trainer class for BERT4Rec model."""
    
    def __init__(self, 
                 model: BERT4Rec,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: BERT4Rec model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # Optimizer and loss function
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)), 
                target_ids.view(-1)
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), 
                    target_ids.view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self, 
              num_epochs: int,
              save_dir: str = "models/checkpoints",
              save_every: int = 5) -> None:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['epoch'] = epoch + 1
            
            self.training_history.append(epoch_metrics)
            
            # Log metrics
            log_str = f"Epoch {epoch + 1}/{num_epochs}"
            for key, value in epoch_metrics.items():
                if key != 'epoch':
                    log_str += f" - {key}: {value:.4f}"
            logger.info(log_str)
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(save_dir / f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Save best model
            if val_metrics and val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(save_dir / "best_model.pt")
                logger.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint(save_dir / "final_model.pt")
        
        # Save training history
        with open(save_dir / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

def load_career_sequences(data_path: str) -> Tuple[List[List[int]], Dict[str, int]]:
    """
    Load career sequences from processed data.
    
    Args:
        data_path: Path to processed career paths data
        
    Returns:
        Tuple of (sequences, job_to_id mapping)
    """
    df = pd.read_parquet(data_path)
    
    # Build job vocabulary
    all_jobs = set()
    for job_list in df['job_sequence_list']:
        all_jobs.update(job_list)
    
    # Create job to ID mapping (reserve 0 for padding, 1 for mask)
    job_to_id = {'<PAD>': 0, '<MASK>': 1}
    for i, job in enumerate(sorted(all_jobs), start=2):
        job_to_id[job] = i
    
    # Convert sequences to ID sequences
    sequences = []
    for job_list in df['job_sequence_list']:
        sequence = [job_to_id[job] for job in job_list if job in job_to_id]
        if len(sequence) > 1:  # Only keep sequences with more than 1 job
            sequences.append(sequence)
    
    logger.info(f"Loaded {len(sequences)} career sequences")
    logger.info(f"Vocabulary size: {len(job_to_id)}")
    
    return sequences, job_to_id

def create_synthetic_data(num_sequences: int = 1000, 
                         vocab_size: int = 100,
                         min_len: int = 3,
                         max_len: int = 10) -> Tuple[List[List[int]], Dict[str, int]]:
    """Create synthetic career sequences for testing."""
    logger.info("Creating synthetic career sequences...")
    
    # Create job vocabulary
    job_to_id = {'<PAD>': 0, '<MASK>': 1}
    for i in range(2, vocab_size):
        job_to_id[f'job_{i}'] = i
    
    # Generate sequences
    sequences = []
    for _ in range(num_sequences):
        seq_len = random.randint(min_len, max_len)
        sequence = [random.randint(2, vocab_size - 1) for _ in range(seq_len)]
        sequences.append(sequence)
    
    logger.info(f"Created {len(sequences)} synthetic sequences")
    return sequences, job_to_id

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train BERT4Rec for career path prediction')
    parser.add_argument('--data-path', type=str, default='data/processed/career_paths.parquet',
                       help='Path to career paths data')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for testing')
    parser.add_argument('--config-path', type=str, default='configs/bert4rec_config.json',
                       help='Path to model configuration')
    parser.add_argument('--output-dir', type=str, default='models/bert4rec',
                       help='Output directory for model and checkpoints')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create data
    if args.synthetic:
        sequences, job_to_id = create_synthetic_data()
    else:
        if not Path(args.data_path).exists():
            logger.error(f"Data file not found: {args.data_path}")
            logger.info("Use --synthetic flag to create synthetic data for testing")
            return
        sequences, job_to_id = load_career_sequences(args.data_path)
    
    # Save vocabulary
    with open(output_dir / 'job_vocab.json', 'w') as f:
        json.dump(job_to_id, f, indent=2)
    
    # Create model config
    config = BERT4RecConfig(
        vocab_size=len(job_to_id),
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=50
    )
    
    # Save config
    with open(output_dir / 'model_config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Split data
    random.shuffle(sequences)
    split_idx = int(len(sequences) * (1 - args.val_split))
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    # Create datasets
    train_dataset = CareerSequenceDataset(train_sequences, config.vocab_size)
    val_dataset = CareerSequenceDataset(val_sequences, config.vocab_size) if val_sequences else None
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, config.pad_token_id)
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, config.pad_token_id)
        )
    
    # Create model
    model = create_bert4rec_model(config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = BERT4RecTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate
    )
    
    # Train model
    trainer.train(
        num_epochs=args.epochs,
        save_dir=output_dir / "checkpoints"
    )

if __name__ == "__main__":
    main()
