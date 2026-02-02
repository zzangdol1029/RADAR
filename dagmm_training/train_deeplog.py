#!/usr/bin/env python3
"""
DeepLog 모델 학습 스크립트 - H100 GPU 최적화
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Any
import logging
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from model_deeplog import DeepLog, create_deeplog_model
from dataset import load_json_files, prepare_deeplog_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepLogTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.num_gpus = torch.cuda.device_count()
        self.output_dir = Path(config.get('output_dir', 'checkpoints'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.best_acc = 0.0
    
    def setup_model(self, num_classes: int):
        model_config = self.config.get('model', {})
        model_config['num_classes'] = num_classes
        
        self.model = create_deeplog_model(model_config).to(self.device)
        if self.num_gpus > 1:
            logger.info(f"Using {self.num_gpus} GPUs")
            self.model = nn.DataParallel(self.model)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.get('training', {}).get('learning_rate', 0.001))
        )
        logger.info(f"Total params: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, dataloader, epoch: int):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            sequences = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            
            outputs = self.model(sequences)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss = self.criterion(logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for batch in dataloader:
                sequences = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                outputs = self.model(sequences)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def save_checkpoint(self, name: str, acc: float = 0.0):
        model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        torch.save({
            'model': model_state,
            'config': self.config,
            'accuracy': acc
        }, self.output_dir / f'{name}.pt')
        logger.info(f"Saved: {name}.pt")
    
    def train(self, train_loader, test_loader):
        num_epochs = self.config['training']['num_epochs']
        logger.info("=" * 60)
        logger.info("DeepLog Training Started")
        logger.info("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            test_loss, test_acc = self.evaluate(test_loader)
            
            logger.info(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%"
            )
            
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.save_checkpoint('best_deeplog_model', test_acc)
        
        self.save_checkpoint('final_deeplog_model', self.best_acc)
        logger.info(f"Training complete! Best accuracy: {self.best_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='DeepLog Training')
    parser.add_argument('--data-dir', type=str, default='../preprocessing/output')
    parser.add_argument('--output-dir', type=str, default='checkpoints/deeplog')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--window-size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    config = {
        'model': {
            'embedding_dim': 32,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.3,
        },
        'training': {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'num_epochs': args.epochs,
        },
        'data': {
            'window_size': args.window_size,
            'preprocessed_dir': args.data_dir,
        },
        'output_dir': args.output_dir,
    }
    
    # Load data
    logger.info("Loading data...")
    data_dir = Path(args.data_dir)
    data_files = [str(f) for f in sorted(data_dir.glob("preprocessed_logs_*.json"))]
    
    if not data_files:
        logger.error(f"No data files found in {data_dir}")
        return
    
    sessions = load_json_files(data_files)
    
    if not sessions:
        logger.error("No sessions loaded!")
        return
    
    # Prepare data
    sequences, labels, num_classes, event_id_map = prepare_deeplog_data(
        sessions,
        window_size=args.window_size,
        stride=1,
        min_seq_length=3
    )
    
    if len(sequences) == 0:
        logger.error("No samples after preprocessing! Try smaller window_size")
        return
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, shuffle=True
    )
    
    logger.info(f"Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup and train
    trainer = DeepLogTrainer(config)
    trainer.setup_model(num_classes)
    trainer.train(train_loader, test_loader)
    
    # Save event mapping
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / 'event_id_mapping.json', 'w') as f:
        json.dump({str(k): v for k, v in event_id_map.items()}, f, indent=2)
    logger.info(f"Event mapping saved to {args.output_dir}/event_id_mapping.json")


if __name__ == '__main__':
    main()
