#!/usr/bin/env python3
"""
DAGMM 모델 학습 스크립트 - H100 GPU 최적화
"""

import os
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Any
import logging
import argparse
import numpy as np
from tqdm import tqdm

from model_dagmm import DAGMM, DAGMMLoss, create_dagmm_model
from dataset import load_json_files, prepare_dagmm_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DAGMMTrainer:
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
        self.best_loss = float('inf')
    
    def setup_model(self, num_classes: int):
        model_config = self.config.get('model', {})
        model_config['num_classes'] = num_classes
        model_config['window_size'] = self.config.get('data', {}).get('window_size', 10)
        
        self.model = create_dagmm_model(model_config).to(self.device)
        if self.num_gpus > 1:
            self.model = nn.DataParallel(self.model)
        
        self.criterion = DAGMMLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=float(self.config.get('training', {}).get('learning_rate', 0.001))
        )
        logger.info(f"Params: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, dataloader, epoch: int):
        self.model.train()
        total_loss, num_batches = 0.0, 0
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        for batch_idx, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            data = data[0].to(self.device) if isinstance(data, (list, tuple)) else data.to(self.device)
            z, x_hat, z_c, gamma, x_flat = self.model(data)
            phi, mu, sigma = model.compute_gmm_params(z_c, gamma)
            
            with torch.no_grad():
                m = 0.9 if batch_idx > 0 or epoch > 1 else 0
                model.phi = m * model.phi + (1 - m) * phi
                model.mu = m * model.mu + (1 - m) * mu
                model.sigma = m * model.sigma + (1 - m) * sigma
            
            energy = model.compute_energy(z_c, phi, mu, sigma)
            loss, _, _, _ = self.criterion(x_flat, x_hat, energy, sigma)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, name: str):
        model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        torch.save({'model': model_state, 'config': self.config}, self.output_dir / f'{name}.pt')
    
    def train(self, train_loader, test_loader=None):
        num_epochs = self.config['training']['num_epochs']
        for epoch in range(1, num_epochs + 1):
            loss = self.train_epoch(train_loader, epoch)
            logger.info(f"Epoch {epoch}/{num_epochs} - Loss: {loss:.4f}")
            if loss < self.best_loss:
                self.best_loss = loss
                self.save_checkpoint('best_model')
        self.save_checkpoint('final_model')
        logger.info(f"Training complete. Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../preprocessing/output')
    parser.add_argument('--output-dir', type=str, default='checkpoints/dagmm')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    
    config = {
        'model': {'embedding_dim': 32, 'hidden_dims': [64, 32], 'latent_dim': 2, 'n_gmm': 4},
        'training': {'batch_size': args.batch_size, 'learning_rate': 0.001, 'num_epochs': args.epochs},
        'data': {'window_size': 10, 'preprocessed_dir': args.data_dir},
        'output_dir': args.output_dir,
    }
    
    data_dir = Path(args.data_dir)
    data_files = [str(f) for f in sorted(data_dir.glob("preprocessed_logs_*.json"))]
    sessions = load_json_files(data_files)
    
    if not sessions:
        logger.error("No data loaded!")
        return
    
    samples, num_classes, event_id_map = prepare_dagmm_data(sessions, 10)
    
    idx = np.random.permutation(len(samples))
    split = int(len(samples) * 0.2)
    train_samples = [samples[i] for i in idx[split:]]
    test_samples = [samples[i] for i in idx[:split]]
    
    train_loader = DataLoader(TensorDataset(torch.LongTensor(train_samples)), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(TensorDataset(torch.LongTensor(test_samples)), batch_size=args.batch_size, num_workers=4, pin_memory=True)
    
    trainer = DAGMMTrainer(config)
    trainer.setup_model(num_classes)
    trainer.train(train_loader, test_loader)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / 'event_id_mapping.json', 'w') as f:
        json.dump({str(k): v for k, v in event_id_map.items()}, f, indent=2)


if __name__ == '__main__':
    main()
