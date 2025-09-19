#!/usr/bin/env python3
"""
Enhanced Training Pipeline for HaWoR
Complete training system for RGB to hand mesh models with comprehensive evaluation
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import argparse
from pathlib import Path
import json
import yaml
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Add HaWoR to path
sys.path.append(str(Path(__file__).parent))

from lib.models.hawor import HAWOR
from lib.utils.geometry import rot6d_to_rotmat, perspective_projection
from enhanced_training_evaluation import EnhancedTrainingLoss, TrainingEvaluator, TrainingMetrics
from training_data_preparation import TrainingDataset

class EnhancedHaWoRTrainer(pl.LightningModule):
    """Enhanced HaWoR trainer with comprehensive evaluation"""
    
    def __init__(self, 
                 config: Dict,
                 training_data_dir: str,
                 validation_data_dir: Optional[str] = None,
                 use_enhanced_loss: bool = True,
                 use_adaptive_weights: bool = True):
        """
        Initialize enhanced HaWoR trainer
        
        Args:
            config: Training configuration
            training_data_dir: Directory containing training data
            validation_data_dir: Directory containing validation data
            use_enhanced_loss: Whether to use enhanced loss function
            use_adaptive_weights: Whether to use adaptive loss weighting
        """
        super().__init__()
        
        self.config = config
        self.training_data_dir = training_data_dir
        self.validation_data_dir = validation_data_dir or training_data_dir
        self.use_enhanced_loss = use_enhanced_loss
        self.use_adaptive_weights = use_adaptive_weights
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize loss function
        if use_enhanced_loss:
            self.loss_function = EnhancedTrainingLoss(
                loss_weights=config.get('loss_weights', None),
                use_adaptive_weights=use_adaptive_weights
            )
        else:
            self.loss_function = self._create_standard_loss()
        
        # Initialize evaluator
        self.evaluator = TrainingEvaluator(
            model=self.model,
            device=self.device,
            use_wandb=config.get('use_wandb', False),
            use_tensorboard=config.get('use_tensorboard', True),
            log_dir=config.get('log_dir', './training_logs')
        )
        
        # Training metrics
        self.training_metrics = []
        self.validation_metrics = []
        
        # Setup logging
        self.logger_instance = logging.getLogger(__name__)
    
    def _create_model(self) -> HAWOR:
        """Create HaWoR model from config"""
        
        # Convert config to CfgNode format expected by HaWoR
        from yacs.config import CfgNode
        cfg = CfgNode()
        
        # Model configuration
        cfg.MODEL = CfgNode()
        cfg.MODEL.IMAGE_SIZE = self.config.get('image_size', 256)
        cfg.MODEL.BACKBONE = CfgNode()
        cfg.MODEL.BACKBONE.TYPE = self.config.get('backbone_type', 'vit')
        cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = self.config.get('pretrained_weights', None)
        cfg.MODEL.BACKBONE.TORCH_COMPILE = self.config.get('torch_compile', 0)
        
        # MANO configuration
        cfg.MANO = CfgNode()
        cfg.MANO.DATA_DIR = self.config.get('mano_data_dir', '_DATA/data/')
        cfg.MANO.MODEL_PATH = self.config.get('mano_model_path', '_DATA/data/mano')
        cfg.MANO.GENDER = self.config.get('mano_gender', 'neutral')
        cfg.MANO.NUM_HAND_JOINTS = self.config.get('num_hand_joints', 15)
        cfg.MANO.MEAN_PARAMS = self.config.get('mano_mean_params', '_DATA/data/mano_mean_params.npz')
        cfg.MANO.CREATE_BODY_POSE = self.config.get('create_body_pose', False)
        
        # Loss weights
        cfg.LOSS_WEIGHTS = CfgNode()
        loss_weights = self.config.get('loss_weights', {})
        for key, value in loss_weights.items():
            cfg.LOSS_WEIGHTS[key] = value
        
        # Training configuration
        cfg.TRAIN = CfgNode()
        cfg.TRAIN.LR = self.config.get('learning_rate', 1e-5)
        cfg.TRAIN.WEIGHT_DECAY = self.config.get('weight_decay', 1e-4)
        cfg.TRAIN.GRAD_CLIP_VAL = self.config.get('grad_clip_val', 0)
        cfg.TRAIN.RENDER_LOG = self.config.get('render_log', True)
        
        # General configuration
        cfg.GENERAL = CfgNode()
        cfg.GENERAL.LOG_STEPS = self.config.get('log_steps', 1000)
        
        # Create model
        model = HAWOR(cfg)
        
        return model
    
    def _create_standard_loss(self):
        """Create standard loss function"""
        
        class StandardLoss(nn.Module):
            def __init__(self, loss_weights):
                super().__init__()
                self.loss_weights = loss_weights
                self.mse_loss = nn.MSELoss()
                self.l1_loss = nn.L1Loss()
            
            def forward(self, pred_output, gt_data, valid_mask=None, occlusion_mask=None):
                losses = {}
                total_loss = torch.tensor(0.0, device=next(iter(pred_output.values())).device)
                
                # 3D Keypoint Loss
                if 'pred_keypoints_3d' in pred_output and 'gt_keypoints_3d' in gt_data:
                    losses['keypoints_3d'] = self.mse_loss(
                        pred_output['pred_keypoints_3d'],
                        gt_data['gt_keypoints_3d']
                    )
                    total_loss += self.loss_weights.get('KEYPOINTS_3D', 0.05) * losses['keypoints_3d']
                
                # 2D Keypoint Loss
                if 'pred_keypoints_2d' in pred_output and 'gt_keypoints_2d' in gt_data:
                    losses['keypoints_2d'] = self.mse_loss(
                        pred_output['pred_keypoints_2d'],
                        gt_data['gt_keypoints_2d']
                    )
                    total_loss += self.loss_weights.get('KEYPOINTS_2D', 0.01) * losses['keypoints_2d']
                
                losses['total_loss'] = total_loss
                return total_loss, losses
        
        return StandardLoss(self.config.get('loss_weights', {}))
    
    def forward(self, batch):
        """Forward pass"""
        return self.model.forward_step(batch, train=self.training)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        
        # Forward pass
        output = self.forward(batch)
        
        # Compute loss
        if self.use_enhanced_loss:
            total_loss, loss_components = self.loss_function(
                output, batch, 
                valid_mask=batch.get('valid_mask'),
                occlusion_mask=batch.get('occlusion_mask')
            )
        else:
            total_loss, loss_components = self.loss_function(output, batch)
        
        # Evaluate batch
        metrics = self.evaluator.evaluate_batch(
            batch=batch,
            output=output,
            loss=total_loss,
            loss_components=loss_components,
            step=self.global_step,
            epoch=self.current_epoch,
            is_training=True
        )
        
        # Store metrics
        self.training_metrics.append(metrics)
        
        # Log metrics
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/mpjpe_3d', metrics.mpjpe_3d, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/mpjpe_2d', metrics.mpjpe_2d, on_step=True, on_epoch=True)
        self.log('train/pck_3d_15mm', metrics.pck_3d_15mm, on_step=True, on_epoch=True)
        
        # Log individual loss components
        for loss_name, loss_value in loss_components.items():
            if isinstance(loss_value, torch.Tensor):
                self.log(f'train/loss_{loss_name}', loss_value, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        
        # Forward pass
        output = self.forward(batch)
        
        # Compute loss
        if self.use_enhanced_loss:
            total_loss, loss_components = self.loss_function(
                output, batch,
                valid_mask=batch.get('valid_mask'),
                occlusion_mask=batch.get('occlusion_mask')
            )
        else:
            total_loss, loss_components = self.loss_function(output, batch)
        
        # Evaluate batch
        metrics = self.evaluator.evaluate_batch(
            batch=batch,
            output=output,
            loss=total_loss,
            loss_components=loss_components,
            step=self.global_step,
            epoch=self.current_epoch,
            is_training=False
        )
        
        # Store metrics
        self.validation_metrics.append(metrics)
        
        # Log metrics
        self.log('val/loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mpjpe_3d', metrics.mpjpe_3d, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mpjpe_2d', metrics.mpjpe_2d, on_step=False, on_epoch=True)
        self.log('val/pck_3d_15mm', metrics.pck_3d_15mm, on_step=False, on_epoch=True)
        
        return total_loss
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        
        if not self.validation_metrics:
            return
        
        # Compute epoch averages
        avg_metrics = self._compute_epoch_averages(self.validation_metrics)
        
        # Log epoch metrics
        for metric_name, metric_value in avg_metrics.items():
            if isinstance(metric_value, (int, float)):
                self.log(f'val_epoch/{metric_name}', metric_value, on_epoch=True)
        
        # Clear validation metrics
        self.validation_metrics.clear()
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        
        if not self.training_metrics:
            return
        
        # Compute epoch averages
        avg_metrics = self._compute_epoch_averages(self.training_metrics)
        
        # Log epoch metrics
        for metric_name, metric_value in avg_metrics.items():
            if isinstance(metric_value, (int, float)):
                self.log(f'train_epoch/{metric_name}', metric_value, on_epoch=True)
        
        # Clear training metrics
        self.training_metrics.clear()
    
    def _compute_epoch_averages(self, metrics_list: List[TrainingMetrics]) -> Dict:
        """Compute average metrics for an epoch"""
        
        if not metrics_list:
            return {}
        
        # Convert to dictionaries
        metrics_dicts = [m.to_dict() for m in metrics_list]
        
        # Compute averages
        avg_metrics = {}
        for key in metrics_dicts[0].keys():
            values = [d[key] for d in metrics_dicts if isinstance(d[key], (int, float))]
            if values:
                avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def configure_optimizers(self):
        """Configure optimizers"""
        
        # Get model parameters
        params = self.model.get_parameters()
        
        # Create optimizer
        optimizer = optim.AdamW(
            params,
            lr=self.config.get('learning_rate', 1e-5),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        # Create scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.get('max_epochs', 100),
            eta_min=self.config.get('min_lr', 1e-7)
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def train_dataloader(self):
        """Training dataloader"""
        
        # Create dataset
        dataset = TrainingDataset(
            data_dir=self.training_data_dir,
            split='train',
            target_resolution=(self.config.get('image_size', 256), self.config.get('image_size', 256))
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        return dataloader
    
    def val_dataloader(self):
        """Validation dataloader"""
        
        # Create dataset
        dataset = TrainingDataset(
            data_dir=self.validation_data_dir,
            split='val',
            target_resolution=(self.config.get('image_size', 256), self.config.get('image_size', 256))
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=False
        )
        
        return dataloader

class TrainingPipeline:
    """Complete training pipeline for HaWoR"""
    
    def __init__(self, config_path: str):
        """
        Initialize training pipeline
        
        Args:
            config_path: Path to training configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir = Path(self.config.get('output_dir', './training_output'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup loggers
        self.loggers = self._setup_loggers()
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks()
    
    def _load_config(self) -> Dict:
        """Load training configuration"""
        
        if self.config_path.suffix == '.yaml' or self.config_path.suffix == '.yml':
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif self.config_path.suffix == '.json':
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {self.config_path.suffix}")
        
        return config
    
    def _setup_loggers(self) -> List:
        """Setup logging backends"""
        
        loggers = []
        
        # TensorBoard logger
        if self.config.get('use_tensorboard', True):
            tb_logger = TensorBoardLogger(
                save_dir=self.output_dir / 'logs',
                name='hawor_training',
                version=datetime.now().strftime('%Y%m%d_%H%M%S')
            )
            loggers.append(tb_logger)
        
        # Weights & Biases logger
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb_logger = WandbLogger(
                project=self.config.get('wandb_project', 'hawor-training'),
                name=self.config.get('experiment_name', f'hawor_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                save_dir=self.output_dir / 'logs'
            )
            loggers.append(wandb_logger)
        
        return loggers
    
    def _setup_callbacks(self) -> List:
        """Setup training callbacks"""
        
        callbacks = []
        
        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir / 'checkpoints',
            filename='hawor-{epoch:02d}-{val_loss:.4f}',
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val/loss',
            mode='min',
            patience=self.config.get('early_stopping_patience', 10),
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
        
        return callbacks
    
    def train(self):
        """Run training"""
        
        self.logger.info("Starting HaWoR training...")
        self.logger.info(f"Configuration: {self.config}")
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.config.get('max_epochs', 100),
            devices=self.config.get('devices', 1),
            accelerator=self.config.get('accelerator', 'auto'),
            precision=self.config.get('precision', 16),
            gradient_clip_val=self.config.get('grad_clip_val', 0),
            log_every_n_steps=self.config.get('log_every_n_steps', 50),
            val_check_interval=self.config.get('val_check_interval', 1.0),
            callbacks=self.callbacks,
            logger=self.loggers,
            default_root_dir=self.output_dir
        )
        
        # Create model
        model = EnhancedHaWoRTrainer(
            config=self.config,
            training_data_dir=self.config.get('training_data_dir'),
            validation_data_dir=self.config.get('validation_data_dir'),
            use_enhanced_loss=self.config.get('use_enhanced_loss', True),
            use_adaptive_weights=self.config.get('use_adaptive_weights', True)
        )
        
        # Start training
        trainer.fit(model)
        
        self.logger.info("Training completed!")
        
        # Generate training report
        model.evaluator.generate_training_report(
            output_dir=self.output_dir / 'reports'
        )
        
        return trainer, model

def create_default_config(output_path: str):
    """Create default training configuration"""
    
    config = {
        # Data configuration
        'training_data_dir': './training_data',
        'validation_data_dir': './validation_data',
        'image_size': 256,
        
        # Model configuration
        'backbone_type': 'vit',
        'pretrained_weights': None,
        'torch_compile': 0,
        'mano_data_dir': '_DATA/data/',
        'mano_model_path': '_DATA/data/mano',
        'mano_gender': 'neutral',
        'num_hand_joints': 15,
        'mano_mean_params': '_DATA/data/mano_mean_params.npz',
        'create_body_pose': False,
        
        # Training configuration
        'max_epochs': 100,
        'batch_size': 8,
        'learning_rate': 1e-5,
        'weight_decay': 1e-4,
        'min_lr': 1e-7,
        'grad_clip_val': 1.0,
        'early_stopping_patience': 10,
        'num_workers': 4,
        
        # Loss configuration
        'use_enhanced_loss': True,
        'use_adaptive_weights': True,
        'loss_weights': {
            'KEYPOINTS_3D': 0.1,
            'KEYPOINTS_2D': 0.05,
            'GLOBAL_ORIENT': 0.01,
            'HAND_POSE': 0.01,
            'BETAS': 0.005,
            'MESH_VERTICES': 0.02,
            'MESH_FACES': 0.01,
            'TEMPORAL_CONSISTENCY': 0.005,
            'OCCLUSION_ROBUSTNESS': 0.01
        },
        
        # Logging configuration
        'use_tensorboard': True,
        'use_wandb': False,
        'wandb_project': 'hawor-training',
        'experiment_name': 'hawor_experiment',
        'log_every_n_steps': 50,
        'val_check_interval': 1.0,
        'render_log': True,
        
        # Hardware configuration
        'devices': 1,
        'accelerator': 'auto',
        'precision': 16,
        
        # Output configuration
        'output_dir': './training_output',
        'log_dir': './training_logs'
    }
    
    # Save configuration
    output_path = Path(output_path)
    if output_path.suffix == '.yaml' or output_path.suffix == '.yml':
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    elif output_path.suffix == '.json':
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    print(f"Default configuration saved to {output_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced HaWoR Training Pipeline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration file')
    parser.add_argument('--create-config', type=str,
                       help='Create default configuration file at specified path')
    parser.add_argument('--resume', type=str,
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config(args.create_config)
        return
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(args.config)
    
    # Start training
    trainer, model = pipeline.train()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
