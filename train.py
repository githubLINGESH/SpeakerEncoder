import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb
import json
from datetime import datetime
from types import SimpleNamespace

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from model.encoder import MultilingualSpeakerEncoder
from model.loss import MultiLoss
from train.dataloader import create_dataloader
from data.augmentations import AudioAugmentations

# Set multiprocessing start method for Windows
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# ============ CHECKPOINT MANAGER ============
class CheckpointManager:
    """
    Manages training checkpoints with PyTorch 2.6+ compatibility
    """
    def __init__(self, checkpoint_dir, model, optimizer, scheduler, train_speakers):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_speakers = train_speakers
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, epoch, batch_idx, loss, is_best=False):
        """Save checkpoint with all training state"""
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_speakers': self.train_speakers,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save periodic checkpoint
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        # Save latest checkpoint for resume
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt'))
        
        # Save best model separately
        if is_best:
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'best_model.pt'))
        
        print(f"✅ Checkpoint saved: Epoch {epoch}, Batch {batch_idx}")
    
    def load_latest(self):
        """Load the latest checkpoint to resume training"""
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        if os.path.exists(checkpoint_path):
            try:
                # Try loading with weights_only=False (safe for trusted checkpoints)
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                print(f"✅ Resuming from Epoch {checkpoint['epoch']}, Batch {checkpoint['batch_idx']}")
                return checkpoint['epoch'], checkpoint['batch_idx']
                
            except Exception as e:
                print(f"⚠️ Could not load checkpoint: {e}")
                print("Starting from scratch...")
                return 0, 0
        return 0, 0

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set wandb to offline mode to avoid interactive prompts
    os.environ['WANDB_MODE'] = 'offline'
    
    # Filter config to only include JSON serializable values
    config_dict = {}
    for k, v in vars(sys.modules['config']).items():
        if not k.startswith('__') and not callable(v) and not isinstance(v, type):
            try:
                json.dumps(v)
                config_dict[k] = v
            except (TypeError, ValueError):
                config_dict[k] = str(v)
    
    # Create a picklable config object
    config_obj = SimpleNamespace(**config_dict)
    
    # Initialize wandb
    wandb.init(project="multilingual-speaker-encoder", config=config_dict)
    
    # Load metadata
    metadata_path = os.path.join(OUTPUT_DIR, "validated_metadata.csv")
    if not os.path.exists(metadata_path):
        metadata_path = METADATA_PATH
    
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} samples")
    
    # Filter speakers with minimum utterances
    speaker_counts = df['speaker_id'].value_counts()
    valid_speakers = speaker_counts[speaker_counts >= MIN_UTTERANCES_PER_SPEAKER].index
    df = df[df['speaker_id'].isin(valid_speakers)]
    
    print(f"After filtering: {len(df)} samples from {len(df['speaker_id'].unique())} speakers")
    
    # Check if we have enough data
    if len(df) < BATCH_SIZE:
        print(f"ERROR: Not enough data! Need at least {BATCH_SIZE} samples, have {len(df)}")
        return
    
    # Split by speakers
    speakers = df['speaker_id'].unique()
    np.random.shuffle(speakers)
    
    train_speakers = speakers[:int(0.8 * len(speakers))]
    val_speakers = speakers[int(0.8 * len(speakers)):int(0.9 * len(speakers))]
    test_speakers = speakers[int(0.9 * len(speakers)):]
    
    train_df = df[df['speaker_id'].isin(train_speakers)]
    val_df = df[df['speaker_id'].isin(val_speakers)]
    test_df = df[df['speaker_id'].isin(test_speakers)]
    
    print(f"Train: {len(train_df)} samples, {len(train_speakers)} speakers")
    print(f"Val: {len(val_df)} samples, {len(val_speakers)} speakers")
    print(f"Test: {len(test_df)} samples, {len(test_speakers)} speakers")
    
    # Create augmentations
    augmentations = AudioAugmentations(config_obj)
    
    # Create dataloaders (with num_workers=0 for Windows)
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(train_df, config_obj, augmentations)
    val_loader = create_dataloader(val_df, config_obj, augmentations=None)
    
    print("\nInitializing model...")
    model = MultilingualSpeakerEncoder(config_obj)

    # Move to device FIRST
    model = model.to(device)
    print(f"Model moved to {device}")

    # Verify all parameters are on correct device
    for name, param in model.named_parameters():
        if param.device != device:
            print(f"Warning: {name} is on {param.device}, moving to {device}")
            param.data = param.data.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Also move criterion to device
    criterion = MultiLoss(config_obj, len(train_speakers))
    criterion = criterion.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # ============ CHECKPOINT SETUP ============
    checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    checkpoint_mgr = CheckpointManager(checkpoint_dir, model, optimizer, scheduler, train_speakers)
    
    # Resume from previous checkpoint if exists
    start_epoch, start_batch = checkpoint_mgr.load_latest()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Training
        model.train()
        train_losses = {'total': 0, 'aam': 0, 'contrastive': 0, 'ge2e': 0}
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for batch_idx, batch in enumerate(train_bar):
            try:
                audio = batch['audio'].to(device)
                speaker_ids = batch['speaker_id']
                
                # Debug: Check device consistency
                if audio.device != device:
                    print(f"Audio on {audio.device}, moving to {device}")
                    audio = audio.to(device)
                
                # Forward pass
                embeddings = model(audio)
                
                # Verify embedding device
                if embeddings.device != device:
                    print(f"Embeddings on {embeddings.device}, moving to {device}")
                    embeddings = embeddings.to(device)
                
                # Convert speaker IDs to indices
                speaker_to_idx = {spk: i for i, spk in enumerate(train_speakers)}
                speaker_indices = []
                for spk in speaker_ids:
                    if spk in speaker_to_idx:
                        speaker_indices.append(speaker_to_idx[spk])
                    else:
                        speaker_indices.append(0)
                
                speaker_indices = torch.tensor(speaker_indices).to(device)
                
                # Compute loss
                losses = criterion(embeddings, speaker_indices)
                
                # Backward pass
                optimizer.zero_grad()
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Update metrics
                for key in train_losses:
                    train_losses[key] += losses[key].item()
                
                train_bar.set_postfix({'loss': losses['total'].item()})
                
                # Save checkpoint every 50 batches
                if batch_idx % 50 == 0 and batch_idx > 0:
                    checkpoint_mgr.save(epoch, batch_idx, losses['total'].item())
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = {k: v / len(train_loader) for k, v in train_losses.items()}
        
        # Validation
        model.eval()
        val_losses = {'total': 0, 'aam': 0, 'contrastive': 0, 'ge2e': 0}
        val_count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                try:
                    audio = batch['audio'].to(device)
                    speaker_ids = batch['speaker_id']
                    
                    embeddings = model(audio)
                    
                    # Filter speakers that are in training set
                    speaker_to_idx = {spk: i for i, spk in enumerate(train_speakers)}
                    valid_indices = []
                    valid_embeddings = []
                    for i, spk in enumerate(speaker_ids):
                        if spk in speaker_to_idx:
                            valid_indices.append(speaker_to_idx[spk])
                            valid_embeddings.append(embeddings[i])
                    
                    if valid_embeddings:
                        valid_embeddings = torch.stack(valid_embeddings)
                        valid_indices = torch.tensor(valid_indices).to(device)
                        losses = criterion(valid_embeddings, valid_indices)
                        
                        for key in val_losses:
                            val_losses[key] += losses[key].item()
                        val_count += 1
                        
                except Exception as e:
                    print(f"\nError in validation batch: {e}")
                    continue
        
        avg_val_loss = {k: v / max(val_count, 1) for k, v in val_losses.items()}
        
        # Update learning rate
        scheduler.step(avg_val_loss['total'])
        
        # Determine if this is the best model
        is_best = avg_val_loss['total'] < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss['total']
        
        # Save end-of-epoch checkpoint
        checkpoint_mgr.save(epoch, len(train_loader), avg_val_loss['total'], is_best)
        
        # Print summary
        print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss['total']:.4f}, Val Loss = {avg_val_loss['total']:.4f}")
        print(f"  AAM: {avg_train_loss['aam']:.4f} | Contrastive: {avg_train_loss['contrastive']:.4f} | GE2E: {avg_train_loss['ge2e']:.4f}")
        
        # Log to wandb
        wandb.log({
            'train_total_loss': avg_train_loss['total'],
            'train_aam_loss': avg_train_loss['aam'],
            'train_contrastive_loss': avg_train_loss['contrastive'],
            'train_ge2e_loss': avg_train_loss['ge2e'],
            'val_total_loss': avg_val_loss['total'],
            'val_aam_loss': avg_val_loss['aam'],
            'val_contrastive_loss': avg_val_loss['contrastive'],
            'val_ge2e_loss': avg_val_loss['ge2e'],
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch
        })
        
        # Also save best model separately (already done in checkpoint_mgr)
        if is_best:
            print(f"Saved best model with val loss: {avg_val_loss['total']:.4f}")
    
    print("\n✅ Training completed!")

if __name__ == "__main__":
    train()