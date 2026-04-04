import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
from model.encoder import MultilingualSpeakerEncoder
from train.dataloader import SpeakerBalancedDataset

def compute_eer(scores, labels):
    """Compute Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer, eer_threshold

def evaluate(model, test_df, device):
    """Evaluate speaker verification performance"""
    model.eval()
    
    # Create dataset
    dataset = SpeakerBalancedDataset(test_df, config, augmentations=None)
    
    # Extract all embeddings
    embeddings = []
    speaker_ids = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Extracting embeddings"):
            item = dataset[idx]
            audio = item['audio'].unsqueeze(0).to(device)
            embedding = model(audio).cpu().numpy()
            embeddings.append(embedding)
            speaker_ids.append(item['speaker_id'])
    
    embeddings = np.vstack(embeddings)
    speaker_ids = np.array(speaker_ids)
    
    # Create positive and negative pairs
    unique_speakers = np.unique(speaker_ids)
    scores = []
    labels = []
    
    print("\nCreating evaluation pairs...")
    for i in tqdm(range(len(unique_speakers))):
        spk = unique_speakers[i]
        spk_indices = np.where(speaker_ids == spk)[0]
        
        # Create positive pairs (same speaker)
        for j in range(len(spk_indices)):
            for k in range(j+1, len(spk_indices)):
                sim = np.dot(embeddings[spk_indices[j]], embeddings[spk_indices[k]])
                scores.append(sim)
                labels.append(1)
        
        # Create negative pairs (different speakers)
        for j in spk_indices:
            for k in np.where(speaker_ids != spk)[0][:min(10, len(spk_indices))]:
                sim = np.dot(embeddings[j], embeddings[k])
                scores.append(sim)
                labels.append(0)
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Compute metrics
    eer, threshold = compute_eer(scores, labels)
    auc = roc_auc_score(labels, scores)
    
    print(f"\n📊 Evaluation Results:")
    print(f"   EER: {eer:.4f} ({eer*100:.2f}%)")
    print(f"   AUC: {auc:.4f}")
    print(f"   Optimal threshold: {threshold:.4f}")
    
    return {'eer': eer, 'auc': auc, 'threshold': threshold}

if __name__ == "__main__":
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultilingualSpeakerEncoder(config)
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_speaker_encoder.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("Model loaded successfully")
    
    # Load test data
    metadata_path = os.path.join(OUTPUT_DIR, "validated_metadata.csv")
    df = pd.read_csv(metadata_path)
    
    # Use held-out test speakers
    speakers = df['speaker_id'].unique()
    test_speakers = speakers[int(0.9 * len(speakers)):]
    test_df = df[df['speaker_id'].isin(test_speakers)]
    
    # Evaluate
    results = evaluate(model, test_df, device)