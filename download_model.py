import os
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

def download_hubert_model(model_name="facebook/hubert-base-ls960", cache_dir="./models"):
    """
    Download HuBERT model and feature extractor
    """
    print(f"📥 Downloading {model_name}...")
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Download model only (no tokenizer needed)
        model = HubertModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        print(f"✅ Model downloaded successfully!")
        print(f"   Model config: {model.config}")
        
        # Download feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        print(f"✅ Feature extractor downloaded successfully!")
        
        # Save locally
        local_model_path = os.path.join(cache_dir, "hubert-base-ls960")
        model.save_pretrained(local_model_path)
        feature_extractor.save_pretrained(local_model_path)
        print(f"✅ Model saved to: {local_model_path}")
        
        return model, feature_extractor
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None, None

def test_model():
    """Test if model works with a dummy input"""
    print("\n🧪 Testing model...")
    
    # Load model
    model_path = "./models/hubert-base-ls960"
    if not os.path.exists(model_path):
        print("Model not found locally. Please run download first.")
        return
    
    model = HubertModel.from_pretrained(model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    
    # Create dummy audio (1 second of silence)
    dummy_audio = torch.zeros(1, 16000)  # 1 second at 16kHz
    
    # Preprocess
    inputs = feature_extractor(
        dummy_audio.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs.input_values, output_hidden_states=True)
    
    print(f"✅ Model works! Output shape: {outputs.last_hidden_state.shape}")
    print(f"   Number of hidden layers: {len(outputs.hidden_states)}")
    print(f"   Hidden size: {model.config.hidden_size}")
    
    return model

if __name__ == "__main__":
    print("=" * 60)
    print("HuBERT Model Downloader")
    print("=" * 60)
    
    # Download model
    model, processor = download_hubert_model()
    
    # Test if successful
    if model is not None:
        test_model()
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Update config.SSL_MODEL_NAME to './models/hubert-base-ls960'")
    print("2. Run: python train.py")
    print("=" * 60)