import os
import argparse
import sys
from config import *
from data.preprocessor import DataPreprocessor
from validate_speakers import validate_speakers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="Validate speakers after preparation")
    parser.add_argument("--min_utterances", type=int, default=3, help="Minimum utterances per speaker")
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(UNIFIED_AUDIO_DIR, exist_ok=True)
    os.makedirs(os.path.join(UNIFIED_AUDIO_DIR, 'en'), exist_ok=True)
    os.makedirs(os.path.join(UNIFIED_AUDIO_DIR, 'ta'), exist_ok=True)
    
    # Process all datasets
    preprocessor = DataPreprocessor(sys.modules['config'])
    df = preprocessor.run()
    
    # Save initial metadata
    df.to_csv(METADATA_PATH, index=False)
    print(f"\n✅ Initial metadata saved to: {METADATA_PATH}")
    
    # Validate speakers if requested
    if args.validate:
        validated_df = validate_speakers(METADATA_PATH, args.min_utterances)
        validated_path = os.path.join(OUTPUT_DIR, "validated_metadata.csv")
        validated_df.to_csv(validated_path, index=False)
        print(f"\n✅ Validated metadata saved to: {validated_path}")
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()