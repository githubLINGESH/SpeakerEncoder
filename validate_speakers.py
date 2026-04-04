import pandas as pd
import argparse

def validate_speakers(metadata_path: str, min_utterances: int = 3):
    """
    Validate speaker IDs and ensure each speaker has enough samples.
    Returns filtered metadata and prints validation report.
    """
    df = pd.read_csv(metadata_path)
    
    print("=" * 60)
    print("SPEAKER VALIDATION REPORT")
    print("=" * 60)
    
    # Count utterances per speaker
    speaker_counts = df['speaker_id'].value_counts()
    
    print(f"\n📊 Total speakers: {len(speaker_counts)}")
    print(f"📊 Total utterances: {len(df)}")
    print(f"\n📊 Utterances per speaker:")
    print(f"   Min: {speaker_counts.min()}")
    print(f"   Max: {speaker_counts.max()}")
    print(f"   Mean: {speaker_counts.mean():.2f}")
    print(f"   Median: {speaker_counts.median()}")
    
    # Show distribution
    bins = [0, 3, 10, 50, 100, 500, float('inf')]
    labels = ['1-2', '3-9', '10-49', '50-99', '100-499', '500+']
    df_counts = pd.cut(speaker_counts, bins=bins, labels=labels, right=False)
    dist = df_counts.value_counts().sort_index()
    
    print(f"\n📊 Distribution of utterances per speaker:")
    for label, count in dist.items():
        print(f"   {label}: {count} speakers")
    
    # Filter speakers with insufficient samples
    valid_speakers = speaker_counts[speaker_counts >= min_utterances].index.tolist()
    filtered_df = df[df['speaker_id'].isin(valid_speakers)]
    
    print(f"\n🔍 Filtering speakers with < {min_utterances} utterances:")
    print(f"   Removed speakers: {len(speaker_counts) - len(valid_speakers)}")
    print(f"   Removed utterances: {len(df) - len(filtered_df)}")
    print(f"   Remaining speakers: {len(valid_speakers)}")
    print(f"   Remaining utterances: {len(filtered_df)}")
    
    # Check Tamil speaker IDs specifically
    tamil_df = filtered_df[filtered_df['language'] == 'ta']
    tamil_speakers = tamil_df['speaker_id'].unique()
    
    print(f"\n🇮🇳 Tamil Speakers Validation:")
    print(f"   Total Tamil speakers: {len(tamil_speakers)}")
    print(f"   Total Tamil utterances: {len(tamil_df)}")
    
    # Show sample Tamil speakers
    print(f"\n   Sample Tamil speakers:")
    for spk in tamil_speakers[:10]:
        count = len(tamil_df[tamil_df['speaker_id'] == spk])
        print(f"      {spk}: {count} utterances")
    
    # Check for potential speaker ID collisions
    print(f"\n⚠️ Checking for speaker ID collisions across datasets:")
    datasets = filtered_df['dataset'].unique()
    for ds1 in datasets:
        speakers_ds1 = set(filtered_df[filtered_df['dataset'] == ds1]['speaker_id'])
        for ds2 in datasets:
            if ds1 < ds2:  # Avoid duplicates
                speakers_ds2 = set(filtered_df[filtered_df['dataset'] == ds2]['speaker_id'])
                overlap = speakers_ds1 & speakers_ds2
                if overlap:
                    print(f"   ⚠️ {ds1} and {ds2} share speakers: {overlap}")
    
    return filtered_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--min_utterances", type=int, default=3)
    parser.add_argument("--output", type=str, default="validated_metadata.csv")
    args = parser.parse_args()
    
    validated_df = validate_speakers(args.metadata, args.min_utterances)
    validated_df.to_csv(args.output, index=False)
    print(f"\n✅ Validated metadata saved to: {args.output}")