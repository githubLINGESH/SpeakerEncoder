import os
import sys
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional
import shutil
import librosa
import time
from datetime import datetime
import json

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.audio_utils import convert_audio, compute_energy

class DataPreprocessor:
    def __init__(self, config):
        """
        Args:
            config: Configuration object imported from config.py
        """
        self.config = config
        self.records = []
        self.stats = {
            'start_time': None,
            'end_time': None,
            'datasets': {}
        }
    
    def _log_progress(self, message: str, level: str = "INFO"):
        """Log progress with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def _track_dataset_stats(self, dataset_name: str, processed: int, 
                              skipped: int, errors: int, duration: float):
        """Track statistics for each dataset."""
        self.stats['datasets'][dataset_name] = {
            'processed': processed,
            'skipped': skipped,
            'errors': errors,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_stats(self):
        """Save processing statistics to JSON file."""
        stats_file = os.path.join(self.config.OUTPUT_DIR, 'processing_stats.json')
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            print(f"\n📊 Statistics saved to: {stats_file}")
        except Exception as e:
            print(f"⚠️  Could not save stats: {e}")
    
    def _get_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds."""
        try:
            if not os.path.exists(audio_path):
                return 0.0
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception as e:
            return 0.0
    
    def process_sps_corpus(self) -> List[Dict]:
        """Process SPS corpus (English spontaneous speech)."""
        dataset_name = "SPS Corpus"
        self._log_progress(f"Starting {dataset_name} processing...")
        
        dataset = self.config.DATASETS['sps_corpus']
        audio_dir = os.path.join(dataset['path'], dataset['audio_dir'])
        tsv_path = os.path.join(dataset['path'], dataset['tsv'])
        
        if not os.path.exists(tsv_path):
            self._log_progress(f"TSV not found: {tsv_path}", "ERROR")
            return []
        
        try:
            df = pd.read_csv(tsv_path, sep='\t')
        except Exception as e:
            self._log_progress(f"Error reading TSV: {e}", "ERROR")
            return []
        
        records = []
        processed = 0
        skipped = 0
        errors = 0
        start_time = time.time()
        
        # Create progress bar with custom formatting
        with tqdm(total=len(df), desc=f"📁 {dataset_name}", 
                  unit="files", ncols=100, 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for idx, row in df.iterrows():
                try:
                    audio_file = row['audio_file']
                    src = os.path.join(audio_dir, audio_file)
                    
                    if not os.path.exists(src):
                        pbar.set_postfix_str(f"⚠️  Missing: {os.path.basename(src)}")
                        skipped += 1
                        pbar.update(1)
                        continue
                    
                    # Generate unique filename
                    client_id = row.get('client_id', 'unknown')
                    audio_id = row.get('audio_id', 'unknown')
                    dst_filename = f"sps_{client_id}_{audio_id}.wav"
                    dst = os.path.join(self.config.UNIFIED_AUDIO_DIR, 'en', dst_filename)
                    
                    # Convert if needed
                    if not os.path.exists(dst):
                        convert_audio(src, dst, self.config.SAMPLE_RATE)
                    
                    # Get duration
                    duration = self._get_duration(dst)
                    if duration == 0.0:
                        pbar.set_postfix_str(f"⚠️  Invalid duration: {os.path.basename(dst)}")
                        skipped += 1
                        pbar.update(1)
                        continue
                    
                    # Get gender (handle missing values)
                    gender = row.get('gender', 'U')
                    if pd.isna(gender):
                        gender = 'U'
                    
                    records.append({
                        'audio_path': dst,
                        'speaker_id': f"sps_{client_id}",
                        'language': 'en',
                        'gender': gender,
                        'dataset': 'sps_corpus',
                        'duration_sec': duration
                    })
                    
                    processed += 1
                    
                    # Update progress bar with stats
                    pbar.set_postfix_str(f"✅ {processed} | ⚠️ {skipped} | ❌ {errors}")
                    pbar.update(1)
                    
                except Exception as e:
                    errors += 1
                    pbar.set_postfix_str(f"❌ Error: {str(e)[:30]}")
                    pbar.update(1)
                    continue
        
        elapsed_time = time.time() - start_time
        self._track_dataset_stats(dataset_name, processed, skipped, errors, elapsed_time)
        
        self._log_progress(f"✅ {dataset_name} completed: {processed} files, {skipped} skipped, {errors} errors in {elapsed_time:.1f}s")
        return records
    
    def process_librispeech(self) -> List[Dict]:
        """Process LibriSpeech train-clean-100."""
        dataset_name = "LibriSpeech"
        self._log_progress(f"Starting {dataset_name} processing...")
        
        dataset = self.config.DATASETS['librispeech']
        root = os.path.join(dataset['path'], dataset['audio_dir'])
        
        if not os.path.exists(root):
            self._log_progress(f"Directory not found: {root}", "ERROR")
            return []
        
        # Read speaker info
        speaker_info = {}
        speakers_file = os.path.join(root, 'SPEAKERS.TXT')
        
        if os.path.exists(speakers_file):
            try:
                with open(speakers_file, 'r') as f:
                    for line in f:
                        if line.startswith(';') or not line.strip():
                            continue
                        parts = line.strip().split('|')
                        if len(parts) >= 2:
                            spk_id = parts[0].strip()
                            gender = parts[1].strip()
                            speaker_info[spk_id] = gender
            except Exception as e:
                self._log_progress(f"Error reading SPEAKERS.TXT: {e}", "WARNING")
        
        # Count total files first for progress bar
        total_files = 0
        for spk in os.listdir(root):
            spk_path = os.path.join(root, spk)
            if not os.path.isdir(spk_path) or not spk.isdigit():
                continue
            for chap in os.listdir(spk_path):
                chap_path = os.path.join(spk_path, chap)
                if not os.path.isdir(chap_path):
                    continue
                total_files += len([f for f in os.listdir(chap_path) if f.endswith('.flac')])
        
        records = []
        processed = 0
        skipped = 0
        errors = 0
        start_time = time.time()
        
        with tqdm(total=total_files, desc=f"📁 {dataset_name}", 
                  unit="files", ncols=100) as pbar:
            
            for spk in os.listdir(root):
                spk_path = os.path.join(root, spk)
                if not os.path.isdir(spk_path) or not spk.isdigit():
                    continue
                
                for chap in os.listdir(spk_path):
                    chap_path = os.path.join(spk_path, chap)
                    if not os.path.isdir(chap_path):
                        continue
                    
                    for fname in os.listdir(chap_path):
                        if not fname.endswith('.flac'):
                            continue
                        
                        try:
                            src = os.path.join(chap_path, fname)
                            dst_filename = f"libri_{spk}_{chap}_{fname.replace('.flac', '.wav')}"
                            dst = os.path.join(self.config.UNIFIED_AUDIO_DIR, 'en', dst_filename)
                            
                            if not os.path.exists(dst):
                                convert_audio(src, dst, self.config.SAMPLE_RATE)
                            
                            duration = self._get_duration(dst)
                            if duration == 0.0:
                                skipped += 1
                                pbar.set_postfix_str(f"⚠️ Invalid duration")
                                pbar.update(1)
                                continue
                            
                            records.append({
                                'audio_path': dst,
                                'speaker_id': f"libri_{spk}",
                                'language': 'en',
                                'gender': speaker_info.get(spk, 'U'),
                                'dataset': 'librispeech',
                                'duration_sec': duration
                            })
                            
                            processed += 1
                            pbar.set_postfix_str(f"✅ {processed} | ⚠️ {skipped} | ❌ {errors}")
                            pbar.update(1)
                            
                        except Exception as e:
                            errors += 1
                            pbar.set_postfix_str(f"❌ Error: {str(e)[:30]}")
                            pbar.update(1)
                            continue
        
        elapsed_time = time.time() - start_time
        self._track_dataset_stats(dataset_name, processed, skipped, errors, elapsed_time)
        
        self._log_progress(f"✅ {dataset_name} completed: {processed} files, {skipped} skipped, {errors} errors in {elapsed_time:.1f}s")
        return records
    
    def process_vctk(self) -> List[Dict]:
        """Process VCTK corpus."""
        dataset_name = "VCTK"
        self._log_progress(f"Starting {dataset_name} processing...")
        
        dataset = self.config.DATASETS['vctk']
        wav_dir = os.path.join(dataset['path'], dataset['audio_dir'])
        
        if not os.path.exists(wav_dir):
            self._log_progress(f"Directory not found: {wav_dir}", "ERROR")
            return []
        
        # Read speaker info
        speaker_info = {}
        info_path = os.path.join(dataset['path'], dataset['speaker_info'])
        
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    lines = f.readlines()[1:]  # Skip header
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            spk = parts[0]
                            gender = parts[2]
                            speaker_info[spk] = gender
            except Exception as e:
                self._log_progress(f"Error reading speaker-info.txt: {e}", "WARNING")
        
        # Count total files
        total_files = 0
        for spk in os.listdir(wav_dir):
            spk_path = os.path.join(wav_dir, spk)
            if os.path.isdir(spk_path):
                total_files += len([f for f in os.listdir(spk_path) if f.endswith('.flac')])
        
        records = []
        processed = 0
        skipped = 0
        errors = 0
        start_time = time.time()
        
        with tqdm(total=total_files, desc=f"📁 {dataset_name}", 
                  unit="files", ncols=100) as pbar:
            
            for spk in os.listdir(wav_dir):
                spk_path = os.path.join(wav_dir, spk)
                if not os.path.isdir(spk_path):
                    continue
                
                for fname in os.listdir(spk_path):
                    if not fname.endswith('.flac'):
                        continue
                    
                    try:
                        src = os.path.join(spk_path, fname)
                        dst_filename = f"vctk_{spk}_{fname.replace('.flac', '.wav')}"
                        dst = os.path.join(self.config.UNIFIED_AUDIO_DIR, 'en', dst_filename)
                        
                        if not os.path.exists(dst):
                            convert_audio(src, dst, self.config.SAMPLE_RATE)
                        
                        duration = self._get_duration(dst)
                        if duration == 0.0:
                            skipped += 1
                            pbar.set_postfix_str(f"⚠️ Invalid duration")
                            pbar.update(1)
                            continue
                        
                        records.append({
                            'audio_path': dst,
                            'speaker_id': f"vctk_{spk}",
                            'language': 'en',
                            'gender': speaker_info.get(spk, 'U'),
                            'dataset': 'vctk',
                            'duration_sec': duration
                        })
                        
                        processed += 1
                        pbar.set_postfix_str(f"✅ {processed} | ⚠️ {skipped} | ❌ {errors}")
                        pbar.update(1)
                        
                    except Exception as e:
                        errors += 1
                        pbar.set_postfix_str(f"❌ Error: {str(e)[:30]}")
                        pbar.update(1)
                        continue
        
        elapsed_time = time.time() - start_time
        self._track_dataset_stats(dataset_name, processed, skipped, errors, elapsed_time)
        
        self._log_progress(f"✅ {dataset_name} completed: {processed} files, {skipped} skipped, {errors} errors in {elapsed_time:.1f}s")
        return records
    
    def process_ta_in_gender(self, gender_type: str) -> List[Dict]:
        """Process Tamil in-house datasets (female or male)."""
        dataset_name = f"Tamil {gender_type.upper()}"
        self._log_progress(f"Starting {dataset_name} processing...")
        
        dataset = self.config.DATASETS[gender_type]
        base_path = dataset['path']
        gender_prefix = dataset['gender_prefix']
        gender = 'F' if gender_prefix == 'taf' else 'M'
        
        if not os.path.exists(base_path):
            self._log_progress(f"Directory not found: {base_path}", "ERROR")
            return []
        
        # Count total files
        total_files = len([f for f in os.listdir(base_path) 
                          if f.endswith('.wav') and f.startswith(gender_prefix)])
        
        records = []
        processed = 0
        skipped = 0
        errors = 0
        start_time = time.time()
        
        with tqdm(total=total_files, desc=f"📁 {dataset_name}", 
                  unit="files", ncols=100) as pbar:
            
            for fname in os.listdir(base_path):
                if not fname.endswith('.wav') or not fname.startswith(gender_prefix):
                    continue
                
                try:
                    # Extract speaker ID from filename
                    parts = fname.split('_')
                    if len(parts) >= 2:
                        spk_id = parts[1]
                    else:
                        spk_id = fname
                    
                    src = os.path.join(base_path, fname)
                    dst_filename = f"ta_{gender_prefix}_{spk_id}_{fname}"
                    dst = os.path.join(self.config.UNIFIED_AUDIO_DIR, 'ta', dst_filename)
                    
                    if not os.path.exists(dst):
                        convert_audio(src, dst, self.config.SAMPLE_RATE)
                    
                    duration = self._get_duration(dst)
                    if duration == 0.0:
                        skipped += 1
                        pbar.set_postfix_str(f"⚠️ Invalid duration")
                        pbar.update(1)
                        continue
                    
                    records.append({
                        'audio_path': dst,
                        'speaker_id': f"ta_{gender_prefix}_{spk_id}",
                        'language': 'ta',
                        'gender': gender,
                        'dataset': gender_type,
                        'duration_sec': duration
                    })
                    
                    processed += 1
                    pbar.set_postfix_str(f"✅ {processed} | ⚠️ {skipped} | ❌ {errors}")
                    pbar.update(1)
                    
                except Exception as e:
                    errors += 1
                    pbar.set_postfix_str(f"❌ Error: {str(e)[:30]}")
                    pbar.update(1)
                    continue
        
        elapsed_time = time.time() - start_time
        self._track_dataset_stats(dataset_name, processed, skipped, errors, elapsed_time)
        
        self._log_progress(f"✅ {dataset_name} completed: {processed} files, {skipped} skipped, {errors} errors in {elapsed_time:.1f}s")
        return records
    
    def process_casual_tamil(self) -> List[Dict]:
        """Process casual Tamil YouTube chunks."""
        dataset_name = "Casual Tamil"
        self._log_progress(f"Starting {dataset_name} processing...")
        
        dataset = self.config.DATASETS['casual_tamil']
        metadata_path = os.path.join(dataset['path'], dataset['metadata'])
        
        if not os.path.exists(metadata_path):
            self._log_progress(f"Metadata not found: {metadata_path}", "ERROR")
            return []
        
        try:
            df = pd.read_csv(metadata_path)
        except Exception as e:
            self._log_progress(f"Error reading metadata: {e}", "ERROR")
            return []
        
        records = []
        processed = 0
        skipped = 0
        errors = 0
        start_time = time.time()
        
        with tqdm(total=len(df), desc=f"📁 {dataset_name}", 
                  unit="files", ncols=100) as pbar:
            
            for idx, row in df.iterrows():
                try:
                    src = row['file_path']
                    if not os.path.exists(src):
                        pbar.set_postfix_str(f"⚠️ Missing file")
                        skipped += 1
                        pbar.update(1)
                        continue
                    
                    # Check quality with sampling
                    try:
                        audio, _ = librosa.load(src, sr=self.config.SAMPLE_RATE, 
                                                mono=True, duration=5.0)
                        if compute_energy(audio) < self.config.ENERGY_THRESHOLD:
                            pbar.set_postfix_str(f"⚠️ Low energy")
                            skipped += 1
                            pbar.update(1)
                            continue
                    except Exception as e:
                        pbar.set_postfix_str(f"⚠️ Audio load error")
                        skipped += 1
                        pbar.update(1)
                        continue
                    
                    # Copy to unified directory
                    creator = str(row['creator']).replace(' ', '_')
                    chunk_id = row['chunk_id']
                    dst_filename = f"casual_{creator}_{chunk_id}.wav"
                    dst = os.path.join(self.config.UNIFIED_AUDIO_DIR, 'ta', dst_filename)
                    
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    
                    duration = self._get_duration(dst)
                    if duration == 0.0:
                        duration = row.get('duration_sec', 0.0)
                    
                    records.append({
                        'audio_path': dst,
                        'speaker_id': f"casual_{creator}",
                        'language': 'ta',
                        'gender': row.get('gender', 'U'),
                        'dialect': row.get('dialect', 'unknown'),
                        'style': row.get('style', 'unknown'),
                        'dataset': 'casual_youtube',
                        'duration_sec': duration
                    })
                    
                    processed += 1
                    pbar.set_postfix_str(f"✅ {processed} | ⚠️ {skipped} | ❌ {errors}")
                    pbar.update(1)
                    
                except Exception as e:
                    errors += 1
                    pbar.set_postfix_str(f"❌ Error: {str(e)[:30]}")
                    pbar.update(1)
                    continue
        
        elapsed_time = time.time() - start_time
        self._track_dataset_stats(dataset_name, processed, skipped, errors, elapsed_time)
        
        self._log_progress(f"✅ {dataset_name} completed: {processed} files, {skipped} skipped, {errors} errors in {elapsed_time:.1f}s")
        return records
    
    def run(self) -> pd.DataFrame:
        """Process all datasets and return unified metadata."""
        self.stats['start_time'] = datetime.now().isoformat()
        overall_start = time.time()
        
        print("\n" + "=" * 80)
        print("🎙️  MULTILINGUAL SPEAKER ENCODER DATA PREPARATION")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.config.UNIFIED_AUDIO_DIR}")
        print("=" * 80)
        
        # Create output directories
        os.makedirs(self.config.UNIFIED_AUDIO_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.config.UNIFIED_AUDIO_DIR, 'en'), exist_ok=True)
        os.makedirs(os.path.join(self.config.UNIFIED_AUDIO_DIR, 'ta'), exist_ok=True)
        
        # Process each dataset
        datasets = [
            ('SPS Corpus', self.process_sps_corpus),
            ('LibriSpeech', self.process_librispeech),
            ('VCTK', self.process_vctk),
            ('Tamil Female', lambda: self.process_ta_in_gender('ta_in_female')),
            ('Tamil Male', lambda: self.process_ta_in_gender('ta_in_male')),
            ('Casual Tamil', self.process_casual_tamil)
        ]
        
        for name, process_func in datasets:
            print(f"\n{'─' * 80}")
            self.records.extend(process_func())
        
        # Create DataFrame
        if not self.records:
            self._log_progress("No records were processed. Check your dataset paths.", "ERROR")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.records)
        
        # Filter by duration
        original_count = len(df)
        df = df[(df['duration_sec'] >= self.config.MIN_DURATION) & 
                (df['duration_sec'] <= self.config.MAX_DURATION)]
        
        # Final summary
        overall_elapsed = time.time() - overall_start
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['total_elapsed_seconds'] = overall_elapsed
        
        print("\n" + "=" * 80)
        print("📊 FINAL PROCESSING SUMMARY")
        print("=" * 80)
        
        print(f"\n✅ Total audio files processed: {len(df)} / {original_count}")
        print(f"   English: {len(df[df['language']=='en'])}")
        print(f"   Tamil: {len(df[df['language']=='ta'])}")
        
        print(f"\n📈 Duration statistics:")
        print(f"   Min: {df['duration_sec'].min():.2f}s")
        print(f"   Max: {df['duration_sec'].max():.2f}s")
        print(f"   Mean: {df['duration_sec'].mean():.2f}s")
        print(f"   Std: {df['duration_sec'].std():.2f}s")
        
        # Speaker statistics
        speaker_counts = df['speaker_id'].value_counts()
        print(f"\n👥 Speaker statistics:")
        print(f"   Total speakers: {len(speaker_counts)}")
        print(f"   Avg utterances/speaker: {speaker_counts.mean():.2f}")
        print(f"   Min utterances: {speaker_counts.min()}")
        print(f"   Max utterances: {speaker_counts.max()}")
        
        # Show top speakers
        print(f"\n🏆 Top 10 speakers by utterance count:")
        for spk, count in speaker_counts.head(10).items():
            language = df[df['speaker_id'] == spk]['language'].iloc[0]
            print(f"   {spk[:40]:<40} ({language}): {count:>5} utterances")
        
        # Dataset breakdown
        print(f"\n📁 Dataset breakdown:")
        dataset_counts = df['dataset'].value_counts()
        for ds, count in dataset_counts.items():
            print(f"   {ds:<20}: {count:>6} files")
        
        # Gender distribution
        print(f"\n🚻 Gender distribution:")
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            print(f"   {gender:<6}: {count:>6} files")
        
        print("\n" + "=" * 80)
        print(f"⏱️  Total processing time: {overall_elapsed:.1f} seconds ({overall_elapsed/60:.1f} minutes)")
        print("=" * 80)
        
        # Save statistics
        self._save_stats()
        
        return df