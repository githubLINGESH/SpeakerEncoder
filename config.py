import os

# Paths
BASE_DIR = "D:/Zero_shotVoiceClone"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")
_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
UNIFIED_AUDIO_DIR = os.path.join(OUTPUT_DIR, "unified_audio")
METADATA_PATH = os.path.join(OUTPUT_DIR, "unified_metadata.csv")

# Dataset paths
DATASETS = {
    'sps_corpus': {
        'path': os.path.join(DATA_DIR, "english/1764158905630-sps-corpus-1.0-2025-11-25-en/sps-corpus-1.0-2025-11-25-en"),
        'audio_dir': "audios",
        'tsv': "ss-corpus-en.tsv",
        'reported_tsv': "ss-reported-audios-en.tsv"
    },
    'librispeech': {
        'path': os.path.join(DATA_DIR, "english/train-clean-100"),
        'audio_dir': "LibriSpeech/train-clean-100"
    },
    'vctk': {
        'path': os.path.join(DATA_DIR, "english/VCTK-Corpus-0.92"),
        'audio_dir': "wav48_silence_trimmed",
        'speaker_info': "speaker-info.txt"
    },
    'ta_in_female': {
        'path': os.path.join(DATA_DIR, "tamil/ta_in_female"),
        'gender_prefix': 'taf'
    },
    'ta_in_male': {
        'path': os.path.join(DATA_DIR, "tamil/ta_in_male"),
        'gender_prefix': 'tam'
    },
    'casual_tamil': {
        'path': os.path.join(DATA_DIR, "tamil/casual/chunked_dataset"),
        'metadata': "chunk_metadata.csv"
    }
}

# Audio parameters
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = "wav"

# Segment sampling (CRITICAL for few-shot)
MIN_SEGMENT_DURATION = 1.0  # seconds
MAX_SEGMENT_DURATION = 5.0  # seconds

# Quality filters
MIN_DURATION = 0.8   # seconds - skip files shorter than this
MAX_DURATION = 15.0  # seconds - split longer files (already handled by VAD)
MIN_UTTERANCES_PER_SPEAKER = 3  # Minimum samples per speaker
ENERGY_THRESHOLD = 0.01  # For filtering low-energy clips

# Training parameters
NUM_SPEAKERS_PER_BATCH = 32
UTTERANCES_PER_SPEAKER = 2
EMBEDDING_DIM = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
BATCH_SIZE = NUM_SPEAKERS_PER_BATCH * UTTERANCES_PER_SPEAKER  # 64

# Augmentation parameters
USE_AUGMENTATION = True
NOISE_SNR_MIN = 10  # dB
NOISE_SNR_MAX = 30  # dB
REVERB_PROB = 0.3
SPEED_PERTURB_PROB = 0.2
SPEED_FACTORS = [0.9, 1.0, 1.1]

# Model architecture
MODEL_TYPE = "ecapa_tdnn"  # or "wavlm"
ECAPA_CHANNELS = [512, 512, 512, 512, 1536]
ECAPA_KERNEL_SIZES = [5, 5, 5, 5, 5]
ECAPA_DILATIONS = [1, 2, 3, 4, 1]

# Loss parameters
MARGIN = 0.2
LOSS_TYPE = "triplet"  # or "aamsoftmax"

# ============ MODEL ARCHITECTURE ============
# Timbre Branch (ECAPA-TDNN)
TIMBRE_EMBED_DIM = 128
ECAPA_CHANNELS = [512, 512, 512, 512, 1536]
ECAPA_KERNEL_SIZES = [5, 5, 5, 5, 5]
ECAPA_DILATIONS = [1, 2, 3, 4, 1]

# Cadence Branch (Temporal)
CADENCE_EMBED_DIM = 128
CADENCE_INPUT_DIM = 44  # MFCC features
CADENCE_HIDDEN_DIM = 256
CADENCE_NUM_LAYERS = 2
CADENCE_DROPOUT = 0.2

# SSL Branch
SSL_EMBED_DIM = 128
SSL_MODEL_NAME = "facebook/hubert-base-ls960"  # or wav2vec2
SSL_LAYER = 9  # Which transformer layer to extract

# Fusion
FUSION_EMBED_DIM = 256
FUSION_TYPE = "attention"  # "attention" or "concat"
FUSION_DROPOUT = 0.2

# Final Embedding
FINAL_EMBED_DIM = 128
USE_L2_NORM = True

# ============ LOSS FUNCTIONS ============
# AAM-Softmax
AAM_MARGIN = 0.2
AAM_SCALE = 30
AAM_EASYMARGIN = False

# Contrastive Loss
CONTRASTIVE_TEMPERATURE = 0.07
CONTRASTIVE_WEIGHT = 0.3

# GE2E Loss (optional)
GE2E_WEIGHT = 0.1

# ============ TRAINING ============
NUM_EPOCHS = 50
NUM_SPEAKERS_PER_BATCH = 16
UTTERANCES_PER_SPEAKER = 4
BATCH_SIZE = NUM_SPEAKERS_PER_BATCH * UTTERANCES_PER_SPEAKER  # 64

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = 1000

# ============ AUGMENTATIONS ============
AUGMENTATION_PROB = 0.8
NOISE_SNR_MIN = 10
NOISE_SNR_MAX = 30
REVERB_PROB = 0.3
SPEED_PERTURB_PROB = 0.2
SPEED_FACTORS = [0.9, 1.0, 1.1]

# ============ EVALUATION ============
EVAL_BATCH_SIZE = 32
EVAL_SPLIT = "test"