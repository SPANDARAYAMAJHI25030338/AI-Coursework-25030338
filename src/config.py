"""Project-wide configuration: paths, hyperparameters, constants.

All other modules import from here so paths and hyperparameters live in one place.
Educational note: centralising config makes experiments reproducible and lets us
swap hyperparameters without hunting through code.
"""

from pathlib import Path

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "Datasets"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "report"

# Primary corpus: MDCC GoFundMe (Xu et al., CIKM 2023)
MDCC_CSV = DATASETS_DIR / "MDCC_GoFundMe" / "raw_data.csv"
MDCC_TEXT_COL = "clean_description"  # cleaner field per the dataset README
MDCC_ID_COL = "campaign_id"
MDCC_CAT_COL = "category"

# Secondary corpora (loaded only by phases that need them)
PFG_DIR = DATASETS_DIR / "PersuasionForGood" / "persuasionforgood-master" / "data"
GOEMOTIONS_DIR = DATASETS_DIR / "GoEmotions"
EMPATHETIC_DIR = DATASETS_DIR / "EmpatheticDialogues" / "empatheticdialogues"
WEBROBOTS_DIR = DATASETS_DIR / "Kickstarter_WebRobots"
KAGGLE_KS = DATASETS_DIR / "ks-projects-201801.csv"

# Derived / output paths
WEAK_LABELS_CSV = DATA_DIR / "annotations" / "mdcc_weak_labels.csv"
PROCESSED_CSV = DATA_DIR / "processed" / "mdcc_processed.csv"
EMBEDDINGS_NPY = DATA_DIR / "embeddings" / "mdcc_embeddings.npy"
FAISS_INDEX = DATA_DIR / "embeddings" / "mdcc.index"

FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"

# --- Reproducibility --------------------------------------------------------
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.2

# --- Label scheme -----------------------------------------------------------
# Binary primary task per the project's "if dataset size is limited, simplify to
# Manipulative / Non-Manipulative" guidance.
LABELS_BINARY = ["non_manipulative", "manipulative"]
LABEL2ID = {l: i for i, l in enumerate(LABELS_BINARY)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# Four-class fine-grained scheme (also produced for analysis / future work)
LABELS_4CLASS = [
    "non_manipulative",
    "sympathy_exploitation",
    "artificial_urgency",
    "guilt_or_fear",  # merged because guilt and fear lexicons overlap heavily
]

# --- TF-IDF baseline hyperparameters ---------------------------------------
TFIDF_MAX_FEATURES = 15000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_SUBLINEAR_TF = True
LR_MAX_ITER = 1000
LR_CLASS_WEIGHT = "balanced"

# --- Transformer hyperparameters -------------------------------------------
TRANSFORMER_NAME = "distilbert-base-uncased"
TRANSFORMER_MAX_LEN = 192   # tuned for CPU throughput; project doc allows up to 512
TRANSFORMER_LR = 2e-5
TRANSFORMER_EPOCHS = 2      # 2 epochs is enough on weak labels; documented
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_EVAL_BATCH = 32
TRANSFORMER_WEIGHT_DECAY = 0.01

# CPU training is slow; subsample MDCC to a stratified subset for the
# transformer to keep this coursework-realistic. 4000 rows is still large
# enough for meaningful metrics and finishes in ~20–30 min on Apple Silicon CPU.
TRANSFORMER_TRAIN_SUBSAMPLE = 4000

# --- Embedding / retrieval -------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
RETRIEVAL_TOP_K = 5

# --- Evaluation -------------------------------------------------------------
METRIC_DECIMALS = 4
