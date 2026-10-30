"""
train.py

Huấn luyện mô hình Text-to-Music Transformer.

Input:
    processed_data/text_vocab.json
    processed_data/music_vocab.json
    processed_data/text_music_dataset.json

Output:
    checkpoints/best_model.pt
    checkpoints/last_model.pt

Quy trình:

    Text + Music Dataset
            ↓
        DataLoader
            ↓
    Text-to-Music Transformer
            ↓
      Cross Entropy Loss
            ↓
        Backpropagation
            ↓
        Model Update
            ↓
       Validation
            ↓
    checkpoints/best_model.pt
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from model import (
    TextToMusicTransformer,
    create_model,
    print_model_summary,
    MAX_TEXT_LENGTH,
    MAX_MUSIC_LENGTH,
    TEXT_PAD_ID,
    MUSIC_PAD_ID,
    MUSIC_BOS_ID,
    MUSIC_EOS_ID,
)
from text_dataset import TextMusicDataset


# ============================================================
# 1. PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "processed_data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

TEXT_VOCAB_FILE = PROCESSED_DIR / "text_vocab.json"
MUSIC_VOCAB_FILE = PROCESSED_DIR / "music_vocab.json"
DATASET_FILE = PROCESSED_DIR / "text_music_dataset.json"

BEST_MODEL_FILE = CHECKPOINT_DIR / "best_model.pt"
LAST_MODEL_FILE = CHECKPOINT_DIR / "last_model.pt"


# ============================================================
# 2. TRAINING CONFIG
# ============================================================

# Số epoch.
NUM_EPOCHS = 4

# Batch size.
#
# Vì music sequence có thể dài đến 2048 token,
# batch size 8 có thể tốn khá nhiều RAM.
#
# Nếu máy yếu có thể giảm xuống 2 hoặc 4.
BATCH_SIZE = 2

# Learning rate.
LEARNING_RATE = 3e-4

# Weight decay giúp regularization.
WEIGHT_DECAY = 1e-4

# Tỷ lệ validation.
VALIDATION_RATIO = 0.1

# Random seed.
RANDOM_SEED = 42

# Số worker cho DataLoader.
#
# Windows thường nên để 0 để tránh các vấn đề multiprocessing
# khi mới bắt đầu.
NUM_WORKERS = 0

# Gradient clipping.
MAX_GRAD_NORM = 1.0
USE_AMP=torch.cuda.is_available()

# Số epoch không cải thiện validation loss trước khi early stopping.
EARLY_STOPPING_PATIENCE = 5

# In progress mỗi bao nhiêu batch.
PRINT_EVERY = 10


# ============================================================
# 3. DEVICE
# ============================================================

def get_device() -> torch.device:
    """
    Chọn device để train.

    Nếu máy có CUDA:
        NVIDIA GPU → cuda

    Nếu không:
        CPU
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print("CUDA available.")
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:
        device = torch.device("cpu")

        print("CUDA not available.")
        print("Using CPU.")

    return device


# ============================================================
# 4. RANDOM SEED
# ============================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Thiết lập random seed để kết quả có tính ổn định hơn.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 5. LOAD JSON
# ============================================================

def load_json(file_path: Path):
    """
    Đọc JSON file.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# 6. LOAD VOCABULARY
# ============================================================

def load_vocabulary(file_path: Path) -> dict:
    """
    Load vocabulary.

    music_vocab.json và text_vocab.json thường có dạng:

        {
            "PAD": 0,
            "BOS": 1,
            ...
        }

    hoặc:

        {
            "<PAD>": 0,
            "<UNK>": 1,
            ...
        }
    """

    vocabulary = load_json(file_path)

    if not isinstance(vocabulary, dict):
        raise ValueError(
            f"Vocabulary phải là dictionary: {file_path}"
        )

    return vocabulary


# ============================================================
# 7. KIỂM TRA SPECIAL TOKENS
# ============================================================

def print_special_tokens(
    text_vocab: dict,
    music_vocab: dict,
) -> None:
    """
    In các special token để kiểm tra vocabulary.
    """

    print()
    print("=" * 60)
    print("VOCABULARY INFORMATION")
    print("=" * 60)

    print(
        f"Text vocabulary size : "
        f"{len(text_vocab)}"
    )

    print(
        f"Music vocabulary size: "
        f"{len(music_vocab)}"
    )

    print()

    print("Text special tokens:")

    for token in [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
    ]:
        if token in text_vocab:
            print(
                f"  {token:10s} → "
                f"{text_vocab[token]}"
            )

    print()

    print("Music special tokens:")

    for token in [
        "PAD",
        "BOS",
        "EOS",
        "UNK",
    ]:
        if token in music_vocab:
            print(
                f"  {token:10s} → "
                f"{music_vocab[token]}"
            )

    print("=" * 60)


# ============================================================
# 8. CREATE DATASET
# ============================================================

def create_dataset():
    """
    Tạo TextMusicDataset từ text_music_dataset.json.
    """

    print()
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    raw_dataset = load_json(DATASET_FILE)

    if not isinstance(raw_dataset, list):
        raise ValueError(
            "text_music_dataset.json phải chứa một list."
        )

    if len(raw_dataset) == 0:
        raise ValueError(
            "Dataset rỗng. "
            "Hãy kiểm tra lại text_dataset.py."
        )

    print(
        f"Total dataset samples: "
        f"{len(raw_dataset)}"
    )

    dataset = TextMusicDataset(
        raw_dataset
    )

    print(
        f"PyTorch dataset size: "
        f"{len(dataset)}"
    )

    return dataset


# ============================================================
# 9. SPLIT DATASET
# ============================================================

def split_dataset(dataset):
    """
    Chia dataset thành:

        Train
        Validation
    """

    dataset_size = len(dataset)

    if dataset_size < 2:
        raise ValueError(
            "Dataset cần ít nhất 2 samples "
            "để chia train/validation."
        )

    # Chia theo MIDI thay vì theo caption để cùng một bản nhạc không
    # xuất hiện đồng thời trong train và validation.
    indices_by_music_file = {}

    for index, item in enumerate(dataset.dataset):
        indices_by_music_file.setdefault(
            item["music_file"], []
        ).append(index)

    music_files = list(indices_by_music_file)
    random.Random(RANDOM_SEED).shuffle(music_files)

    validation_file_count = max(
        1,
        int(len(music_files) * VALIDATION_RATIO),
    )
    validation_files = set(music_files[:validation_file_count])

    train_indices = []
    validation_indices = []

    for music_file, indices in indices_by_music_file.items():
        if music_file in validation_files:
            validation_indices.extend(indices)
        else:
            train_indices.extend(indices)

    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)

    print()
    print("Dataset split:")
    print(
        f"  Train      : "
        f"{len(train_dataset)}"
    )

    print(
        f"  Validation : "
        f"{len(validation_dataset)}"
    )

    return train_dataset, validation_dataset


# ============================================================
# 10. CREATE DATALOADER
# ============================================================

def create_dataloaders(
    train_dataset,
    validation_dataset,
):
    """
    Tạo DataLoader cho train và validation.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, validation_loader


# ============================================================
# 11. PREPARE MUSIC INPUT / TARGET
# ============================================================

def prepare_music_sequences(
    music_ids: torch.Tensor,
):
    """
    Tạo input và target cho Teacher Forcing.

    Ví dụ:

        Original:

        [BOS, A, B, C, EOS, PAD]

        music_input:

        [BOS, A, B, C]

        target:

        [A, B, C, EOS]

    Model học:

        BOS → A
        A   → B
        B   → C
        C   → EOS
    """

    music_input_ids = music_ids[:, :-1]

    music_target_ids = music_ids[:, 1:]

    return (
        music_input_ids,
        music_target_ids,
    )


# ============================================================
# 12. CREATE LOSS
# ============================================================

def create_loss_function() -> nn.Module:
    """
    Cross Entropy Loss.

    PAD token sẽ không được tính vào loss.

    Điều này rất quan trọng vì music sequences
    được padding tới MAX_MUSIC_LENGTH.
    """

    criterion = nn.CrossEntropyLoss(
        ignore_index=MUSIC_PAD_ID,
    )

    return criterion


# ============================================================
# 13. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model: TextToMusicTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    scaler,
):
    """
    Train model trong một epoch.
    """

    model.train()

    total_loss = 0.0

    total_batches = len(dataloader)

    start_time = time.time()

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):

        # ----------------------------------------------------
        # Load batch
        # ----------------------------------------------------

        text_ids = batch["text_ids"].to(
            device,
            non_blocking=True,
        )

        music_ids = batch["music_ids"].to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Teacher Forcing
        # ----------------------------------------------------

        (
            music_input_ids,
            music_target_ids,
        ) = prepare_music_sequences(
            music_ids
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled= USE_AMP,
        ):
            logits = model(
                text_ids=text_ids,
                music_input_ids=music_input_ids,
            )

            logits = logits.reshape(
                -1,
                logits.size(-1),
            )

            targets = music_target_ids.reshape(
                -1
            )

            loss = criterion(
                logits,
                targets,
            )

        # ----------------------------------------------------
        # Check NaN
        # ----------------------------------------------------

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Loss trở thành NaN hoặc Inf."
            )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        scaler.scale(loss).backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )

        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        scaler.step(optimizer)
        scaler.update()

        # ----------------------------------------------------
        # Accumulate loss
        # ----------------------------------------------------

        total_loss += loss.item()

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        if (
            batch_index == 1
            or batch_index % PRINT_EVERY == 0
            or batch_index == total_batches
        ):

            average_loss = (
                total_loss / batch_index
            )

            elapsed = (
                time.time() - start_time
            )

            print(
                f"Epoch {epoch:02d} | "
                f"Batch {batch_index:04d}/{total_batches:04d} | "
                f"Loss: {loss.item():.4f} | "
                f"Avg: {average_loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

    epoch_loss = (
        total_loss / total_batches
    )

    return epoch_loss


# ============================================================
# 14. VALIDATE ONE EPOCH
# ============================================================

@torch.no_grad()
def validate(
    model: TextToMusicTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """
    Validation model.
    """

    model.eval()

    total_loss = 0.0

    total_batches = len(dataloader)

    for batch in dataloader:

        text_ids = batch["text_ids"].to(
            device,
            non_blocking=True,
        )

        music_ids = batch["music_ids"].to(
            device,
            non_blocking=True,
        )

        # Teacher forcing.
        (
            music_input_ids,
            music_target_ids,
        ) = prepare_music_sequences(
            music_ids
        )

        # Forward.
        logits = model(
            text_ids=text_ids,
            music_input_ids=music_input_ids,
        )

        # Reshape.
        logits = logits.reshape(
            -1,
            logits.size(-1),
        )

        targets = music_target_ids.reshape(
            -1
        )

        # Loss.
        loss = criterion(
            logits,
            targets,
        )

        total_loss += loss.item()

    validation_loss = (
        total_loss / total_batches
    )

    return validation_loss


# ============================================================
# 15. SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    validation_loss,
    file_path,
):
    """
    Lưu checkpoint.
    """

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict()
            if scheduler is not None
            else None,

        "train_loss":
            train_loss,

        "validation_loss":
            validation_loss,
    }

    torch.save(
        checkpoint,
        file_path,
    )


# ============================================================
# 16. MAIN TRAINING
# ============================================================

def main():

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Create checkpoint directory
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load vocabulary
    # --------------------------------------------------------

    print()
    print("Loading vocabularies...")

    text_vocab = load_vocabulary(
        TEXT_VOCAB_FILE
    )

    music_vocab = load_vocabulary(
        MUSIC_VOCAB_FILE
    )

    print_special_tokens(
        text_vocab,
        music_vocab,
    )

    # --------------------------------------------------------
    # Check vocabulary size
    # --------------------------------------------------------

    text_vocab_size = len(
        text_vocab
    )

    music_vocab_size = len(
        music_vocab
    )

    if text_vocab_size <= 0:
        raise ValueError(
            "Text vocabulary rỗng."
        )

    if music_vocab_size <= 0:
        raise ValueError(
            "Music vocabulary rỗng."
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = create_dataset()

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
    ) = split_dataset(dataset)

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    (
        train_loader,
        validation_loader,
    ) = create_dataloaders(
        train_dataset,
        validation_dataset,
    )

    print()
    print(
        f"Train batches      : "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches : "
        f"{len(validation_loader)}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()
    print("Creating model...")

    model = create_model(
        text_vocab_size=text_vocab_size,
        music_vocab_size=music_vocab_size,
        device=device,
    )

    print_model_summary(model)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = create_loss_function()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # --------------------------------------------------------
    # Training variables
    # --------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    training_start = time.time()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("START TRAINING")
    print("=" * 60)

    print(
        f"Epochs       : {NUM_EPOCHS}"
    )

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Device       : {device}"
    )

    print("=" * 60)

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print()
        print(
            "=" * 60
        )

        print(
            f"EPOCH {epoch}/{NUM_EPOCHS}"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            scaler=scaler,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_loss = validate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            validation_loss
        )

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print()
        print(
            f"Epoch {epoch:02d} result:"
        )

        print(
            f"  Train Loss      : "
            f"{train_loss:.4f}"
        )

        print(
            f"  Validation Loss : "
            f"{validation_loss:.4f}"
        )

        print(
            f"  Learning Rate   : "
            f"{current_lr:.8f}"
        )

        # ----------------------------------------------------
        # Save last checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            file_path=LAST_MODEL_FILE,
        )

        # ----------------------------------------------------
        # Check best model
        # ----------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                file_path=BEST_MODEL_FILE,
            )

            print()
            print(
                "✓ Validation loss improved."
            )

            print(
                f"✓ Best model saved to:"
            )

            print(
                f"  {BEST_MODEL_FILE}"
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                "Validation loss did not improve."
            )

            print(
                f"No improvement: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Training completed
    # --------------------------------------------------------

    total_time = (
        time.time() - training_start
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"Training time: "
        f"{total_time / 60:.2f} minutes"
    )

    print()
    print(
        "Best model:"
    )

    print(
        f"{BEST_MODEL_FILE}"
    )

    print()
    print(
        "Last model:"
    )

    print(
        f"{LAST_MODEL_FILE}"
    )

    print("=" * 60)


# ============================================================
# 17. RUN
# ============================================================

if __name__ == "__main__":
    main()
