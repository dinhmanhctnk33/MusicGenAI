"""
text_dataset.py

Xây dựng dataset Text → Music cho hệ thống Music GenAI.

Pipeline:

    music_tokens.json
            +
    music_vocab.json
            +
    caption_dataset.json
            ↓
       text_dataset.py
            ↓
    text_vocab.json
            +
    text_music_dataset.json


Input:

    processed_data/music_tokens.json
        Chứa music token dạng STRING, ví dụ:

            [
                "BOS",
                "NOTE_ON_60",
                "VELOCITY_80",
                "DURATION_4",
                "EOS"
            ]


    processed_data/music_vocab.json
        Mapping:

            {
                "PAD": 0,
                "BOS": 1,
                "EOS": 2,
                ...
            }


    processed_data/caption_dataset.json
        Chứa caption tự nhiên của từng MIDI.


Output:

    processed_data/text_vocab.json

    processed_data/text_music_dataset.json


Mỗi sample cuối cùng có dạng:

    {
        "text": "một bản nhạc vui vẻ với piano",

        "text_ids": [
            2,
            15,
            37,
            52,
            91,
            3
        ],

        "music_file": "song001.midi",

        "music_tokens": [
            1,
            25,
            87,
            42,
            2
        ]
    }


Lưu ý:

    - Text token được chuyển thành text ID.
    - Music token STRING được chuyển thành music ID
      thông qua music_vocab.json.
    - Mỗi caption của cùng một MIDI trở thành một
      training sample riêng.
"""

import json
import os
import re
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

MUSIC_TOKEN_FILE = (
    "./processed_data/music_tokens.json"
)

MUSIC_VOCAB_FILE = (
    "./processed_data/music_vocab.json"
)

CAPTION_DATASET_FILE = (
    "./processed_data/caption_dataset.json"
)

OUTPUT_DIR = (
    "./processed_data"
)

TEXT_VOCAB_FILE = os.path.join(
    OUTPUT_DIR,
    "text_vocab.json"
)

TEXT_MUSIC_DATASET_FILE = os.path.join(
    OUTPUT_DIR,
    "text_music_dataset.json"
)


# ============================================================
# TEXT CONFIGURATION
# ============================================================

MAX_TEXT_LENGTH = 64

MIN_TEXT_FREQUENCY = 1

TEXT_SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


# ============================================================
# MUSIC CONFIGURATION
# ============================================================

MAX_MUSIC_LENGTH = 512


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42


# ============================================================
# LOAD JSON
# ============================================================

def load_json(file_path: str):
    """
    Đọc dữ liệu từ file JSON.

    Args:
        file_path:
            Đường dẫn tới file JSON.

    Returns:
        object:
            Nội dung JSON.
    """

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Không tìm thấy file: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    data,
    file_path: str
):
    """
    Lưu dữ liệu thành JSON.

    Args:
        data:
            Dữ liệu cần lưu.

        file_path:
            Đường dẫn output.
    """

    output_path = Path(
        file_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    text: str
) -> str:
    """
    Chuẩn hóa text.

    Thực hiện:

        1. Chuyển về lowercase.
        2. Loại bỏ khoảng trắng thừa.
        3. Giữ Unicode tiếng Việt.
        4. Giữ dấu câu.

    Ví dụ:

        "  Một   Bản Nhạc Vui Vẻ  "

    thành:

        "một bản nhạc vui vẻ"

    Args:
        text:
            Text đầu vào.

    Returns:
        str:
            Text đã chuẩn hóa.
    """

    if not isinstance(
        text,
        str
    ):

        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = text.strip()

    return text


# ============================================================
# TOKENIZE TEXT
# ============================================================

def tokenize_text(
    text: str
):
    """
    Tokenize text.

    Ví dụ:

        "một bản nhạc vui vẻ"

    thành:

        [
            "một",
            "bản",
            "nhạc",
            "vui",
            "vẻ"
        ]

    Args:
        text:
            Text đầu vào.

    Returns:
        list[str]
    """

    text = normalize_text(
        text
    )

    if not text:

        return []

    tokens = re.findall(
        r"\w+|[^\w\s]",
        text,
        flags=re.UNICODE
    )

    return tokens


# ============================================================
# BUILD TEXT VOCABULARY
# ============================================================

def build_text_vocabulary(
    captions
):
    """
    Xây dựng vocabulary cho text.

    Special tokens:

        <PAD> = 0
        <UNK> = 1
        <BOS> = 2
        <EOS> = 3

    Args:
        captions:
            Danh sách caption.

    Returns:
        dict:
            token -> ID
    """

    token_counter = {}

    for caption in captions:

        tokens = tokenize_text(
            caption
        )

        for token in tokens:

            token_counter[token] = (
                token_counter.get(
                    token,
                    0
                ) + 1
            )

    vocabulary = {}

    # --------------------------------------------------------
    # Special tokens
    # --------------------------------------------------------

    for index, token in enumerate(
        TEXT_SPECIAL_TOKENS
    ):

        vocabulary[token] = index

    # --------------------------------------------------------
    # Sort token
    # --------------------------------------------------------

    sorted_tokens = sorted(
        token_counter.items(),
        key=lambda item: (
            -item[1],
            item[0]
        )
    )

    next_id = len(
        vocabulary
    )

    # --------------------------------------------------------
    # Add normal tokens
    # --------------------------------------------------------

    for token, frequency in sorted_tokens:

        if frequency < MIN_TEXT_FREQUENCY:
            continue

        if token in vocabulary:
            continue

        vocabulary[token] = next_id

        next_id += 1

    return vocabulary


# ============================================================
# TEXT → IDS
# ============================================================

def text_to_ids(
    text: str,
    vocabulary: dict,
    max_length: int = MAX_TEXT_LENGTH
):
    """
    Chuyển một câu text thành ID.

    Format:

        <BOS>
        token_1
        token_2
        ...
        <EOS>

    Ví dụ:

        "bản nhạc vui"

    →

        [2, 15, 28, 72, 3]

    Args:
        text:
            Caption.

        vocabulary:
            Text vocabulary.

        max_length:
            Độ dài tối đa.

    Returns:
        list[int]
    """

    tokens = tokenize_text(
        text
    )

    token_ids = [
        BOS_ID
    ]

    for token in tokens:

        token_id = vocabulary.get(
            token,
            UNK_ID
        )

        token_ids.append(
            token_id
        )

    token_ids.append(
        EOS_ID
    )

    # --------------------------------------------------------
    # Truncate
    # --------------------------------------------------------

    token_ids = token_ids[
        :max_length
    ]

    # --------------------------------------------------------
    # Ensure EOS
    # --------------------------------------------------------

    if not token_ids:

        return [
            BOS_ID,
            EOS_ID
        ]

    if token_ids[-1] != EOS_ID:

        token_ids[-1] = EOS_ID

    return token_ids


# ============================================================
# MUSIC TOKEN NORMALIZATION
# ============================================================

def normalize_music_tokens(
    music_tokens_data
):
    """
    Chuẩn hóa cấu trúc music_tokens.json.

    Hỗ trợ nhiều format.

    Format 1:

        {
            "song001.midi": [
                "BOS",
                "NOTE_ON_60",
                "EOS"
            ]
        }


    Format 2:

        {
            "song001.midi": {
                "tokens": [
                    "BOS",
                    "NOTE_ON_60",
                    "EOS"
                ]
            }
        }


    Format 3:

        [
            {
                "file_name": "song001.midi",
                "tokens": [
                    "BOS",
                    "NOTE_ON_60",
                    "EOS"
                ]
            }
        ]

    Args:
        music_tokens_data:
            Dữ liệu music tokens.

    Returns:
        dict:
            file_name -> token list
    """

    # ========================================================
    # FORMAT DICT
    # ========================================================

    if isinstance(
        music_tokens_data,
        dict
    ):

        # ----------------------------------------------------
        # Format:
        #
        # {
        #     "song001.midi": [...]
        # }
        # ----------------------------------------------------

        direct_format = {}

        for key, value in (
            music_tokens_data.items()
        ):

            if isinstance(
                value,
                list
            ):

                direct_format[key] = value

        if direct_format:

            return direct_format

        # ----------------------------------------------------
        # Format:
        #
        # {
        #     "song001.midi": {
        #         "tokens": [...]
        #     }
        # }
        # ----------------------------------------------------

        nested_format = {}

        for key, value in (
            music_tokens_data.items()
        ):

            if not isinstance(
                value,
                dict
            ):

                continue

            tokens = value.get(
                "tokens"
            )

            if tokens is None:

                tokens = value.get(
                    "music_tokens"
                )

            if isinstance(
                tokens,
                list
            ):

                nested_format[key] = tokens

        if nested_format:

            return nested_format

    # ========================================================
    # FORMAT LIST
    # ========================================================

    if isinstance(
        music_tokens_data,
        list
    ):

        normalized = {}

        for item in music_tokens_data:

            if not isinstance(
                item,
                dict
            ):

                continue

            file_name = (
                item.get(
                    "file_name"
                )
                or
                item.get(
                    "music_file"
                )
            )

            tokens = (
                item.get(
                    "tokens"
                )
                or
                item.get(
                    "music_tokens"
                )
            )

            if (
                file_name
                and isinstance(
                    tokens,
                    list
                )
            ):

                normalized[
                    file_name
                ] = tokens

        return normalized

    raise ValueError(
        "Không nhận diện được format "
        "của music_tokens.json."
    )


# ============================================================
# MUSIC TOKEN → MUSIC ID
# ============================================================

def music_tokens_to_ids(
    tokens,
    music_vocabulary
):
    """
    Chuyển music token STRING thành music token ID.

    Ví dụ:

        [
            "BOS",
            "NOTE_ON_60",
            "VELOCITY_80",
            "EOS"
        ]

    thành:

        [
            1,
            25,
            87,
            2
        ]

    Args:
        tokens:
            Danh sách music token dạng string.

        music_vocabulary:
            Mapping token -> ID.

    Returns:
        list[int]
    """

    token_ids = []

    for token in tokens:

        # ----------------------------------------------------
        # Trường hợp token đã là số
        # ----------------------------------------------------

        if isinstance(
            token,
            int
        ):

            token_ids.append(
                token
            )

            continue

        # ----------------------------------------------------
        # Trường hợp token là string
        # ----------------------------------------------------

        token = str(
            token
        )

        if token not in music_vocabulary:

            raise ValueError(
                "Music token không tồn tại "
                f"trong music vocabulary: {token}"
            )

        token_id = music_vocabulary[
            token
        ]

        token_ids.append(
            int(token_id)
        )

    return token_ids


# ============================================================
# FIND MUSIC TOKENS
# ============================================================

def find_music_tokens(
    music_file,
    music_tokens
):
    """
    Tìm music tokens tương ứng với một MIDI.

    Hỗ trợ:

        song001.midi

        dataset_clean/song001.midi

        ./dataset_clean/song001.midi

    Args:
        music_file:
            Tên hoặc đường dẫn MIDI.

        music_tokens:
            Mapping file -> tokens.

    Returns:
        list hoặc None
    """

    if music_file in music_tokens:

        return music_tokens[
            music_file
        ]

    normalized_target = os.path.normpath(
        music_file
    )

    # --------------------------------------------------------
    # Match full normalized path
    # --------------------------------------------------------

    for key, tokens in (
        music_tokens.items()
    ):

        normalized_key = (
            os.path.normpath(
                key
            )
        )

        if normalized_key == normalized_target:

            return tokens

    # --------------------------------------------------------
    # Match basename
    # --------------------------------------------------------

    target_name = os.path.basename(
        music_file
    )

    for key, tokens in (
        music_tokens.items()
    ):

        key_name = os.path.basename(
            key
        )

        if key_name == target_name:

            return tokens

    return None


# ============================================================
# PAD SEQUENCE
# ============================================================

def pad_sequence(
    sequence,
    max_length,
    pad_value
):
    """
    Padding sequence tới max_length.

    Args:
        sequence:
            Sequence ID.

        max_length:
            Chiều dài tối đa.

        pad_value:
            Giá trị padding.

    Returns:
        list[int]
    """

    sequence = sequence[
        :max_length
    ]

    padding_length = (
        max_length - len(sequence)
    )

    if padding_length > 0:

        sequence = (
            sequence
            +
            [pad_value] * padding_length
        )

    return sequence


# ============================================================
# CREATE TEXT → MUSIC PAIRS
# ============================================================

def create_text_music_pairs(
    caption_dataset,
    music_tokens,
    music_vocabulary,
    text_vocabulary
):
    """
    Tạo các cặp Text → Music.

    Một MIDI có nhiều caption:

        MIDI
         ├── Caption 1
         ├── Caption 2
         ├── Caption 3
         └── Caption 4

    sẽ trở thành:

        Sample 1
        Sample 2
        Sample 3
        Sample 4


    Args:
        caption_dataset:
            Caption dataset.

        music_tokens:
            Mapping MIDI -> music tokens.

        music_vocabulary:
            Music token -> ID.

        text_vocabulary:
            Text token -> ID.

    Returns:
        list[dict]
    """

    pairs = []

    skipped_no_file = 0
    skipped_no_tokens = 0
    skipped_invalid_tokens = 0
    skipped_empty_caption = 0

    # --------------------------------------------------------
    # Process each MIDI
    # --------------------------------------------------------

    for index, item in enumerate(
        caption_dataset,
        start=1
    ):

        music_file = item.get(
            "music_file"
        )

        if not music_file:

            skipped_no_file += 1

            continue

        # ----------------------------------------------------
        # Find music tokens
        # ----------------------------------------------------

        tokens = find_music_tokens(
            music_file,
            music_tokens
        )

        if tokens is None:

            skipped_no_tokens += 1

            print(
                "[WARNING] Không tìm thấy "
                f"music tokens cho: {music_file}"
            )

            continue

        # ----------------------------------------------------
        # Convert music tokens → IDs
        # ----------------------------------------------------

        try:

            music_ids = music_tokens_to_ids(
                tokens,
                music_vocabulary
            )

        except ValueError as error:

            skipped_invalid_tokens += 1

            print(
                f"[WARNING] {error}"
            )

            continue

        # ----------------------------------------------------
        # Truncate music sequence
        # ----------------------------------------------------

        # Luôn dành vị trí cuối cho EOS khi chuỗi bị cắt ngắn.
        music_ids = music_ids[:MAX_MUSIC_LENGTH]

        eos_id = music_vocabulary["EOS"]

        if len(music_ids) == MAX_MUSIC_LENGTH:
            music_ids[-1] = eos_id
        elif music_ids[-1] != eos_id:
            music_ids.append(eos_id)

        if not music_ids:

            skipped_invalid_tokens += 1

            continue

        # ----------------------------------------------------
        # Captions
        # ----------------------------------------------------

        captions = item.get(
            "captions",
            []
        )

        for caption in captions:

            caption = normalize_text(
                caption
            )

            if not caption:

                skipped_empty_caption += 1

                continue

            # ------------------------------------------------
            # Text → IDs
            # ------------------------------------------------

            text_ids = text_to_ids(
                caption,
                text_vocabulary,
                MAX_TEXT_LENGTH
            )

            # ------------------------------------------------
            # Create sample
            # ------------------------------------------------

            sample = {

                "text": caption,

                "text_ids": text_ids,

                "music_file": music_file,

                "music_tokens": music_ids,

                "attributes": item.get(
                    "attributes",
                    {}
                )
            }

            pairs.append(
                sample
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print(
        "Dataset creation statistics:"
    )

    print(
        f"  Skipped no file       : "
        f"{skipped_no_file}"
    )

    print(
        f"  Skipped no tokens     : "
        f"{skipped_no_tokens}"
    )

    print(
        f"  Skipped invalid token : "
        f"{skipped_invalid_tokens}"
    )

    print(
        f"  Skipped empty caption : "
        f"{skipped_empty_caption}"
    )

    return pairs


# ============================================================
# PYTORCH DATASET
# ============================================================

class TextMusicDataset(
    Dataset
):
    """
    PyTorch Dataset cho Text → Music.
    """

    def __init__(
        self,
        dataset,
        max_text_length=MAX_TEXT_LENGTH,
        max_music_length=MAX_MUSIC_LENGTH
    ):
        """
        Khởi tạo Dataset.

        Args:
            dataset:
                Danh sách text/music samples.

            max_text_length:
                Độ dài text tối đa.

            max_music_length:
                Độ dài music tối đa.
        """

        self.dataset = dataset

        self.max_text_length = (
            max_text_length
        )

        self.max_music_length = (
            max_music_length
        )

    def __len__(
        self
    ):

        return len(
            self.dataset
        )

    def __getitem__(
        self,
        index
    ):

        item = self.dataset[
            index
        ]

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        text_ids = pad_sequence(
            item["text_ids"],
            self.max_text_length,
            PAD_ID
        )

        # ----------------------------------------------------
        # Music
        # ----------------------------------------------------

        music_ids = pad_sequence(
            item["music_tokens"],
            self.max_music_length,
            0
        )

        return {

            "text_ids": torch.tensor(
                text_ids,
                dtype=torch.long
            ),

            "music_ids": torch.tensor(
                music_ids,
                dtype=torch.long
            ),

            "text": item[
                "text"
            ],

            "music_file": item[
                "music_file"
            ]
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================

def create_collate_fn():
    """
    Tạo collate function cho DataLoader.

    Returns:
        function
    """

    def collate_fn(
        batch
    ):

        text_ids = torch.stack(
            [
                item["text_ids"]
                for item in batch
            ]
        )

        music_ids = torch.stack(
            [
                item["music_ids"]
                for item in batch
            ]
        )

        texts = [
            item["text"]
            for item in batch
        ]

        music_files = [
            item["music_file"]
            for item in batch
        ]

        return {

            "text_ids": text_ids,

            "music_ids": music_ids,

            "texts": texts,

            "music_files": music_files
        }

    return collate_fn


# ============================================================
# CREATE DATALOADER
# ============================================================

def create_dataloader(
    dataset,
    batch_size=8,
    shuffle=True
):
    """
    Tạo PyTorch DataLoader.

    Args:
        dataset:
            TextMusicDataset.

        batch_size:
            Batch size.

        shuffle:
            Có shuffle hay không.

    Returns:
        DataLoader
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=create_collate_fn()
    )


# ============================================================
# DATASET STATISTICS
# ============================================================

def print_dataset_statistics(
    pairs,
    vocabulary
):
    """
    In thống kê dataset.

    Args:
        pairs:
            Text/music pairs.

        vocabulary:
            Text vocabulary.
    """

    print()
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    print(
        f"Text vocabulary size : "
        f"{len(vocabulary)}"
    )

    print(
        f"Training pairs       : "
        f"{len(pairs)}"
    )

    if not pairs:

        print(
            "Không có training pair."
        )

        return

    # --------------------------------------------------------
    # Text length
    # --------------------------------------------------------

    text_lengths = [
        len(item["text_ids"])
        for item in pairs
    ]

    # --------------------------------------------------------
    # Music length
    # --------------------------------------------------------

    music_lengths = [
        len(item["music_tokens"])
        for item in pairs
    ]

    print(
        f"Average text length  : "
        f"{sum(text_lengths) / len(text_lengths):.2f}"
    )

    print(
        f"Average music length : "
        f"{sum(music_lengths) / len(music_lengths):.2f}"
    )

    print(
        f"Max text length      : "
        f"{max(text_lengths)}"
    )

    print(
        f"Max music length     : "
        f"{max(music_lengths)}"
    )


# ============================================================
# PRINT SAMPLE
# ============================================================

def print_samples(
    pairs,
    number=5
):
    """
    In một số sample để kiểm tra.

    Args:
        pairs:
            Training pairs.

        number:
            Số sample muốn in.
    """

    print()
    print("=" * 70)
    print("DATASET SAMPLES")
    print("=" * 70)

    if not pairs:

        print(
            "Không có sample."
        )

        return

    for index, item in enumerate(
        pairs[:number],
        start=1
    ):

        print()
        print(
            f"Sample {index}"
        )

        print(
            f"  Text: "
            f"{item['text']}"
        )

        print(
            f"  Text IDs: "
            f"{item['text_ids']}"
        )

        print(
            f"  Music: "
            f"{item['music_file']}"
        )

        print(
            f"  Music IDs: "
            f"{item['music_tokens'][:20]}"
            f"{' ...' if len(item['music_tokens']) > 20 else ''}"
        )


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset():
    """
    Xây dựng toàn bộ Text → Music dataset.

    Pipeline:

        music_tokens.json
                +
        music_vocab.json
                +
        caption_dataset.json
                ↓
        text vocabulary
                ↓
        text/music pairs
                ↓
        JSON output
    """

    print("=" * 70)
    print("BUILD TEXT → MUSIC DATASET")
    print("=" * 70)

    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print("[1/6] Loading music tokens...")

    music_tokens_raw = load_json(
        MUSIC_TOKEN_FILE
    )

    music_tokens = (
        normalize_music_tokens(
            music_tokens_raw
        )
    )

    print(
        f"Music files: "
        f"{len(music_tokens)}"
    )

    # ========================================================
    # STEP 2
    # ========================================================

    print()
    print("[2/6] Loading music vocabulary...")

    music_vocabulary = load_json(
        MUSIC_VOCAB_FILE
    )

    print(
        f"Music vocabulary size: "
        f"{len(music_vocabulary)}"
    )

    # ========================================================
    # STEP 3
    # ========================================================

    print()
    print("[3/6] Loading caption dataset...")

    caption_dataset = load_json(
        CAPTION_DATASET_FILE
    )

    print(
        f"Caption records: "
        f"{len(caption_dataset)}"
    )

    # ========================================================
    # STEP 4
    # ========================================================

    print()
    print("[4/6] Building text vocabulary...")

    all_captions = []

    for item in caption_dataset:

        captions = item.get(
            "captions",
            []
        )

        for caption in captions:

            normalized = normalize_text(
                caption
            )

            if normalized:

                all_captions.append(
                    normalized
                )

    print(
        f"Total captions: "
        f"{len(all_captions)}"
    )

    text_vocabulary = (
        build_text_vocabulary(
            all_captions
        )
    )

    print(
        f"Text vocabulary size: "
        f"{len(text_vocabulary)}"
    )

    # ========================================================
    # STEP 5
    # ========================================================

    print()
    print(
        "[5/6] Creating Text → Music pairs..."
    )

    pairs = create_text_music_pairs(
        caption_dataset=caption_dataset,
        music_tokens=music_tokens,
        music_vocabulary=music_vocabulary,
        text_vocabulary=text_vocabulary
    )

    print()
    print(
        f"Total training pairs: "
        f"{len(pairs)}"
    )

    # ========================================================
    # STEP 6
    # ========================================================

    print()
    print("[6/6] Saving output files...")

    save_json(
        text_vocabulary,
        TEXT_VOCAB_FILE
    )

    save_json(
        pairs,
        TEXT_MUSIC_DATASET_FILE
    )

    # ========================================================
    # Statistics
    # ========================================================

    print_dataset_statistics(
        pairs,
        text_vocabulary
    )

    print_samples(
        pairs,
        number=5
    )

    print()
    print("=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Text vocabulary:"
    )

    print(
        f"    {TEXT_VOCAB_FILE}"
    )

    print(
        f"Text-Music dataset:"
    )

    print(
        f"    {TEXT_MUSIC_DATASET_FILE}"
    )

    return (
        text_vocabulary,
        pairs
    )


# ============================================================
# TEST DATALOADER
# ============================================================

def test_dataloader(
    pairs
):
    """
    Kiểm tra dataset bằng PyTorch DataLoader.

    Args:
        pairs:
            Training pairs.
    """

    print()
    print("=" * 70)
    print("TEST PYTORCH DATALOADER")
    print("=" * 70)

    if not pairs:

        print(
            "Không có sample để test."
        )

        return

    dataset = TextMusicDataset(
        pairs
    )

    print(
        f"Dataset size: "
        f"{len(dataset)}"
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    batch = next(
        iter(dataloader)
    )

    print()
    print(
        "Batch information:"
    )

    print(
        f"  text_ids shape : "
        f"{batch['text_ids'].shape}"
    )

    print(
        f"  music_ids shape: "
        f"{batch['music_ids'].shape}"
    )

    print()

    for index in range(
        min(
            len(batch["texts"]),
            4
        )
    ):

        print(
            f"Sample {index + 1}:"
        )

        print(
            f"  Text: "
            f"{batch['texts'][index]}"
        )

        print(
            f"  Music: "
            f"{batch['music_files'][index]}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main entry point.
    """

    random.seed(
        RANDOM_SEED
    )

    torch.manual_seed(
        RANDOM_SEED
    )

    try:

        vocabulary, pairs = (
            build_dataset()
        )

        test_dataloader(
            pairs
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
