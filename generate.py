"""
generate.py

Sinh nhạc từ prompt văn bản bằng Text-to-Music Transformer.

Pipeline:

    Text Prompt
        |
        v
    Text Tokenizer
        |
        v
    Text IDs
        |
        v
    Text Encoder
        |
        v
    Music Decoder
        |
        v
    Music Token IDs
        |
        v
    Music Tokens
        |
        v
    MIDI
        |
        v
    generated.mid


Ví dụ:

    Prompt:
        Tôi muốn tạo một bản nhạc vui vẻ và sôi động

    Output:

        generated.mid
"""

import json
import os
import random

import numpy as np
import torch
import pretty_midi

from model import (
    TextToMusicTransformer,
    MUSIC_BOS_ID,
    MUSIC_EOS_ID,
    MUSIC_PAD_ID,
    MAX_TEXT_LENGTH,
    MAX_MUSIC_LENGTH,
)


# ============================================================
# CONFIG
# ============================================================

PROCESSED_DIR = "./processed_data"
CHECKPOINT_DIR = "./checkpoints"

TEXT_VOCAB_FILE = os.path.join(
    PROCESSED_DIR,
    "text_vocab.json"
)

MUSIC_VOCAB_FILE = os.path.join(
    PROCESSED_DIR,
    "music_vocab.json"
)

BEST_MODEL_FILE = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pt"
)

OUTPUT_MIDI_FILE = "./generated.mid"


# ============================================================
# GENERATION CONFIG
# ============================================================

# Độ dài tối đa chuỗi Music Token được sinh.
MAX_GENERATION_LENGTH = MAX_MUSIC_LENGTH

# Temperature:
#
# < 1.0:
#     sinh bảo thủ hơn
#
# = 1.0:
#     bình thường
#
# > 1.0:
#     đa dạng hơn
#
TEMPERATURE = 0.9


# Top-K sampling.
#
# Chỉ xét K token có xác suất cao nhất.
#
# Đặt 0 để tắt.
TOP_K = 50


# Top-P / Nucleus Sampling.
#
# Chỉ giữ các token có tổng xác suất đạt P.
#
TOP_P = 0.95


# Seed để kết quả có thể tái lập.
#
# Nếu muốn mỗi lần sinh nhạc khác nhau,
# có thể đặt thành None.
SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):
    """
    Thiết lập random seed.

    Parameters
    ----------
    seed : int | None
        Seed.
    """

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(file_path):
    """
    Đọc JSON.

    Parameters
    ----------
    file_path : str
        Đường dẫn file JSON.

    Returns
    -------
    object
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
# LOAD VOCABULARY
# ============================================================

def load_vocabularies():
    """
    Load text vocabulary và music vocabulary.

    Returns
    -------
    text_vocab : dict
        Token -> ID

    music_vocab : dict
        Token -> ID

    text_id_to_token : dict
        ID -> Token

    music_id_to_token : dict
        ID -> Token
    """

    print()
    print("[1] Loading vocabulary...")

    text_vocab = load_json(
        TEXT_VOCAB_FILE
    )

    music_vocab = load_json(
        MUSIC_VOCAB_FILE
    )

    text_id_to_token = {
        int(index): token
        for token, index
        in text_vocab.items()
    }

    music_id_to_token = {
        int(index): token
        for token, index
        in music_vocab.items()
    }

    print(
        f"    Text vocabulary : {len(text_vocab)}"
    )

    print(
        f"    Music vocabulary: {len(music_vocab)}"
    )

    return (
        text_vocab,
        music_vocab,
        text_id_to_token,
        music_id_to_token,
    )


# ============================================================
# TEXT TOKENIZATION
# ============================================================

def tokenize_text(
    text,
    text_vocab,
):
    """
    Chuyển prompt thành Text Token IDs.

    Format:

        <BOS>
        token_1
        token_2
        ...
        <EOS>

    Các từ không tồn tại trong vocabulary
    sẽ được chuyển thành <UNK>.

    Parameters
    ----------
    text : str
        Prompt người dùng.

    text_vocab : dict
        Text vocabulary.

    Returns
    -------
    list[int]
        Text token IDs.
    """

    # --------------------------------------------------------
    # Chuẩn hóa text
    # --------------------------------------------------------

    text = text.strip().lower()

    # --------------------------------------------------------
    # Lấy special token IDs
    # --------------------------------------------------------

    pad_id = text_vocab.get(
        "<PAD>",
        0
    )

    unk_id = text_vocab.get(
        "<UNK>",
        1
    )

    bos_id = text_vocab.get(
        "<BOS>",
        2
    )

    eos_id = text_vocab.get(
        "<EOS>",
        3
    )

    # --------------------------------------------------------
    # Tokenize đơn giản bằng whitespace
    # --------------------------------------------------------

    words = text.split()

    token_ids = [
        bos_id
    ]

    for word in words:

        token_id = text_vocab.get(
            word,
            unk_id
        )

        token_ids.append(
            token_id
        )

    token_ids.append(
        eos_id
    )

    # --------------------------------------------------------
    # Truncate
    # --------------------------------------------------------

    token_ids = token_ids[
        :MAX_TEXT_LENGTH
    ]

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    if len(token_ids) < MAX_TEXT_LENGTH:

        token_ids.extend(
            [pad_id]
            * (
                MAX_TEXT_LENGTH
                - len(token_ids)
            )
        )

    return token_ids


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    text_vocab,
    music_vocab,
):
    """
    Load Text-to-Music Transformer
    và checkpoint tốt nhất.

    Parameters
    ----------
    text_vocab : dict
        Text vocabulary.

    music_vocab : dict
        Music vocabulary.

    Returns
    -------
    TextToMusicTransformer
        Model đã load checkpoint.
    """

    print()
    print("[2] Loading model...")

    if not os.path.exists(
        BEST_MODEL_FILE
    ):

        raise FileNotFoundError(
            "Không tìm thấy checkpoint:\n"
            f"{BEST_MODEL_FILE}\n\n"
            "Hãy train model trước."
        )

    # --------------------------------------------------------
    # Tạo model
    # --------------------------------------------------------

    model = TextToMusicTransformer(
        text_vocab_size=len(text_vocab),
        music_vocab_size=len(music_vocab),
    )

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        BEST_MODEL_FILE,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Hỗ trợ nhiều format checkpoint
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            # Trường hợp checkpoint chính là state_dict.
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        f"    Checkpoint: {BEST_MODEL_FILE}"
    )

    print(
        f"    Device    : {DEVICE}"
    )

    return model


# ============================================================
# TOP-K FILTER
# ============================================================

def top_k_filter(
    logits,
    k,
):
    """
    Giữ lại K token có logits cao nhất.

    Parameters
    ----------
    logits : torch.Tensor
        Shape [vocab_size]

    k : int
        Số token giữ lại.

    Returns
    -------
    torch.Tensor
        Logits sau filtering.
    """

    if k <= 0:
        return logits

    k = min(
        k,
        logits.size(-1)
    )

    values, _ = torch.topk(
        logits,
        k
    )

    threshold = values[
        -1
    ]

    filtered_logits = logits.clone()

    filtered_logits[
        filtered_logits < threshold
    ] = float("-inf")

    return filtered_logits


# ============================================================
# TOP-P FILTER
# ============================================================

def top_p_filter(
    logits,
    p,
):
    """
    Nucleus / Top-P sampling.

    Chỉ giữ các token nằm trong vùng
    xác suất tích lũy <= P.

    Parameters
    ----------
    logits : torch.Tensor
        Shape [vocab_size]

    p : float
        Ngưỡng cumulative probability.

    Returns
    -------
    torch.Tensor
        Logits sau filtering.
    """

    if p <= 0.0 or p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True
    )

    sorted_probabilities = torch.softmax(
        sorted_logits,
        dim=-1
    )

    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1
    )

    # Token đầu tiên vượt P cũng được giữ lại.
    sorted_mask = (
        cumulative_probabilities > p
    )

    sorted_mask[1:] = sorted_mask[:-1].clone()

    sorted_mask[0] = False

    indices_to_remove = sorted_indices[
        sorted_mask
    ]

    filtered_logits = logits.clone()

    filtered_logits[
        indices_to_remove
    ] = float("-inf")

    return filtered_logits


# ============================================================
# SAMPLE NEXT TOKEN
# ============================================================

def sample_next_token(
    logits,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
):
    """
    Sinh token tiếp theo bằng sampling.

    Pipeline:

        logits
          ↓
        temperature
          ↓
        top-k
          ↓
        top-p
          ↓
        softmax
          ↓
        multinomial sampling

    Parameters
    ----------
    logits : torch.Tensor
        Shape [vocab_size]

    temperature : float
        Temperature.

    top_k : int
        Top-K.

    top_p : float
        Top-P.

    Returns
    -------
    int
        Token ID tiếp theo.
    """

    if temperature <= 0:
        raise ValueError(
            "temperature phải > 0"
        )

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    logits = logits / temperature

    # --------------------------------------------------------
    # Top-K
    # --------------------------------------------------------

    logits = top_k_filter(
        logits,
        top_k
    )

    # --------------------------------------------------------
    # Top-P
    # --------------------------------------------------------

    logits = top_p_filter(
        logits,
        top_p
    )

    # --------------------------------------------------------
    # Softmax
    # --------------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    next_token = torch.multinomial(
        probabilities,
        num_samples=1
    )

    return int(
        next_token.item()
    )


# ============================================================
# GENERATE MUSIC TOKENS
# ============================================================

@torch.no_grad()
def generate_music_tokens(
    model,
    text_ids,
    max_length=MAX_GENERATION_LENGTH,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
):
    """
    Sinh Music Token autoregressively.

    Parameters
    ----------
    model : TextToMusicTransformer
        Model đã train.

    text_ids : list[int]
        Text token IDs.

    max_length : int
        Độ dài tối đa.

    temperature : float
        Temperature sampling.

    top_k : int
        Top-K sampling.

    top_p : float
        Top-P sampling.

    Returns
    -------
    list[int]
        Music token IDs.
    """

    # --------------------------------------------------------
    # Text input
    # --------------------------------------------------------

    text_tensor = torch.tensor(
        [text_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Encode text
    # --------------------------------------------------------

    memory, text_padding_mask = model.encode(
        text_tensor
    )

    # --------------------------------------------------------
    # Bắt đầu bằng BOS
    # --------------------------------------------------------

    generated = [
        MUSIC_BOS_ID
    ]

    print()
    print("[4] Generating music tokens...")

    print(
        f"    Max length : {max_length}"
    )

    print(
        f"    Temperature: {temperature}"
    )

    print(
        f"    Top-K      : {top_k}"
    )

    print(
        f"    Top-P      : {top_p}"
    )

    # --------------------------------------------------------
    # Autoregressive generation
    # --------------------------------------------------------

    for step in range(
        max_length - 1
    ):

        music_tensor = torch.tensor(
            [generated],
            dtype=torch.long,
            device=DEVICE,
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        logits = model.decode(
            music_ids=music_tensor,
            memory=memory,
            text_padding_mask=text_padding_mask,
        )

        # Lấy logits tại token cuối.
        next_token_logits = logits[
            0,
            -1,
            :
        ]

        # ----------------------------------------------------
        # Không cho sinh PAD
        # ----------------------------------------------------

        next_token_logits[
            MUSIC_PAD_ID
        ] = float("-inf")

        # ----------------------------------------------------
        # Sample
        # --------------------------------------------------------

        next_token_id = sample_next_token(
            logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        generated.append(
            next_token_id
        )

        # ----------------------------------------------------
        # EOS
        # ----------------------------------------------------

        if next_token_id == MUSIC_EOS_ID:

            print(
                f"    EOS generated at step {step + 1}"
            )

            break

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (step + 1) % 100 == 0
        ):

            print(
                f"    Generated: "
                f"{step + 1} tokens"
            )

    print(
        f"    Total tokens: {len(generated)}"
    )

    return generated


# ============================================================
# MUSIC IDs -> MUSIC TOKENS
# ============================================================

def ids_to_music_tokens(
    music_ids,
    music_id_to_token,
):
    """
    Chuyển Music Token IDs thành Music Token.

    Parameters
    ----------
    music_ids : list[int]
        Music token IDs.

    music_id_to_token : dict[int, str]
        Mapping ID -> token.

    Returns
    -------
    list[str]
        Music tokens.
    """

    tokens = []

    for token_id in music_ids:

        token = music_id_to_token.get(
            int(token_id),
            "UNK"
        )

        tokens.append(
            token
        )

    return tokens


# ============================================================
# MUSIC TOKENS -> MIDI
# ============================================================

def music_tokens_to_midi(
    tokens,
    output_file,
):
    """
    Chuyển Music Token thành MIDI.

    Format token từ midi_preprocess.py:

        TIME_SHIFT_n
        NOTE_ON_pitch
        VELOCITY_velocity
        DURATION_n

    Ví dụ:

        NOTE_ON_60
        VELOCITY_80
        DURATION_4
        TIME_SHIFT_2
        NOTE_ON_64
        VELOCITY_90
        DURATION_4

    Quy đổi:

        1 step = TIME_RESOLUTION = 0.125 giây

    MIDI output:

        120 BPM
        Piano
        Acoustic Grand Piano
    """

    print()
    print("[5] Converting Music Tokens -> MIDI...")

    # --------------------------------------------------------
    # MIDI
    # --------------------------------------------------------

    midi = pretty_midi.PrettyMIDI(
        initial_tempo=120.0
    )

    # --------------------------------------------------------
    # Piano instrument
    # --------------------------------------------------------

    instrument = pretty_midi.Instrument(
        program=0,
        name="Generated Piano"
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    current_time = 0.0

    pending_pitch = None
    pending_velocity = 80

    created_notes = 0

    TIME_RESOLUTION = 0.125

    # --------------------------------------------------------
    # Parse tokens
    # --------------------------------------------------------

    for token in tokens:

        # ----------------------------------------------------
        # Special tokens
        # ----------------------------------------------------

        if token == "BOS":
            continue

        if token == "EOS":
            break

        if token == "PAD":
            continue

        if token == "UNK":
            continue

        # ----------------------------------------------------
        # TIME SHIFT
        # ----------------------------------------------------

        if token.startswith(
            "TIME_SHIFT_"
        ):

            try:

                steps = int(
                    token.split(
                        "_"
                    )[-1]
                )

                if steps > 0:

                    current_time += (
                        steps
                        * TIME_RESOLUTION
                    )

            except ValueError:

                continue

        # ----------------------------------------------------
        # NOTE ON
        # ----------------------------------------------------

        elif token.startswith(
            "NOTE_ON_"
        ):

            try:

                pitch = int(
                    token.split(
                        "_"
                    )[-1]
                )

                if (
                    0
                    <= pitch
                    <= 127
                ):

                    pending_pitch = pitch

            except ValueError:

                continue

        # ----------------------------------------------------
        # VELOCITY
        # ----------------------------------------------------

        elif token.startswith(
            "VELOCITY_"
        ):

            try:

                velocity = int(
                    token.split(
                        "_"
                    )[-1]
                )

                velocity = max(
                    0,
                    min(
                        127,
                        velocity
                    )
                )

                pending_velocity = velocity

            except ValueError:

                continue

        # ----------------------------------------------------
        # DURATION
        # ----------------------------------------------------

        elif token.startswith(
            "DURATION_"
        ):

            try:

                duration_steps = int(
                    token.split(
                        "_"
                    )[-1]
                )

                if (
                    pending_pitch is None
                    or duration_steps <= 0
                ):
                    continue

                duration = (
                    duration_steps
                    * TIME_RESOLUTION
                )

                # --------------------------------------------
                # Không cho note vượt quá giới hạn
                # --------------------------------------------

                start = current_time

                end = (
                    start
                    + duration
                )

                # --------------------------------------------
                # Tạo MIDI note
                # --------------------------------------------

                note = pretty_midi.Note(
                    velocity=pending_velocity,
                    pitch=pending_pitch,
                    start=start,
                    end=end,
                )

                instrument.notes.append(
                    note
                )

                created_notes += 1

                # --------------------------------------------
                # Reset pending note
                # --------------------------------------------

                pending_pitch = None
                pending_velocity = 80

            except ValueError:

                continue

    # --------------------------------------------------------
    # Nếu không có note
    # --------------------------------------------------------

    if created_notes == 0:

        raise RuntimeError(
            "Không tạo được note nào từ Music Token.\n"
            "Model có thể chưa học tốt hoặc sequence "
            "được sinh ra không hợp lệ."
        )

    # --------------------------------------------------------
    # Add instrument
    # --------------------------------------------------------

    midi.instruments.append(
        instrument
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_directory = os.path.dirname(
        output_file
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Write MIDI
    # --------------------------------------------------------

    midi.write(
        output_file
    )

    print(
        f"    Created notes: {created_notes}"
    )

    print(
        f"    MIDI file     : {output_file}"
    )


# ============================================================
# PRINT GENERATED TOKENS
# ============================================================

def print_music_tokens(
    tokens,
    max_tokens=100,
):
    """
    In một phần Music Token để kiểm tra.
    """

    print()
    print("[6] Generated Music Tokens:")
    print("-" * 70)

    display_tokens = tokens[
        :max_tokens
    ]

    for index, token in enumerate(
        display_tokens
    ):

        print(
            f"{index:4d}: {token}"
        )

    if len(tokens) > max_tokens:

        print(
            "..."
        )

        print(
            f"({len(tokens) - max_tokens} "
            f"tokens khác)"
        )

    print("-" * 70)


# ============================================================
# MAIN GENERATION PIPELINE
# ============================================================

def generate_from_prompt(
    prompt,
    output_file=OUTPUT_MIDI_FILE,
):
    """
    Pipeline hoàn chỉnh:

        Prompt
          ↓
        Text IDs
          ↓
        Transformer
          ↓
        Music IDs
          ↓
        Music Tokens
          ↓
        MIDI
    """

    print()
    print("=" * 70)
    print("MUSIC GENAI - TEXT TO MUSIC")
    print("=" * 70)

    print()
    print("Prompt:")
    print(
        f"  {prompt}"
    )

    # --------------------------------------------------------
    # Load vocab
    # --------------------------------------------------------

    (
        text_vocab,
        music_vocab,
        text_id_to_token,
        music_id_to_token,
    ) = load_vocabularies()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        text_vocab=text_vocab,
        music_vocab=music_vocab,
    )

    # --------------------------------------------------------
    # Tokenize prompt
    # --------------------------------------------------------

    print()
    print("[3] Tokenizing prompt...")

    text_ids = tokenize_text(
        prompt,
        text_vocab,
    )

    # Chỉ hiển thị token thực sự trước PAD.
    pad_id = text_vocab.get(
        "<PAD>",
        0
    )

    visible_text_ids = [
        token_id
        for token_id in text_ids
        if token_id != pad_id
    ]

    print(
        f"    Text IDs: {visible_text_ids}"
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    music_ids = generate_music_tokens(
        model=model,
        text_ids=text_ids,
        max_length=MAX_GENERATION_LENGTH,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )

    # --------------------------------------------------------
    # IDs -> Tokens
    # --------------------------------------------------------

    music_tokens = ids_to_music_tokens(
        music_ids,
        music_id_to_token,
    )

    # --------------------------------------------------------
    # Print tokens
    # --------------------------------------------------------

    print_music_tokens(
        music_tokens
    )

    # --------------------------------------------------------
    # Tokens -> MIDI
    # --------------------------------------------------------

    music_tokens_to_midi(
        tokens=music_tokens,
        output_file=output_file,
    )

    print()
    print("=" * 70)
    print("GENERATION HOÀN TẤT")
    print("=" * 70)

    print()
    print(
        f"Prompt : {prompt}"
    )

    print(
        f"Output : {output_file}"
    )

    print(
        f"Tokens : {len(music_tokens)}"
    )

    print("=" * 70)


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode():
    """
    Chế độ nhập prompt từ terminal.
    """

    print()
    print("=" * 70)
    print("TEXT-TO-MUSIC GENERATOR")
    print("=" * 70)

    print()
    print(
        "Nhập prompt mô tả bản nhạc."
    )

    print(
        "Ví dụ:"
    )

    print(
        "  Tôi muốn tạo một bản nhạc vui vẻ và sôi động"
    )

    print()
    print(
        "Gõ 'exit' để thoát."
    )

    print("=" * 70)

    while True:

        print()

        prompt = input(
            "Nhập prompt: "
        ).strip()

        if prompt.lower() == "exit":

            print(
                "Đã thoát."
            )

            break

        if not prompt:

            print(
                "Prompt không được để trống."
            )

            continue

        try:

            generate_from_prompt(
                prompt=prompt,
                output_file=OUTPUT_MIDI_FILE,
            )

        except Exception as error:

            print()
            print(
                "[ERROR]"
            )

            print(
                str(error)
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Seed.
    set_seed(
        SEED
    )

    # Chạy interactive mode.
    interactive_mode()
