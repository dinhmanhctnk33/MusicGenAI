"""
midi_preprocess.py

Tiền xử lý dữ liệu MIDI cho hệ thống Music GenAI.

Pipeline:

    dataset_clean/
        |
        |-- song_001.midi
        |-- song_002.midi
        |-- song_003.midi
        |
        v
    MIDI Preprocessing
        |
        v
    Music Tokens
        |
        v
    Token IDs
        |
        +----------------------+
        |                      |
        v                      v
    music_vocab.json     music_tokens.json

Nhiệm vụ chính:

    1. Tìm tất cả file MIDI trong dataset_clean/.
    2. Đọc và kiểm tra MIDI bằng pretty_midi.
    3. Chuyển các note MIDI thành Music Token.
    4. Xây dựng vocabulary.
    5. Chuyển Music Token thành integer ID.
    6. Lưu dữ liệu đã xử lý vào processed_data/.

Định dạng MIDI được hỗ trợ:

    .mid
    .midi
"""

import json
import os
from collections import Counter

import pretty_midi
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

# Thư mục chứa MIDI đã được làm sạch.
INPUT_DIR = "./dataset_clean"

# Thư mục lưu dữ liệu sau preprocessing.
OUTPUT_DIR = "./processed_data"

# File vocabulary.
VOCAB_FILE = os.path.join(
    OUTPUT_DIR,
    "music_vocab.json"
)

# File token data.
TOKEN_FILE = os.path.join(
    OUTPUT_DIR,
    "music_tokens.json"
)


# ============================================================
# SPECIAL TOKENS
# ============================================================

"""
Các token đặc biệt.

PAD:
    Padding sequence.

BOS:
    Beginning Of Sequence.

EOS:
    End Of Sequence.

UNK:
    Unknown token.
"""

SPECIAL_TOKENS = [
    "PAD",
    "BOS",
    "EOS",
    "UNK"
]


# ============================================================
# MIDI CONFIGURATION
# ============================================================

"""
Độ phân giải thời gian.

Ví dụ:

    TIME_RESOLUTION = 0.125

    0.125 giây -> 1 time step
    0.250 giây -> 2 time steps
    0.500 giây -> 4 time steps
"""

TIME_RESOLUTION = 0.125


"""
Khoảng velocity hợp lệ của MIDI.
"""

MIN_VELOCITY = 0
MAX_VELOCITY = 127


"""
Khoảng pitch hợp lệ của MIDI.
"""

MIN_PITCH = 0
MAX_PITCH = 127


# ============================================================
# FIND MIDI FILES
# ============================================================

def find_midi_files(directory):
    """
    Tìm tất cả file MIDI trong một thư mục.

    Hàm duyệt đệ quy toàn bộ thư mục con.

    Các định dạng được hỗ trợ:

        .mid
        .midi

    Parameters
    ----------
    directory : str
        Đường dẫn tới thư mục chứa dataset MIDI.

    Returns
    -------
    list[str]
        Danh sách đường dẫn tới các file MIDI.

    Raises
    ------
    FileNotFoundError
        Nếu thư mục không tồn tại.
    """

    midi_files = []

    if not os.path.exists(directory):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục: {directory}"
        )

    for root, _, files in os.walk(directory):

        for file_name in files:

            # Hỗ trợ cả .mid và .midi.
            if file_name.lower().endswith(
                (".mid", ".midi")
            ):

                file_path = os.path.join(
                    root,
                    file_name
                )

                midi_files.append(
                    file_path
                )

    return sorted(midi_files)


# ============================================================
# TIME QUANTIZATION
# ============================================================

def quantize_time(time):
    """
    Lượng tử hóa thời gian MIDI.

    MIDI sử dụng thời gian dạng float, ví dụ:

        0.123
        0.251
        0.500

    Transformer làm việc thuận tiện hơn với các giá trị
    rời rạc nên thời gian được chuyển thành time step.

    Parameters
    ----------
    time : float
        Thời gian tính bằng giây.

    Returns
    -------
    int
        Số time step.

    Notes
    -----
    Nếu time <= 0 thì trả về 0.

    Điều này rất quan trọng đối với các note bắt đầu
    cùng một thời điểm.
    """

    if time <= 0:
        return 0

    return max(
        1,
        round(
            time / TIME_RESOLUTION
        )
    )


# ============================================================
# NORMALIZE VELOCITY
# ============================================================

def normalize_velocity(velocity):
    """
    Đưa velocity MIDI về khoảng 0-127.

    Parameters
    ----------
    velocity : int
        Giá trị velocity ban đầu.

    Returns
    -------
    int
        Velocity sau khi được giới hạn trong khoảng 0-127.
    """

    return max(
        MIN_VELOCITY,
        min(
            MAX_VELOCITY,
            int(velocity)
        )
    )


# ============================================================
# MIDI -> EVENTS
# ============================================================

def midi_to_events(file_path):
    """
    Chuyển một file MIDI thành danh sách Music Token.

    Mỗi note được biểu diễn bằng:

        NOTE_ON_pitch
        VELOCITY_velocity
        DURATION_steps

    Giữa các note có TIME_SHIFT nếu cần.

    Ví dụ:

        NOTE_ON_60
        VELOCITY_80
        DURATION_4

    Parameters
    ----------
    file_path : str
        Đường dẫn tới file MIDI.

    Returns
    -------
    list[str]
        Danh sách Music Token.

    Notes
    -----
    Drum track được bỏ qua trong phiên bản đầu tiên.

    Tất cả melodic notes từ tất cả instrument sẽ được
    gom lại thành một sequence.

    Raises
    ------
    Exception
        Nếu file MIDI không thể đọc.
    """

    # --------------------------------------------------------
    # Đọc MIDI
    # --------------------------------------------------------

    midi = pretty_midi.PrettyMIDI(
        file_path
    )

    # Danh sách tất cả note.
    all_notes = []

    # --------------------------------------------------------
    # Thu thập note
    # --------------------------------------------------------

    for instrument in midi.instruments:

        # Bỏ qua drum.
        if instrument.is_drum:
            continue

        for note in instrument.notes:

            all_notes.append(
                note
            )

    # MIDI không có melodic note.
    if not all_notes:
        return []

    # --------------------------------------------------------
    # Sắp xếp note
    # --------------------------------------------------------

    all_notes.sort(
        key=lambda note: (
            note.start,
            note.pitch,
            note.end
        )
    )

    # --------------------------------------------------------
    # Convert note -> token
    # --------------------------------------------------------

    events = []

    previous_start = 0.0

    for note in all_notes:

        pitch = int(
            note.pitch
        )

        velocity = normalize_velocity(
            note.velocity
        )

        start = float(
            note.start
        )

        end = float(
            note.end
        )

        duration = end - start

        # ----------------------------------------------------
        # Validate note
        # ----------------------------------------------------

        if pitch < MIN_PITCH:
            continue

        if pitch > MAX_PITCH:
            continue

        if duration <= 0:
            continue

        # ----------------------------------------------------
        # TIME SHIFT
        # ----------------------------------------------------

        time_delta = (
            start - previous_start
        )

        time_steps = quantize_time(
            time_delta
        )

        if time_steps > 0:

            events.append(
                f"TIME_SHIFT_{time_steps}"
            )

        # ----------------------------------------------------
        # NOTE ON
        # ----------------------------------------------------

        events.append(
            f"NOTE_ON_{pitch}"
        )

        # ----------------------------------------------------
        # VELOCITY
        # ----------------------------------------------------

        events.append(
            f"VELOCITY_{velocity}"
        )

        # ----------------------------------------------------
        # DURATION
        # ----------------------------------------------------

        duration_steps = quantize_time(
            duration
        )

        if duration_steps <= 0:
            duration_steps = 1

        events.append(
            f"DURATION_{duration_steps}"
        )

        # Cập nhật thời điểm note hiện tại.
        previous_start = start

    return events


# ============================================================
# BUILD VOCABULARY
# ============================================================

def build_vocabulary(
    token_sequences,
    min_frequency=1
):
    """
    Xây dựng vocabulary từ các Music Token.

    Parameters
    ----------
    token_sequences : list[list[str]]
        Danh sách các sequence Music Token.

    min_frequency : int, optional
        Tần suất xuất hiện tối thiểu của token.

    Returns
    -------
    dict[str, int]
        Vocabulary dạng:

            token -> integer ID
    """

    counter = Counter()

    # Đếm frequency của token.
    for sequence in token_sequences:

        counter.update(
            sequence
        )

    vocabulary = {}

    # --------------------------------------------------------
    # Special tokens
    # --------------------------------------------------------

    for token in SPECIAL_TOKENS:

        vocabulary[token] = len(
            vocabulary
        )

    # --------------------------------------------------------
    # Music tokens
    # --------------------------------------------------------

    sorted_tokens = sorted(
        counter.items(),
        key=lambda item: (
            -item[1],
            item[0]
        )
    )

    for token, frequency in sorted_tokens:

        if frequency < min_frequency:
            continue

        if token in vocabulary:
            continue

        vocabulary[token] = len(
            vocabulary
        )

    return vocabulary


# ============================================================
# TOKEN -> ID
# ============================================================

def tokens_to_ids(
    tokens,
    vocabulary
):
    """
    Chuyển Music Token thành token ID.

    Parameters
    ----------
    tokens : list[str]
        Danh sách Music Token.

    vocabulary : dict[str, int]
        Vocabulary.

    Returns
    -------
    list[int]
        Danh sách token ID.

    Notes
    -----
    Nếu token không tồn tại trong vocabulary,
    token sẽ được chuyển thành UNK.
    """

    unk_id = vocabulary["UNK"]

    return [
        vocabulary.get(
            token,
            unk_id
        )
        for token in tokens
    ]


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    data,
    file_path
):
    """
    Lưu dữ liệu dưới dạng JSON.

    Parameters
    ----------
    data : object
        Dữ liệu cần lưu.

    file_path : str
        Đường dẫn file JSON.
    """

    directory = os.path.dirname(
        file_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset():
    """
    Xử lý toàn bộ dataset MIDI.

    Pipeline:

        dataset_clean/
            |
            v
        Find MIDI
            |
            v
        MIDI -> Events
            |
            v
        Build Vocabulary
            |
            v
        Events -> Token IDs
            |
            v
        Save JSON

    Returns
    -------
    None
    """

    print()
    print("=" * 70)
    print("MUSIC GENAI - MIDI PREPROCESSING")
    print("=" * 70)

    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print("[1] Đang tìm file MIDI...")
    print(
        f"    Input directory: {INPUT_DIR}"
    )

    try:

        midi_files = find_midi_files(
            INPUT_DIR
        )

    except FileNotFoundError as error:

        print(
            f"[ERROR] {error}"
        )

        return

    print(
        f"    Tìm thấy: {len(midi_files)} file MIDI"
    )

    if not midi_files:

        print()
        print(
            "[WARNING] Không tìm thấy file .mid hoặc .midi."
        )

        return

    # ========================================================
    # STEP 2
    # ========================================================

    print()
    print(
        "[2] Đang chuyển MIDI thành Music Token..."
    )

    token_sequences = []
    valid_files = []
    failed_files = []

    for file_path in tqdm(
        midi_files,
        desc="Tokenizing MIDI"
    ):

        try:

            tokens = midi_to_events(
                file_path
            )

            # MIDI không có melodic note.
            if not tokens:

                print()
                print(
                    "[WARNING] MIDI không có note:"
                )

                print(
                    f"           {file_path}"
                )

                continue

            # ------------------------------------------------
            # BOS
            # ------------------------------------------------

            tokens.insert(
                0,
                "BOS"
            )

            # ------------------------------------------------
            # EOS
            # ------------------------------------------------

            tokens.append(
                "EOS"
            )

            token_sequences.append(
                tokens
            )

            valid_files.append(
                file_path
            )

        except Exception as error:

            failed_files.append(
                {
                    "file_path": file_path,
                    "error": str(error)
                }
            )

            print()
            print(
                "[ERROR] Không thể xử lý:"
            )

            print(
                f"        {file_path}"
            )

            print(
                f"        Lý do: {error}"
            )

    # ========================================================
    # STEP 3
    # ========================================================

    print()
    print(
        "[3] Đang xây dựng vocabulary..."
    )

    vocabulary = build_vocabulary(
        token_sequences,
        min_frequency=1
    )

    print(
        f"    Vocabulary size: {len(vocabulary)}"
    )

    # ========================================================
    # STEP 4
    # ========================================================

    print()
    print(
        "[4] Đang chuyển Music Token -> Token ID..."
    )

    processed_data = []

    for file_path, tokens in zip(
        valid_files,
        token_sequences
    ):

        token_ids = tokens_to_ids(
            tokens,
            vocabulary
        )

        processed_data.append(
            {
                "file_path": file_path,
                "file_name": os.path.basename(
                    file_path
                ),
                "tokens": tokens,
                "token_ids": token_ids,
                "length": len(token_ids)
            }
        )

    # ========================================================
    # STEP 5
    # ========================================================

    print()
    print(
        "[5] Đang lưu vocabulary..."
    )

    save_json(
        vocabulary,
        VOCAB_FILE
    )

    print(
        f"    Saved: {VOCAB_FILE}"
    )

    # ========================================================
    # STEP 6
    # ========================================================

    print()
    print(
        "[6] Đang lưu Music Token..."
    )

    save_json(
        processed_data,
        TOKEN_FILE
    )

    print(
        f"    Saved: {TOKEN_FILE}"
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    lengths = [
        item["length"]
        for item in processed_data
    ]

    print()
    print("=" * 70)
    print("KẾT QUẢ MIDI PREPROCESSING")
    print("=" * 70)

    print(
        f"Số MIDI đầu vào : {len(midi_files)}"
    )

    print(
        f"Số MIDI xử lý   : {len(processed_data)}"
    )

    print(
        f"Số MIDI lỗi     : {len(failed_files)}"
    )

    print(
        f"Vocabulary size  : {len(vocabulary)}"
    )

    if lengths:

        print(
            f"Token ngắn nhất : {min(lengths)}"
        )

        print(
            f"Token dài nhất  : {max(lengths)}"
        )

        print(
            f"Token trung bình: "
            f"{sum(lengths) / len(lengths):.2f}"
        )

    # ========================================================
    # FAILED FILES
    # ========================================================

    if failed_files:

        print()
        print(
            "Các file MIDI không xử lý được:"
        )

        for item in failed_files:

            print(
                f"  - {item['file_path']}"
            )

            print(
                f"    {item['error']}"
            )

    print()
    print(
        "Vocabulary:"
    )

    print(
        f"  {VOCAB_FILE}"
    )

    print()
    print(
        "Token data:"
    )

    print(
        f"  {TOKEN_FILE}"
    )

    print()
    print("=" * 70)
    print("PREPROCESSING HOÀN TẤT")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    process_dataset()

