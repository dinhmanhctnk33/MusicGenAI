"""
music_feature_extractor.py

Trích xuất các đặc trưng âm nhạc từ tập dữ liệu MIDI.

Input:
    dataset_clean/
        ├── *.mid
        ├── *.midi
        └── ...

Output:
    processed_data/music_features.json

Các đặc trưng được trích xuất:
    - duration
    - tempo
    - note_count
    - average_velocity
    - max_velocity
    - min_velocity
    - note_density
    - pitch_min
    - pitch_max
    - pitch_range
    - average_pitch
    - instrument_count
    - instruments
    - polyphony
    - drum_present
    - time_signature
    - key_signature

Lưu ý:
    Các đặc trưng này là dữ liệu khách quan/đo được từ MIDI.
    File này KHÔNG tự kết luận bài nhạc là "vui", "buồn",
    "hùng tráng"... Việc tạo mô tả ngôn ngữ tự nhiên
    sẽ được thực hiện ở bước caption_generator.py.
"""

import json
import os
from collections import Counter
from pathlib import Path

import pretty_midi


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "./dataset_clean"
OUTPUT_DIR = "./processed_data"

OUTPUT_FEATURE_FILE = os.path.join(
    OUTPUT_DIR,
    "music_features.json"
)


# ============================================================
# MIDI FILE DISCOVERY
# ============================================================

def find_midi_files(input_dir: str):
    """
    Tìm tất cả file MIDI trong thư mục dataset.

    Hỗ trợ:
        .mid
        .midi

    Args:
        input_dir: Thư mục chứa dataset MIDI.

    Returns:
        list[str]: Danh sách đường dẫn MIDI.
    """

    midi_files = []

    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dataset: {input_dir}"
        )

    for file_path in input_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in {
            ".mid",
            ".midi"
        }:
            midi_files.append(str(file_path))

    midi_files.sort()

    return midi_files


# ============================================================
# TEMPO
# ============================================================

def extract_tempo(midi: pretty_midi.PrettyMIDI) -> float:
    """
    Lấy tempo trung bình của MIDI.

    Args:
        midi: PrettyMIDI object.

    Returns:
        float: Tempo BPM.
    """

    try:
        tempi = midi.get_tempo_changes()

        if len(tempi[1]) == 0:
            return 120.0

        return float(tempi[1][0])

    except Exception:
        return 120.0


# ============================================================
# INSTRUMENTS
# ============================================================

def extract_instruments(midi: pretty_midi.PrettyMIDI):
    """
    Trích xuất danh sách nhạc cụ.

    Args:
        midi: PrettyMIDI object.

    Returns:
        dict:
            instrument_count
            instruments
            drum_present
    """

    instruments = []
    drum_present = False

    for instrument in midi.instruments:

        if instrument.is_drum:
            drum_present = True
            instruments.append("Drums")
            continue

        instrument_name = pretty_midi.program_to_instrument_name(
            instrument.program
        )

        instruments.append(instrument_name)

    # Loại bỏ trùng lặp nhưng giữ thứ tự
    unique_instruments = list(
        dict.fromkeys(instruments)
    )

    return {
        "instrument_count": len(unique_instruments),
        "instruments": unique_instruments,
        "drum_present": drum_present
    }


# ============================================================
# NOTES
# ============================================================

def collect_notes(midi: pretty_midi.PrettyMIDI):
    """
    Thu thập toàn bộ note không phải drum.

    Args:
        midi: PrettyMIDI object.

    Returns:
        list[pretty_midi.Note]
    """

    notes = []

    for instrument in midi.instruments:

        if instrument.is_drum:
            continue

        notes.extend(instrument.notes)

    return notes


# ============================================================
# NOTE FEATURES
# ============================================================

def extract_note_features(
    notes,
    duration: float
):
    """
    Trích xuất các đặc trưng liên quan đến note.

    Args:
        notes: Danh sách PrettyMIDI Note.
        duration: Thời lượng MIDI.

    Returns:
        dict
    """

    if not notes:

        return {
            "note_count": 0,
            "average_velocity": 0.0,
            "min_velocity": 0,
            "max_velocity": 0,
            "pitch_min": 0,
            "pitch_max": 0,
            "pitch_range": 0,
            "average_pitch": 0.0,
            "note_density": 0.0,
            "average_note_duration": 0.0,
            "polyphony": 0.0
        }

    velocities = [
        note.velocity
        for note in notes
    ]

    pitches = [
        note.pitch
        for note in notes
    ]

    note_durations = [
        max(0.0, note.end - note.start)
        for note in notes
    ]

    # --------------------------------------------------------
    # Note density
    # --------------------------------------------------------

    safe_duration = max(duration, 0.001)

    note_density = (
        len(notes) / safe_duration
    )

    # --------------------------------------------------------
    # Average note duration
    # --------------------------------------------------------

    average_note_duration = (
        sum(note_durations) / len(note_durations)
    )

    # --------------------------------------------------------
    # Polyphony
    # --------------------------------------------------------
    # Tính trung bình số note đang phát cùng lúc.
    # Đây là một phép đo xấp xỉ mức độ nhiều bè của bản nhạc.

    events = []

    for note in notes:
        events.append(
            (note.start, 1)
        )
        events.append(
            (note.end, -1)
        )

    events.sort(
        key=lambda event: (
            event[0],
            event[1]
        )
    )

    active_notes = 0
    last_time = None

    weighted_active_time = 0.0
    total_time = 0.0

    for time, change in events:

        if last_time is not None:

            delta = time - last_time

            if delta > 0:

                weighted_active_time += (
                    active_notes * delta
                )

                total_time += delta

        active_notes += change
        last_time = time

    if total_time > 0:

        average_polyphony = (
            weighted_active_time / total_time
        )

    else:
        average_polyphony = 0.0

    return {
        "note_count": len(notes),

        "average_velocity": round(
            sum(velocities) / len(velocities),
            2
        ),

        "min_velocity": min(velocities),

        "max_velocity": max(velocities),

        "pitch_min": min(pitches),

        "pitch_max": max(pitches),

        "pitch_range": (
            max(pitches) - min(pitches)
        ),

        "average_pitch": round(
            sum(pitches) / len(pitches),
            2
        ),

        "note_density": round(
            note_density,
            4
        ),

        "average_note_duration": round(
            average_note_duration,
            4
        ),

        "polyphony": round(
            average_polyphony,
            4
        )
    }


# ============================================================
# TIME SIGNATURE
# ============================================================

def extract_time_signature(midi: pretty_midi.PrettyMIDI):
    """
    Lấy time signature đầu tiên của MIDI.

    Nếu MIDI không có time signature:
        mặc định 4/4.

    Args:
        midi: PrettyMIDI object.

    Returns:
        dict
    """

    if not midi.time_signature_changes:

        return {
            "numerator": 4,
            "denominator": 4
        }

    time_signature = (
        midi.time_signature_changes[0]
    )

    return {
        "numerator": int(
            time_signature.numerator
        ),
        "denominator": int(
            time_signature.denominator
        )
    }


# ============================================================
# KEY SIGNATURE
# ============================================================

def extract_key_signature(midi: pretty_midi.PrettyMIDI):
    """
    Lấy key signature đầu tiên.

    Args:
        midi: PrettyMIDI object.

    Returns:
        dict
    """

    if not midi.key_signature_changes:

        return {
            "key": None,
            "mode": None
        }

    key_change = (
        midi.key_signature_changes[0]
    )

    key_number = key_change.key_number

    # PrettyMIDI:
    # 0  = C major
    # 12 = C major
    #
    # 0-11  : major
    # 12-23 : minor

    if key_number < 12:

        pitch_class = key_number
        mode = "major"

    else:

        pitch_class = key_number - 12
        mode = "minor"

    note_names = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B"
    ]

    key_name = note_names[pitch_class]

    return {
        "key": key_name,
        "mode": mode
    }


# ============================================================
# MAIN FEATURE EXTRACTION
# ============================================================

def extract_features(file_path: str):
    """
    Trích xuất toàn bộ feature từ một file MIDI.

    Args:
        file_path: Đường dẫn MIDI.

    Returns:
        dict: Feature dictionary.
    """

    midi = pretty_midi.PrettyMIDI(
        file_path
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    duration = float(
        midi.get_end_time()
    )

    tempo = extract_tempo(
        midi
    )

    # --------------------------------------------------------
    # Instruments
    # --------------------------------------------------------

    instrument_features = (
        extract_instruments(midi)
    )

    # --------------------------------------------------------
    # Notes
    # --------------------------------------------------------

    notes = collect_notes(
        midi
    )

    note_features = (
        extract_note_features(
            notes,
            duration
        )
    )

    # --------------------------------------------------------
    # Time signature
    # --------------------------------------------------------

    time_signature = (
        extract_time_signature(midi)
    )

    # --------------------------------------------------------
    # Key signature
    # --------------------------------------------------------

    key_signature = (
        extract_key_signature(midi)
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    features = {

        "file_name": os.path.basename(
            file_path
        ),

        "file_path": file_path,

        "duration": round(
            duration,
            3
        ),

        "tempo": round(
            tempo,
            2
        ),

        **note_features,

        **instrument_features,

        "time_signature": time_signature,

        "key_signature": key_signature
    }

    return features


# ============================================================
# SAVE JSON
# ============================================================

def save_features(
    features,
    output_file: str
):
    """
    Lưu feature ra JSON.

    Args:
        features: Danh sách feature.
        output_file: Đường dẫn output.
    """

    output_path = Path(
        output_file
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
            features,
            file,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset():
    """
    Xử lý toàn bộ dataset MIDI.

    Với mỗi MIDI:
        MIDI → features

    Các file lỗi sẽ được bỏ qua,
    không làm dừng toàn bộ quá trình.
    """

    print("=" * 60)
    print("MUSIC FEATURE EXTRACTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Find MIDI files
    # --------------------------------------------------------

    midi_files = find_midi_files(
        INPUT_DIR
    )

    print(
        f"Tìm thấy {len(midi_files)} file MIDI."
    )

    if not midi_files:

        print(
            "Không tìm thấy file MIDI."
        )

        return

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    all_features = []

    success_count = 0
    failed_count = 0

    for index, file_path in enumerate(
        midi_files,
        start=1
    ):

        print(
            f"[{index}/{len(midi_files)}] "
            f"{os.path.basename(file_path)}"
        )

        try:

            features = extract_features(
                file_path
            )

            all_features.append(
                features
            )

            success_count += 1

        except Exception as error:

            failed_count += 1

            print(
                f"    [ERROR] {error}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_features(
        all_features,
        OUTPUT_FEATURE_FILE
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("HOÀN THÀNH")
    print("=" * 60)

    print(
        f"Thành công : {success_count}"
    )

    print(
        f"Lỗi        : {failed_count}"
    )

    print(
        f"Tổng        : {len(midi_files)}"
    )

    print()
    print(
        f"Output: {OUTPUT_FEATURE_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    process_dataset()