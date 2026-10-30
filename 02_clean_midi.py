# 02_clean_midi.py

import os
import shutil
import pretty_midi
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_REPORT = "midi_validation_report.csv"
OUTPUT_DIR = "dataset_clean"
CLEANING_REPORT = "cleaning_report.csv"

MAX_DURATION = 600          # seconds
MIN_DURATION = 5            # seconds
SEGMENT_BARS = 32           # số bar tối đa mỗi segment


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_midi(file_path):
    """
    Đọc MIDI bằng PrettyMIDI.
    """
    try:
        return pretty_midi.PrettyMIDI(file_path)
    except Exception as e:
        print(f"[ERROR] Không đọc được: {file_path}")
        print(f"        {e}")
        return None


def get_active_time_range(midi):
    """
    Tìm khoảng thời gian thực sự có note.

    Returns:
        first_note_time
        last_note_time

    Nếu MIDI không có note:
        (None, None)
    """

    note_starts = []
    note_ends = []

    for instrument in midi.instruments:

        for note in instrument.notes:
            note_starts.append(note.start)
            note_ends.append(note.end)

    if not note_starts:
        return None, None

    return min(note_starts), max(note_ends)


def remove_silence(midi):
    """
    Loại silence ở đầu và cuối MIDI.

    Ví dụ:

        0s ─── silence ─── note ───── note ─── silence ─── 100s

                       ↓

        note ───── note

    Tất cả note được dịch về bắt đầu từ thời điểm 0.
    """

    start_time, end_time = get_active_time_range(midi)

    if start_time is None:
        return midi, 0, 0

    # Dịch toàn bộ note về bắt đầu từ 0
    for instrument in midi.instruments:

        for note in instrument.notes:

            note.start -= start_time
            note.end -= start_time

    removed_start = start_time
    removed_end = midi.get_end_time() - end_time

    return midi, removed_start, removed_end


def get_tempo(midi):
    """
    Lấy tempo đầu tiên của MIDI.

    Nếu không có tempo event:
    sử dụng 120 BPM.
    """

    tempos, times = midi.get_tempo_changes()

    if len(tempos) > 0:
        return float(tempos[0])

    return 120.0


def estimate_bar_duration(midi):
    """
    Ước lượng thời lượng một bar.

    Ưu tiên Time Signature đầu tiên.

    Nếu không có Time Signature:
        mặc định 4/4.

    Công thức:

        bar_duration =
            60 / BPM * beats_per_bar
    """

    tempo = get_tempo(midi)

    # BPM -> thời gian của một beat
    beat_duration = 60.0 / tempo

    # Mặc định 4/4
    numerator = 4
    denominator = 4

    if midi.time_signature_changes:

        ts = midi.time_signature_changes[0]

        numerator = ts.numerator
        denominator = ts.denominator

    # Một quarter note tương ứng với denominator = 4
    beats_per_bar = numerator * (4.0 / denominator)

    bar_duration = beat_duration * beats_per_bar

    return bar_duration


def split_midi_by_bars(midi, bars_per_segment=32):
    """
    Chia MIDI thành các segment dựa trên bar.

    Mỗi segment tối đa bars_per_segment bar.

    Lưu ý:
    Đây là phiên bản đơn giản dựa trên tempo/time signature.
    """

    bar_duration = estimate_bar_duration(midi)

    duration = midi.get_end_time()

    segment_duration = bar_duration * bars_per_segment

    segments = []

    current_start = 0.0

    while current_start < duration:

        current_end = min(
            current_start + segment_duration,
            duration
        )

        segment = pretty_midi.PrettyMIDI(
            resolution=midi.resolution
        )

        # ------------------------------------------------
        # Copy instruments
        # ------------------------------------------------

        for instrument in midi.instruments:

            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name
            )

            # Copy note thuộc segment
            for note in instrument.notes:

                # Note hoàn toàn nằm ngoài segment
                if note.end <= current_start:
                    continue

                if note.start >= current_end:
                    continue

                # Cắt note nếu note nằm xuyên qua boundary
                new_start = max(
                    note.start,
                    current_start
                ) - current_start

                new_end = min(
                    note.end,
                    current_end
                ) - current_start

                # Chỉ thêm note hợp lệ
                if new_end > new_start:

                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=new_start,
                        end=new_end
                    )

                    new_instrument.notes.append(
                        new_note
                    )

            # Chỉ thêm instrument có note
            if len(new_instrument.notes) > 0:
                segment.instruments.append(
                    new_instrument
                )

        # ------------------------------------------------
        # Copy tempo
        # ------------------------------------------------

        tempo = get_tempo(midi)

        segment._tick_scales = midi._tick_scales

        # ------------------------------------------------
        # Kiểm tra segment có note không
        # ------------------------------------------------

        has_notes = any(
            len(inst.notes) > 0
            for inst in segment.instruments
        )

        if has_notes:
            segments.append(segment)

        current_start = current_end

    return segments


def save_midi(midi, output_path):
    """
    Lưu MIDI.
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    midi.write(output_path)


# ============================================================
# PROCESS NORMAL MIDI
# ============================================================

def process_normal_midi(
    file_path,
    output_path
):
    """
    Xử lý MIDI <= 600 giây.

    1. Load MIDI
    2. Remove silence
    3. Save
    """

    midi = load_midi(file_path)

    if midi is None:
        return False, "Không đọc được MIDI"

    # Remove silence
    midi, removed_start, removed_end = remove_silence(midi)

    duration = midi.get_end_time()

    if duration < MIN_DURATION:
        return False, "Sau khi remove silence, duration quá ngắn"

    save_midi(
        midi,
        output_path
    )

    return True, "Cleaned"


# ============================================================
# PROCESS LONG MIDI
# ============================================================

def process_long_midi(
    file_path,
    output_dir
):
    """
    Xử lý MIDI > 600 giây.

    1. Load MIDI
    2. Remove silence
    3. Split theo bar
    4. Save từng segment
    """

    midi = load_midi(file_path)

    if midi is None:
        return 0, "Không đọc được MIDI"

    # ------------------------------------------------
    # Remove silence
    # ------------------------------------------------

    midi, removed_start, removed_end = remove_silence(midi)

    duration = midi.get_end_time()

    # Nếu remove silence xong lại <= 600s
    if duration <= MAX_DURATION:

        filename = os.path.basename(file_path)

        output_path = os.path.join(
            output_dir,
            filename
        )

        save_midi(
            midi,
            output_path
        )

        return 1, "Cleaned after silence removal"

    # ------------------------------------------------
    # Split
    # ------------------------------------------------

    segments = split_midi_by_bars(
        midi,
        SEGMENT_BARS
    )

    if len(segments) == 0:
        return 0, "Không tạo được segment"

    base_name = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    saved_count = 0

    for index, segment in enumerate(segments):

        segment_name = (
            f"{base_name}_segment_{index + 1:03d}.mid"
        )

        output_path = os.path.join(
            output_dir,
            segment_name
        )

        # Kiểm tra duration
        segment_duration = segment.get_end_time()

        if segment_duration < MIN_DURATION:
            continue

        save_midi(
            segment,
            output_path
        )

        saved_count += 1

    return saved_count, "Split into segments"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MIDI CLEANING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load validation report
    # --------------------------------------------------------

    if not os.path.exists(VALIDATION_REPORT):

        print(
            f"[ERROR] Không tìm thấy {VALIDATION_REPORT}"
        )

        return

    df = pd.read_csv(
        VALIDATION_REPORT
    )

    print(
        f"Tổng số record: {len(df)}"
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Remove duplicate
    # --------------------------------------------------------

    if "file_hash" in df.columns:

        duplicate_mask = df.duplicated(
            subset=["file_hash"],
            keep="first"
        )

        duplicate_count = duplicate_mask.sum()

        print(
            f"File duplicate: {duplicate_count}"
        )

        df = df[
            ~duplicate_mask
        ].copy()

    else:

        print(
            "[WARNING] Không tìm thấy file_hash."
        )

        print(
            "Không thể kiểm tra exact duplicate."
        )

    # --------------------------------------------------------
    # Chỉ xử lý VALID
    # --------------------------------------------------------

    if "status" in df.columns:

        df = df[
            df["status"] == "VALID"
        ].copy()

    print(
        f"File hợp lệ cần xử lý: {len(df)}"
    )

    # --------------------------------------------------------
    # Cleaning results
    # --------------------------------------------------------

    cleaning_results = []

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Cleaning MIDI"
    ):

        file_path = row["file"]

        if not os.path.exists(file_path):

            cleaning_results.append({
                "source_file": file_path,
                "status": "ERROR",
                "action": "File không tồn tại",
                "output_file": "",
                "segments": 0
            })

            continue

        duration = row["duration_seconds"]

        # ====================================================
        # MIDI <= 600 seconds
        # ====================================================

        if duration <= MAX_DURATION:

            filename = os.path.basename(
                file_path
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                filename
            )

            success, message = process_normal_midi(
                file_path,
                output_path
            )

            cleaning_results.append({
                "source_file": file_path,
                "status": "SUCCESS" if success else "ERROR",
                "action": message,
                "output_file": output_path
                if success else "",
                "segments": 1 if success else 0
            })

        # ====================================================
        # MIDI > 600 seconds
        # ====================================================

        else:

            segment_count, message = process_long_midi(
                file_path,
                OUTPUT_DIR
            )

            cleaning_results.append({
                "source_file": file_path,
                "status": (
                    "SUCCESS"
                    if segment_count > 0
                    else "ERROR"
                ),
                "action": message,
                "output_file": OUTPUT_DIR,
                "segments": segment_count
            })

    # --------------------------------------------------------
    # Save cleaning report
    # --------------------------------------------------------

    cleaning_df = pd.DataFrame(
        cleaning_results
    )

    cleaning_df.to_csv(
        CLEANING_REPORT,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("KẾT QUẢ CLEANING")
    print("=" * 60)

    print(
        f"Input hợp lệ: {len(df)}"
    )

    print(
        f"Processed thành công: "
        f"{(cleaning_df['status'] == 'SUCCESS').sum()}"
    )

    print(
        f"Lỗi: "
        f"{(cleaning_df['status'] == 'ERROR').sum()}"
    )

    print(
        f"Tổng segment/output: "
        f"{cleaning_df['segments'].sum()}"
    )

    print(
        f"\nDataset sạch: {OUTPUT_DIR}/"
    )

    print(
        f"Cleaning report: {CLEANING_REPORT}"
    )


if __name__ == "__main__":
    main()