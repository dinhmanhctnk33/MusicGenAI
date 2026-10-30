import os
import pretty_midi
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = "./dataset"

OUTPUT_METADATA = "./metadata.csv"

MIN_DURATION = 5          # giây
MAX_DURATION = 600        # 10 phút
MIN_NOTES = 10

MIN_PITCH = 0
MAX_PITCH = 127

MIN_VELOCITY = 0
MAX_VELOCITY = 127


# ============================================================
# 1. TÌM TẤT CẢ FILE MIDI
# ============================================================

def find_midi_files(root_path):

    midi_files = []

    for root, dirs, files in os.walk(root_path):

        for file in files:

            if file.lower().endswith(
                (".mid", ".midi")
            ):

                full_path = os.path.join(
                    root,
                    file
                )

                midi_files.append(full_path)

    return midi_files


# ============================================================
# 2. KIỂM TRA MIDI
# ============================================================

def validate_midi(file_path):

    result = {
        "valid": False,
        "error": "",
        "duration": 0,
        "num_notes": 0,
        "num_instruments": 0,
        "min_pitch": None,
        "max_pitch": None,
        "avg_velocity": 0
    }

    try:

        midi = pretty_midi.PrettyMIDI(
            file_path
        )

        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        duration = midi.get_end_time()

        result["duration"] = duration

        if duration < MIN_DURATION:

            result["error"] = (
                f"Duration quá ngắn: {duration:.2f}s"
            )

            return result

        if duration > MAX_DURATION:

            result["error"] = (
                f"Duration quá dài: {duration:.2f}s"
            )

            return result

        # ----------------------------------------------------
        # Instruments
        # ----------------------------------------------------

        result["num_instruments"] = len(
            midi.instruments
        )

        # ----------------------------------------------------
        # Notes
        # ----------------------------------------------------

        all_notes = []

        for instrument in midi.instruments:

            for note in instrument.notes:

                all_notes.append(note)

        num_notes = len(all_notes)

        result["num_notes"] = num_notes

        if num_notes < MIN_NOTES:

            result["error"] = (
                f"Quá ít note: {num_notes}"
            )

            return result

        # ----------------------------------------------------
        # Pitch
        # ----------------------------------------------------

        pitches = [
            note.pitch
            for note in all_notes
        ]

        min_pitch = min(pitches)
        max_pitch = max(pitches)

        result["min_pitch"] = min_pitch
        result["max_pitch"] = max_pitch

        if min_pitch < MIN_PITCH:

            result["error"] = (
                f"Pitch không hợp lệ: {min_pitch}"
            )

            return result

        if max_pitch > MAX_PITCH:

            result["error"] = (
                f"Pitch không hợp lệ: {max_pitch}"
            )

            return result

        # ----------------------------------------------------
        # Velocity
        # ----------------------------------------------------

        velocities = [
            note.velocity
            for note in all_notes
        ]

        avg_velocity = sum(
            velocities
        ) / len(velocities)

        result["avg_velocity"] = avg_velocity

        # ----------------------------------------------------
        # Kiểm tra note
        # ----------------------------------------------------

        for note in all_notes:

            if note.start < 0:

                result["error"] = (
                    "Note có start < 0"
                )

                return result

            if note.end <= note.start:

                result["error"] = (
                    "Note có end <= start"
                )

                return result

            if not (
                MIN_PITCH
                <= note.pitch
                <= MAX_PITCH
            ):

                result["error"] = (
                    "Pitch không hợp lệ"
                )

                return result

            if not (
                MIN_VELOCITY
                <= note.velocity
                <= MAX_VELOCITY
            ):

                result["error"] = (
                    "Velocity không hợp lệ"
                )

                return result

        # ----------------------------------------------------
        # MIDI hợp lệ
        # ----------------------------------------------------

        result["valid"] = True

        return result

    except Exception as e:

        result["error"] = str(e)

        return result


# ============================================================
# 3. XỬ LÝ DATASET
# ============================================================

def process_dataset():

    print("=" * 60)
    print("MUSIC GENAI - MIDI DATA PROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Tìm MIDI
    # --------------------------------------------------------

    print("\n[1] Đang tìm file MIDI...")

    midi_files = find_midi_files(
        DATASET_ROOT
    )

    print(
        f"Tìm thấy {len(midi_files)} file MIDI."
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print("\n[2] Đang kiểm tra MIDI...")

    records = []

    for file_path in tqdm(
        midi_files,
        desc="Processing MIDI"
    ):

        result = validate_midi(
            file_path
        )

        record = {

            "file_path": file_path,

            "file_name": os.path.basename(
                file_path
            ),

            "valid": result["valid"],

            "error": result["error"],

            "duration": result["duration"],

            "num_notes": result["num_notes"],

            "num_instruments":
                result["num_instruments"],

            "min_pitch":
                result["min_pitch"],

            "max_pitch":
                result["max_pitch"],

            "avg_velocity":
                result["avg_velocity"]
        }

        records.append(record)

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_METADATA,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total = len(df)

    valid = df["valid"].sum()

    invalid = total - valid

    print("\n" + "=" * 60)
    print("KẾT QUẢ")
    print("=" * 60)

    print(
        f"Tổng số MIDI     : {total}"
    )

    print(
        f"MIDI hợp lệ      : {valid}"
    )

    print(
        f"MIDI không hợp lệ: {invalid}"
    )

    if total > 0:

        valid_percent = (
            valid / total * 100
        )

        print(
            f"Tỷ lệ hợp lệ     : "
            f"{valid_percent:.2f}%"
        )

    print(
        f"\nMetadata được lưu tại: "
        f"{OUTPUT_METADATA}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_dataset()