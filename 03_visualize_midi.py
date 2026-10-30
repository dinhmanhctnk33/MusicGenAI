import os
import pretty_midi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "dataset_clean"
OUTPUT_DIR = "visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SCAN MIDI DATASET
# ============================================================

def scan_midi_dataset(dataset_path):

    results = []

    midi_files = []

    for root, dirs, files in os.walk(dataset_path):

        for file in files:

            if file.lower().endswith((".mid", ".midi")):

                midi_files.append(
                    os.path.join(root, file)
                )

    print(f"Tìm thấy {len(midi_files)} file MIDI")

    for file_path in tqdm(
        midi_files,
        desc="Analyzing MIDI"
    ):

        try:

            midi = pretty_midi.PrettyMIDI(
                file_path
            )

            notes = []

            velocities = []

            for instrument in midi.instruments:

                for note in instrument.notes:

                    notes.append(note)

                    velocities.append(
                        note.velocity
                    )

            if len(notes) == 0:
                continue

            pitches = [
                note.pitch
                for note in notes
            ]

            durations = [
                note.end - note.start
                for note in notes
            ]

            # ----------------------------------------
            # Tempo
            # ----------------------------------------

            tempos, tempo_times = (
                midi.get_tempo_changes()
            )

            if len(tempos) > 0:

                tempo = float(
                    np.mean(tempos)
                )

            else:

                tempo = 120.0

            # ----------------------------------------
            # Duration
            # ----------------------------------------

            duration = midi.get_end_time()

            # ----------------------------------------
            # Note density
            # ----------------------------------------

            note_density = (
                len(notes) / duration
                if duration > 0
                else 0
            )

            # ----------------------------------------
            # Instruments
            # ----------------------------------------

            instrument_count = len(
                midi.instruments
            )

            active_instruments = sum(
                1
                for instrument in midi.instruments
                if len(instrument.notes) > 0
            )

            # ----------------------------------------
            # Result
            # ----------------------------------------

            results.append({

                "file": file_path,

                "duration_seconds": duration,

                "total_notes": len(notes),

                "tempo_bpm": tempo,

                "note_density": note_density,

                "mean_pitch": np.mean(pitches),

                "min_pitch": np.min(pitches),

                "max_pitch": np.max(pitches),

                "pitch_range": (
                    np.max(pitches)
                    - np.min(pitches)
                ),

                "mean_velocity": np.mean(
                    velocities
                ),

                "mean_note_duration": np.mean(
                    durations
                ),

                "instruments": instrument_count,

                "active_instruments":
                    active_instruments
            })

        except Exception as e:

            print(
                f"\n[WARNING] "
                f"{file_path}: {e}"
            )

    return pd.DataFrame(results)


# ============================================================
# HISTOGRAM FUNCTION
# ============================================================

def save_histogram(
    data,
    title,
    xlabel,
    filename,
    bins=30
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        data,
        bins=bins
    )

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel("Number of MIDI files")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# PITCH DISTRIBUTION
# ============================================================

def analyze_pitch_distribution(
    dataset_path
):

    pitches = []

    midi_files = []

    for root, dirs, files in os.walk(
        dataset_path
    ):

        for file in files:

            if file.lower().endswith(
                (".mid", ".midi")
            ):

                midi_files.append(
                    os.path.join(
                        root,
                        file
                    )
                )

    for file_path in tqdm(
        midi_files,
        desc="Analyzing pitch"
    ):

        try:

            midi = pretty_midi.PrettyMIDI(
                file_path
            )

            for instrument in midi.instruments:

                for note in instrument.notes:

                    pitches.append(
                        note.pitch
                    )

        except Exception:
            continue

    if not pitches:
        return

    plt.figure(
        figsize=(12, 6)
    )

    plt.hist(
        pitches,
        bins=np.arange(
            0,
            129,
            1
        )
    )

    plt.title(
        "MIDI Pitch Distribution"
    )

    plt.xlabel(
        "MIDI Pitch"
    )

    plt.ylabel(
        "Number of Notes"
    )

    plt.xticks(
        range(0, 128, 12)
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "pitch_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# VELOCITY DISTRIBUTION
# ============================================================

def analyze_velocity_distribution(
    dataset_path
):

    velocities = []

    for root, dirs, files in os.walk(
        dataset_path
    ):

        for file in files:

            if not file.lower().endswith(
                (".mid", ".midi")
            ):
                continue

            file_path = os.path.join(
                root,
                file
            )

            try:

                midi = pretty_midi.PrettyMIDI(
                    file_path
                )

                for instrument in midi.instruments:

                    for note in instrument.notes:

                        velocities.append(
                            note.velocity
                        )

            except Exception:
                continue

    if not velocities:
        return

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        velocities,
        bins=32
    )

    plt.title(
        "MIDI Velocity Distribution"
    )

    plt.xlabel(
        "Velocity"
    )

    plt.ylabel(
        "Number of Notes"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "velocity_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# PIANO ROLL
# ============================================================

def create_piano_roll(
    midi_path,
    output_filename,
    max_seconds=60
):

    midi = pretty_midi.PrettyMIDI(
        midi_path
    )

    plt.figure(
        figsize=(16, 8)
    )

    for instrument in midi.instruments:

        for note in instrument.notes:

            # Chỉ hiển thị 60 giây đầu
            if note.start > max_seconds:
                continue

            start = note.start

            end = min(
                note.end,
                max_seconds
            )

            plt.plot(
                [start, end],
                [note.pitch, note.pitch],
                linewidth=3
            )

    plt.title(
        f"Piano Roll - "
        f"{os.path.basename(midi_path)}"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "MIDI Pitch"
    )

    plt.xlim(
        0,
        max_seconds
    )

    plt.ylim(
        20,
        108
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            output_filename
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "MIDI DATASET VISUALIZATION"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

    df = scan_midi_dataset(
        DATASET_PATH
    )

    if df.empty:

        print(
            "Không tìm thấy dữ liệu MIDI hợp lệ."
        )

        return

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "midi_statistics.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nĐã phân tích {len(df)} MIDI"
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n===== THỐNG KÊ =====")

    print(
        df[
            [
                "duration_seconds",
                "total_notes",
                "tempo_bpm",
                "note_density",
                "mean_pitch",
                "mean_velocity"
            ]
        ].describe()
    )

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    save_histogram(
        df["duration_seconds"],
        "MIDI Duration Distribution",
        "Duration (seconds)",
        "duration_distribution.png"
    )

    # --------------------------------------------------------
    # Note count
    # --------------------------------------------------------

    save_histogram(
        df["total_notes"],
        "MIDI Note Count Distribution",
        "Number of Notes",
        "note_count_distribution.png"
    )

    # --------------------------------------------------------
    # Tempo
    # --------------------------------------------------------

    save_histogram(
        df["tempo_bpm"],
        "MIDI Tempo Distribution",
        "Tempo (BPM)",
        "tempo_distribution.png"
    )

    # --------------------------------------------------------
    # Note density
    # --------------------------------------------------------

    save_histogram(
        df["note_density"],
        "MIDI Note Density Distribution",
        "Notes per second",
        "note_density_distribution.png"
    )

    # --------------------------------------------------------
    # Mean velocity
    # --------------------------------------------------------

    save_histogram(
        df["mean_velocity"],
        "MIDI Mean Velocity Distribution",
        "Mean Velocity",
        "velocity_mean_distribution.png"
    )

    # --------------------------------------------------------
    # Pitch
    # --------------------------------------------------------

    analyze_pitch_distribution(
        DATASET_PATH
    )

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    analyze_velocity_distribution(
        DATASET_PATH
    )

    # --------------------------------------------------------
    # Piano Roll
    # --------------------------------------------------------

    first_file = df.iloc[0]["file"]

    create_piano_roll(
        first_file,
        "piano_roll_sample.png"
    )

    print(
        "\nĐã tạo visualization tại:"
    )

    print(
        f"{OUTPUT_DIR}/"
    )


if __name__ == "__main__":

    main()