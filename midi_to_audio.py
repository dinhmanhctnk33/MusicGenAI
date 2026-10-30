"""
midi_to_audio.py

Pipeline:

    generated.mid
          ↓
    FluidSynth portable
          +
    FluidR3_GM.sf2
          ↓
    generated_temp.wav
          ↓
    imageio-ffmpeg
          ↓
    generated.mp3
"""

from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 2. INPUT / OUTPUT
# ============================================================

INPUT_MIDI = BASE_DIR / "generated.mid"

TEMP_WAV = BASE_DIR / "generated_temp.wav"

OUTPUT_MP3 = BASE_DIR / "generated.mp3"


# ============================================================
# 3. SOUNDFONT
# ============================================================

SOUNDFONT_FILE = (
    BASE_DIR
    / "soundfont"
    / "FluidR3_GM.sf2"
)


# ============================================================
# 4. FLUIDSYNTH DIRECTORY
# ============================================================

FLUIDSYNTH_DIR = (
    BASE_DIR
    / "tools"
    / "fluidsynth"
)


# ============================================================
# 5. AUDIO CONFIG
# ============================================================

SAMPLE_RATE = 44100

MP3_BITRATE = "192k"


# ============================================================
# 6. FIND FLUIDSYNTH
# ============================================================

def find_fluidsynth():
    """
    Tìm fluidsynth.exe bên trong project.

    Không cần cài FluidSynth vào Windows PATH.
    """

    candidates = list(
        FLUIDSYNTH_DIR.rglob(
            "fluidsynth.exe"
        )
    )

    if not candidates:

        raise FileNotFoundError(
            f"""
Không tìm thấy fluidsynth.exe.

Đã tìm trong:

{FLUIDSYNTH_DIR}

Hãy kiểm tra:

    dir /s /b tools\\fluidsynth\\fluidsynth.exe
"""
        )

    fluidsynth = candidates[0]

    print(
        f"✓ FluidSynth: {fluidsynth}"
    )

    return fluidsynth


# ============================================================
# 7. CHECK FILES
# ============================================================

def check_files():
    """
    Kiểm tra các file cần thiết.
    """

    print()
    print("=" * 70)
    print("CHECKING FILES")
    print("=" * 70)

    # --------------------------------------------------------
    # MIDI
    # --------------------------------------------------------

    if not INPUT_MIDI.exists():

        raise FileNotFoundError(
            f"""
Không tìm thấy:

{INPUT_MIDI}

Hãy chạy:

    python generate.py

trước.
"""
        )

    print(
        f"✓ MIDI      : {INPUT_MIDI}"
    )

    # --------------------------------------------------------
    # SoundFont
    # --------------------------------------------------------

    if not SOUNDFONT_FILE.exists():

        raise FileNotFoundError(
            f"""
Không tìm thấy:

{SOUNDFONT_FILE}

Hãy tải:

FluidR3_GM.sf2
"""
        )

    print(
        f"✓ SoundFont : {SOUNDFONT_FILE}"
    )

    # --------------------------------------------------------
    # FluidSynth
    # --------------------------------------------------------

    find_fluidsynth()

    # --------------------------------------------------------
    # FFmpeg
    # --------------------------------------------------------

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    print(
        f"✓ FFmpeg    : {ffmpeg}"
    )

    print("=" * 70)


# ============================================================
# 8. MIDI -> WAV
# ============================================================

def midi_to_wav():
    """
    Chuyển MIDI thành WAV bằng FluidSynth.
    """

    fluidsynth = find_fluidsynth()

    print()
    print("=" * 70)
    print("STEP 1: MIDI -> WAV")
    print("=" * 70)

    print(
        f"FluidSynth : {fluidsynth}"
    )

    print(
        f"Input MIDI : {INPUT_MIDI}"
    )

    print(
        f"SoundFont  : {SOUNDFONT_FILE}"
    )

    print(
        f"Output WAV : {TEMP_WAV}"
    )

    print()

    command = [
        str(fluidsynth),

        "-ni",

        "-F",
        str(TEMP_WAV),

        "-r",
        str(SAMPLE_RATE),

        str(SOUNDFONT_FILE),

        str(INPUT_MIDI),
    ]

    result = subprocess.run(
        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True,
    )

    if result.returncode != 0:

        print()

        print("FluidSynth ERROR:")

        print(result.stderr)

        raise RuntimeError(
            "MIDI -> WAV thất bại."
        )

    if not TEMP_WAV.exists():

        raise RuntimeError(
            "FluidSynth chạy nhưng "
            "không tạo được WAV."
        )

    print()

    print(
        f"✓ WAV created:"
    )

    print(
        f"  {TEMP_WAV}"
    )


# ============================================================
# 9. WAV -> MP3
# ============================================================

def wav_to_mp3():
    """
    Chuyển WAV thành MP3 bằng FFmpeg
    được cung cấp bởi imageio-ffmpeg.
    """

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    print()
    print("=" * 70)
    print("STEP 2: WAV -> MP3")
    print("=" * 70)

    print(
        f"FFmpeg     : {ffmpeg}"
    )

    print(
        f"Input WAV  : {TEMP_WAV}"
    )

    print(
        f"Output MP3 : {OUTPUT_MP3}"
    )

    print()

    command = [
        ffmpeg,

        "-y",

        "-i",

        str(TEMP_WAV),

        "-codec:a",

        "libmp3lame",

        "-b:a",

        MP3_BITRATE,

        str(OUTPUT_MP3),
    ]

    result = subprocess.run(
        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True,
    )

    if result.returncode != 0:

        print()

        print("FFmpeg ERROR:")

        print(result.stderr)

        raise RuntimeError(
            "WAV -> MP3 thất bại."
        )

    if not OUTPUT_MP3.exists():

        raise RuntimeError(
            "FFmpeg chạy nhưng "
            "không tạo được MP3."
        )

    print()

    print(
        f"✓ MP3 created:"
    )

    print(
        f"  {OUTPUT_MP3}"
    )


# ============================================================
# 10. CLEANUP
# ============================================================

def cleanup():
    """
    Xóa WAV tạm sau khi MP3 được tạo.
    """

    if TEMP_WAV.exists():

        TEMP_WAV.unlink()

        print()

        print(
            "✓ Temporary WAV deleted."
        )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # Check
        # ----------------------------------------------------

        check_files()

        # ----------------------------------------------------
        # MIDI -> WAV
        # ----------------------------------------------------

        midi_to_wav()

        # ----------------------------------------------------
        # WAV -> MP3
        # ----------------------------------------------------

        wav_to_mp3()

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        cleanup()

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("GENERATION AUDIO HOÀN TẤT")
        print("=" * 70)

        print()

        print(
            f"Input MIDI : {INPUT_MIDI}"
        )

        print(
            f"Output MP3 : {OUTPUT_MP3}"
        )

        print(
            f"Bitrate    : {MP3_BITRATE}"
        )

        print()

        print("=" * 70)

    except Exception as error:

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)

        print()

        print(error)

        print()

        print("=" * 70)

        sys.exit(1)


# ============================================================
# 12. RUN
# ============================================================

if __name__ == "__main__":
    main()