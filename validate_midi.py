# import os
# import pretty_midi
# import pandas as pd
# from tqdm import tqdm
#
#
# DATASET_PATH = "dataset"
#
# # Các ngưỡng kiểm tra
# MIN_DURATION = 5       # giây
# MAX_DURATION = 600     # 10 phút
# MIN_NOTES = 10
#
#
# def validate_midi(file_path):
#     """
#     Kiểm tra tính hợp lệ của một file MIDI.
#     Trả về:
#         status: valid / invalid
#         reason: lý do
#         info: thông tin thống kê
#     """
#
#     try:
#         midi = pretty_midi.PrettyMIDI(file_path)
#
#         total_notes = 0
#         invalid_notes = 0
#
#         for instrument in midi.instruments:
#
#             for note in instrument.notes:
#                 total_notes += 1
#
#                 # Kiểm tra pitch
#                 if not 0 <= note.pitch <= 127:
#                     invalid_notes += 1
#
#                 # Kiểm tra velocity
#                 if not 0 <= note.velocity <= 127:
#                     invalid_notes += 1
#
#                 # Kiểm tra thời gian note
#                 if note.start < 0:
#                     invalid_notes += 1
#
#                 if note.end <= note.start:
#                     invalid_notes += 1
#
#         # Thời lượng bài nhạc
#         duration = midi.get_end_time()
#
#         # Kiểm tra track có note
#         instrument_count = len(midi.instruments)
#         active_instruments = sum(
#             1 for inst in midi.instruments
#             if len(inst.notes) > 0
#         )
#
#         # Kiểm tra các điều kiện
#         if total_notes == 0:
#             return False, "Không có note", {
#                 "duration": duration,
#                 "notes": total_notes,
#                 "tracks": instrument_count
#             }
#
#         if total_notes < MIN_NOTES:
#             return False, f"Quá ít note (< {MIN_NOTES})", {
#                 "duration": duration,
#                 "notes": total_notes,
#                 "tracks": instrument_count
#             }
#
#         if duration < MIN_DURATION:
#             return False, f"Thời lượng quá ngắn (< {MIN_DURATION}s)", {
#                 "duration": duration,
#                 "notes": total_notes,
#                 "tracks": instrument_count
#             }
#
#         if duration > MAX_DURATION:
#             return False, f"Thời lượng quá dài (> {MAX_DURATION}s)", {
#                 "duration": duration,
#                 "notes": total_notes,
#                 "tracks": instrument_count
#             }
#
#         if invalid_notes > 0:
#             return False, f"Có {invalid_notes} note không hợp lệ", {
#                 "duration": duration,
#                 "notes": total_notes,
#                 "tracks": instrument_count
#             }
#
#         if active_instruments == 0:
#             return False, "Không có track chứa note", {
#                 "duration": duration,
#                 "notes": total_notes,
#                 "tracks": instrument_count
#             }
#
#         return True, "Valid", {
#             "duration": duration,
#             "notes": total_notes,
#             "tracks": instrument_count
#         }
#
#     except Exception as e:
#
#         return False, f"Lỗi đọc MIDI: {str(e)}", {
#             "duration": None,
#             "notes": None,
#             "tracks": None
#         }
#
#
# def scan_dataset(dataset_path):
#
#     results = []
#
#     midi_files = []
#
#     # Tìm tất cả file .mid và .midi
#     for root, dirs, files in os.walk(dataset_path):
#         for file in files:
#             if file.lower().endswith((".mid", ".midi")):
#                 midi_files.append(
#                     os.path.join(root, file)
#                 )
#
#     print(f"Tìm thấy {len(midi_files)} file MIDI")
#
#     # Kiểm tra từng file
#     for file_path in tqdm(midi_files):
#
#         valid, reason, info = validate_midi(file_path)
#
#         results.append({
#             "file": file_path,
#             "status": "VALID" if valid else "INVALID",
#             "reason": reason,
#             "duration_seconds": info["duration"],
#             "total_notes": info["notes"],
#             "tracks": info["tracks"]
#         })
#
#     return pd.DataFrame(results)
#
#
# if __name__ == "__main__":
#
#     df = scan_dataset(DATASET_PATH)
#
#     # Xuất báo cáo
#     df.to_csv(
#         "midi_validation_report.csv",
#         index=False,
#         encoding="utf-8-sig"
#     )
#
#     # Thống kê
#     print("\n===== KẾT QUẢ KIỂM TRA =====")
#
#     print(f"Tổng số file: {len(df)}")
#
#     print(
#         f"Valid: {(df['status'] == 'VALID').sum()}"
#     )
#
#     print(
#         f"Invalid: {(df['status'] == 'INVALID').sum()}"
#     )
#
#     print("\nChi tiết lỗi:")
#
#     print(
#         df[df["status"] == "INVALID"][
#             ["file", "reason"]
#         ].to_string(index=False)
#     )
import os
import hashlib
import pretty_midi
import pandas as pd
from tqdm import tqdm


DATASET_PATH = "dataset"

# Các ngưỡng kiểm tra
MIN_DURATION = 5
MAX_DURATION = 600
MIN_NOTES = 10


def calculate_file_hash(file_path):
    """
    Tính SHA-256 dựa trên nội dung binary của file MIDI.
    Hai file có cùng hash => nội dung file giống hệt nhau.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            data = f.read(8192)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


def validate_midi(file_path):
    """
    Kiểm tra tính hợp lệ của một file MIDI.
    """

    try:
        midi = pretty_midi.PrettyMIDI(file_path)

        total_notes = 0
        invalid_notes = 0

        for instrument in midi.instruments:

            for note in instrument.notes:

                total_notes += 1

                # Kiểm tra pitch
                if not 0 <= note.pitch <= 127:
                    invalid_notes += 1

                # Kiểm tra velocity
                if not 0 <= note.velocity <= 127:
                    invalid_notes += 1

                # Kiểm tra thời gian
                if note.start < 0:
                    invalid_notes += 1

                if note.end <= note.start:
                    invalid_notes += 1

        # Thời lượng
        duration = midi.get_end_time()

        # Số track
        instrument_count = len(midi.instruments)

        # Số track thực sự chứa note
        active_instruments = sum(
            1
            for inst in midi.instruments
            if len(inst.notes) > 0
        )

        # =========================
        # KIỂM TRA ĐIỀU KIỆN
        # =========================

        if total_notes == 0:
            return False, "Không có note", {
                "duration": duration,
                "notes": total_notes,
                "tracks": instrument_count
            }

        if total_notes < MIN_NOTES:
            return False, f"Quá ít note (< {MIN_NOTES})", {
                "duration": duration,
                "notes": total_notes,
                "tracks": instrument_count
            }

        if duration < MIN_DURATION:
            return False, f"Thời lượng quá ngắn (< {MIN_DURATION}s)", {
                "duration": duration,
                "notes": total_notes,
                "tracks": instrument_count
            }

        if duration > MAX_DURATION:
            return False, f"Thời lượng quá dài (> {MAX_DURATION}s)", {
                "duration": duration,
                "notes": total_notes,
                "tracks": instrument_count
            }

        if invalid_notes > 0:
            return False, f"Có {invalid_notes} note không hợp lệ", {
                "duration": duration,
                "notes": total_notes,
                "tracks": instrument_count
            }

        if active_instruments == 0:
            return False, "Không có track chứa note", {
                "duration": duration,
                "notes": total_notes,
                "tracks": instrument_count
            }

        return True, "Valid", {
            "duration": duration,
            "notes": total_notes,
            "tracks": instrument_count
        }

    except Exception as e:

        return False, f"Lỗi đọc MIDI: {str(e)}", {
            "duration": None,
            "notes": None,
            "tracks": None
        }


def scan_dataset(dataset_path):

    results = []

    midi_files = []

    # =========================
    # TÌM FILE MIDI
    # =========================

    for root, dirs, files in os.walk(dataset_path):

        for file in files:

            if file.lower().endswith((".mid", ".midi")):

                midi_files.append(
                    os.path.join(root, file)
                )

    print(f"Tìm thấy {len(midi_files)} file MIDI")

    # =========================
    # KIỂM TRA FILE
    # =========================

    for file_path in tqdm(midi_files):

        valid, reason, info = validate_midi(file_path)

        # Tính hash
        file_hash = calculate_file_hash(file_path)

        results.append({
            "file": file_path,
            "status": "VALID" if valid else "INVALID",
            "reason": reason,
            "duration_seconds": info["duration"],
            "total_notes": info["notes"],
            "tracks": info["tracks"],
            "file_hash": file_hash
        })

    df = pd.DataFrame(results)

    # =========================
    # PHÁT HIỆN FILE TRÙNG
    # =========================

    duplicate_mask = df.duplicated(
        subset=["file_hash"],
        keep="first"
    )

    df["duplicate"] = duplicate_mask

    # Đánh dấu rõ file nào là bản sao
    df.loc[
        df["duplicate"],
        "reason"
    ] = "File trùng với file khác"

    return df


if __name__ == "__main__":

    df = scan_dataset(DATASET_PATH)

    # =========================
    # XUẤT BÁO CÁO
    # =========================

    df.to_csv(
        "midi_validation_report.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # =========================
    # THỐNG KÊ
    # =========================

    print("\n===== KẾT QUẢ KIỂM TRA =====")

    print(f"Tổng số file: {len(df)}")

    print(
        f"Valid: {(df['status'] == 'VALID').sum()}"
    )

    print(
        f"Invalid: {(df['status'] == 'INVALID').sum()}"
    )

    print(
        f"File trùng: {df['duplicate'].sum()}"
    )

    # =========================
    # HIỂN THỊ FILE TRÙNG
    # =========================

    print("\n===== FILE TRÙNG =====")

    duplicates = df[df["duplicate"]]

    if len(duplicates) == 0:

        print("Không phát hiện file trùng.")

    else:

        print(
            duplicates[
                ["file", "file_hash", "reason"]
            ].to_string(index=False)
        )