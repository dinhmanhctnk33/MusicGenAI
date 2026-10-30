"""
caption_generator.py

Tạo các caption tự nhiên cho dữ liệu MIDI dựa trên
các đặc trưng âm nhạc đã được trích xuất.

Input:
    processed_data/music_features.json

Output:
    processed_data/caption_dataset.json

Mục tiêu:
    MIDI
      ↓
    Music Features
      ↓
    Musical Attributes
      ↓
    Multiple Natural Language Captions

Lưu ý:
    Caption được sinh bằng rule-based/heuristic.
    Đây chưa phải nhãn cảm xúc chính xác do con người gán.
    Các nhãn như "vui", "buồn", "sôi động"... chỉ là
    mô tả xấp xỉ dựa trên đặc trưng âm nhạc.
"""

import json
import os
import random
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = "./processed_data/music_features.json"

OUTPUT_DIR = "./processed_data"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "caption_dataset.json"
)

# Số caption muốn tạo cho mỗi MIDI
CAPTIONS_PER_MIDI = 8

# Seed để mỗi lần chạy có kết quả ổn định
RANDOM_SEED = 42


# ============================================================
# LOAD JSON
# ============================================================

def load_json(file_path: str):
    """
    Đọc file JSON.

    Args:
        file_path: Đường dẫn file JSON.

    Returns:
        object: Nội dung JSON.
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
# TEMPO CLASSIFICATION
# ============================================================

def classify_tempo(tempo: float) -> str:
    """
    Phân loại tempo.

    Args:
        tempo: BPM.

    Returns:
        str
    """

    if tempo < 60:
        return "very_slow"

    if tempo < 90:
        return "slow"

    if tempo < 120:
        return "moderate"

    if tempo < 150:
        return "fast"

    return "very_fast"


# ============================================================
# ENERGY CLASSIFICATION
# ============================================================

def classify_energy(
    average_velocity: float,
    tempo: float
) -> str:
    """
    Ước lượng mức năng lượng của bản nhạc.

    Đây là heuristic dựa trên:
        - velocity
        - tempo

    Args:
        average_velocity: Velocity trung bình.
        tempo: BPM.

    Returns:
        str
    """

    velocity_score = (
        average_velocity / 127.0
    )

    tempo_score = min(
        tempo / 160.0,
        1.0
    )

    score = (
        velocity_score * 0.5
        +
        tempo_score * 0.5
    )

    if score < 0.30:
        return "low"

    if score < 0.55:
        return "medium"

    if score < 0.75:
        return "high"

    return "very_high"


# ============================================================
# DENSITY CLASSIFICATION
# ============================================================

def classify_density(
    note_density: float
) -> str:
    """
    Phân loại mật độ note.

    Args:
        note_density: Số note / giây.

    Returns:
        str
    """

    if note_density < 2:
        return "very_low"

    if note_density < 5:
        return "low"

    if note_density < 10:
        return "medium"

    if note_density < 20:
        return "high"

    return "very_high"


# ============================================================
# DURATION CLASSIFICATION
# ============================================================

def classify_duration(
    duration: float
) -> str:
    """
    Phân loại thời lượng bản nhạc.

    Args:
        duration: Thời lượng tính bằng giây.

    Returns:
        str
    """

    if duration < 30:
        return "very_short"

    if duration < 120:
        return "short"

    if duration < 300:
        return "medium"

    if duration < 600:
        return "long"

    return "very_long"


# ============================================================
# MOOD HEURISTIC
# ============================================================

def infer_mood(
    mode,
    tempo,
    energy,
    density
) -> str:
    """
    Ước lượng mood từ các đặc trưng.

    Đây chỉ là heuristic.

    Không nên coi kết quả là ground-truth
    về cảm xúc của bản nhạc.

    Args:
        mode: major/minor.
        tempo: BPM.
        energy: Mức năng lượng.
        density: Mật độ note.

    Returns:
        str
    """

    if mode == "major":

        if energy in {
            "high",
            "very_high"
        }:

            return "happy"

        if tempo >= 110:

            return "bright"

        return "positive"

    if mode == "minor":

        if energy == "low":

            return "melancholic"

        if tempo < 90:

            return "sad"

        return "dramatic"

    # Không xác định mode

    if energy == "very_high":

        return "energetic"

    if energy == "high":

        return "lively"

    if energy == "low":

        return "calm"

    return "neutral"


# ============================================================
# INSTRUMENT DESCRIPTION
# ============================================================

def simplify_instrument(
    instrument_name: str
) -> str:
    """
    Chuyển tên instrument MIDI sang nhóm dễ hiểu.

    Args:
        instrument_name: Tên instrument.

    Returns:
        str
    """

    name = instrument_name.lower()

    if any(
        keyword in name
        for keyword in [
            "piano",
            "keyboard",
            "organ"
        ]
    ):
        return "piano"

    if any(
        keyword in name
        for keyword in [
            "guitar"
        ]
    ):
        return "guitar"

    if any(
        keyword in name
        for keyword in [
            "violin",
            "viola",
            "cello",
            "string"
        ]
    ):
        return "strings"

    if any(
        keyword in name
        for keyword in [
            "flute",
            "oboe",
            "clarinet",
            "sax",
            "trumpet",
            "trombone"
        ]
    ):
        return "wind"

    if any(
        keyword in name
        for keyword in [
            "bass"
        ]
    ):
        return "bass"

    if "drum" in name:
        return "drums"

    return "other"


def get_instrument_groups(
    instruments
):
    """
    Chuyển danh sách instrument thành nhóm.

    Args:
        instruments: list[str]

    Returns:
        list[str]
    """

    groups = []

    for instrument in instruments:

        group = simplify_instrument(
            instrument
        )

        if group not in groups:
            groups.append(group)

    return groups


# ============================================================
# ATTRIBUTE EXTRACTION
# ============================================================

def extract_attributes(
    feature: dict
) -> dict:
    """
    Chuyển music features thành các musical attributes.

    Args:
        feature: Feature dictionary.

    Returns:
        dict
    """

    tempo = feature.get(
        "tempo",
        120
    )

    duration = feature.get(
        "duration",
        0
    )

    average_velocity = feature.get(
        "average_velocity",
        0
    )

    note_density = feature.get(
        "note_density",
        0
    )

    polyphony = feature.get(
        "polyphony",
        0
    )

    key_signature = feature.get(
        "key_signature",
        {}
    )

    mode = key_signature.get(
        "mode"
    )

    key = key_signature.get(
        "key"
    )

    instruments = feature.get(
        "instruments",
        []
    )

    tempo_category = classify_tempo(
        tempo
    )

    energy_category = classify_energy(
        average_velocity,
        tempo
    )

    density_category = classify_density(
        note_density
    )

    duration_category = classify_duration(
        duration
    )

    mood = infer_mood(
        mode,
        tempo,
        energy_category,
        density_category
    )

    instrument_groups = get_instrument_groups(
        instruments
    )

    return {

        "tempo": round(
            tempo,
            2
        ),

        "tempo_category": tempo_category,

        "energy": energy_category,

        "density": density_category,

        "duration_category": duration_category,

        "duration_seconds": round(
            duration,
            2
        ),

        "mood": mood,

        "key": key,

        "mode": mode,

        "instrument_groups": instrument_groups,

        "instrument_count": feature.get(
            "instrument_count",
            0
        ),

        "note_count": feature.get(
            "note_count",
            0
        ),

        "note_density": round(
            note_density,
            4
        ),

        "average_velocity": round(
            average_velocity,
            2
        ),

        "polyphony": round(
            polyphony,
            4
        ),

        "pitch_range": feature.get(
            "pitch_range",
            0
        )
    }


# ============================================================
# HUMAN-READABLE VALUES
# ============================================================

TEMPO_TEXT = {

    "very_slow": [
        "rất chậm",
        "cực kỳ chậm",
        "có tiết tấu rất chậm"
    ],

    "slow": [
        "chậm",
        "nhẹ nhàng",
        "có tiết tấu chậm"
    ],

    "moderate": [
        "vừa phải",
        "ổn định",
        "có nhịp độ vừa phải"
    ],

    "fast": [
        "nhanh",
        "sôi động",
        "có tiết tấu nhanh"
    ],

    "very_fast": [
        "rất nhanh",
        "cực kỳ sôi động",
        "có tiết tấu rất nhanh"
    ]
}


ENERGY_TEXT = {

    "low": [
        "nhẹ nhàng",
        "êm dịu",
        "mềm mại"
    ],

    "medium": [
        "cân bằng",
        "vừa phải",
        "ổn định"
    ],

    "high": [
        "giàu năng lượng",
        "sôi động",
        "mạnh mẽ"
    ],

    "very_high": [
        "tràn đầy năng lượng",
        "cực kỳ mạnh mẽ",
        "rất sôi động"
    ]
}


DENSITY_TEXT = {

    "very_low": [
        "thưa",
        "tối giản"
    ],

    "low": [
        "thoáng",
        "không quá dày"
    ],

    "medium": [
        "cân bằng",
        "vừa phải"
    ],

    "high": [
        "dày",
        "phong phú",
        "nhiều chuyển động"
    ],

    "very_high": [
        "rất dày",
        "dồn dập",
        "nhiều chuyển động"
    ]
}


MOOD_TEXT = {

    "happy": [
        "vui vẻ",
        "tươi vui",
        "tươi sáng",
        "lạc quan"
    ],

    "bright": [
        "tươi sáng",
        "rạng rỡ",
        "tích cực"
    ],

    "positive": [
        "tích cực",
        "ấm áp",
        "tươi sáng"
    ],

    "sad": [
        "buồn",
        "u sầu",
        "trầm lắng"
    ],

    "melancholic": [
        "da diết",
        "trầm buồn",
        "man mác buồn"
    ],

    "dramatic": [
        "kịch tính",
        "ấn tượng",
        "có chiều sâu"
    ],

    "energetic": [
        "năng lượng cao",
        "mạnh mẽ",
        "sôi động"
    ],

    "lively": [
        "sống động",
        "tươi vui",
        "năng động"
    ],

    "calm": [
        "thư giãn",
        "êm dịu",
        "bình yên"
    ],

    "neutral": [
        "cân bằng",
        "trung tính",
        "ổn định"
    ]
}


# ============================================================
# INSTRUMENT TEXT
# ============================================================

def instrument_text(
    instrument_groups
) -> str:
    """
    Tạo mô tả nhạc cụ.

    Args:
        instrument_groups: list[str]

    Returns:
        str
    """

    if not instrument_groups:
        return "không xác định rõ nhạc cụ"

    if len(instrument_groups) == 1:

        return instrument_groups[0]

    if len(instrument_groups) == 2:

        return (
            f"{instrument_groups[0]} "
            f"và {instrument_groups[1]}"
        )

    return ", ".join(
        instrument_groups[:-1]
    ) + (
        " và "
        + instrument_groups[-1]
    )


# ============================================================
# CAPTION TEMPLATES
# ============================================================

def build_caption_templates(
    attributes
):
    """
    Tạo danh sách template caption.

    Các template cố tình đa dạng về cách diễn đạt
    để model không chỉ học một mẫu câu.

    Args:
        attributes: Musical attributes.

    Returns:
        list[str]
    """

    tempo = random.choice(
        TEMPO_TEXT[
            attributes["tempo_category"]
        ]
    )

    energy = random.choice(
        ENERGY_TEXT[
            attributes["energy"]
        ]
    )

    density = random.choice(
        DENSITY_TEXT[
            attributes["density"]
        ]
    )

    mood = random.choice(
        MOOD_TEXT[
            attributes["mood"]
        ]
    )

    instrument = instrument_text(
        attributes["instrument_groups"]
    )

    captions = []

    # --------------------------------------------------------
    # Caption 1
    # --------------------------------------------------------

    captions.append(
        f"Một bản nhạc {mood} "
        f"với nhịp độ {tempo}."
    )

    # --------------------------------------------------------
    # Caption 2
    # --------------------------------------------------------

    captions.append(
        f"Một giai điệu {mood}, "
        f"có {instrument} làm nhạc cụ nổi bật."
    )

    # --------------------------------------------------------
    # Caption 3
    # --------------------------------------------------------

    captions.append(
        f"Tạo một bản nhạc {mood}, "
        f"{energy} với tiết tấu {tempo}."
    )

    # --------------------------------------------------------
    # Caption 4
    # --------------------------------------------------------

    captions.append(
        f"Một bản nhạc {instrument} "
        f"có âm hưởng {mood} và "
        f"mật độ nốt {density}."
    )

    # --------------------------------------------------------
    # Caption 5
    # --------------------------------------------------------

    captions.append(
        f"Một giai điệu {mood} "
        f"với năng lượng {energy}."
    )

    # --------------------------------------------------------
    # Caption 6
    # --------------------------------------------------------

    captions.append(
        f"Hãy tạo một bản nhạc "
        f"{tempo}, {energy} và {mood}."
    )

    # --------------------------------------------------------
    # Caption 7
    # --------------------------------------------------------

    if attributes["mode"]:

        mode_text = (
            "trưởng"
            if attributes["mode"] == "major"
            else "thứ"
        )

        captions.append(
            f"Một bản nhạc ở giọng "
            f"{attributes['key']} {mode_text}, "
            f"có không khí {mood}."
        )

    else:

        captions.append(
            f"Một bản nhạc có không khí "
            f"{mood}, tiết tấu {tempo} "
            f"và năng lượng {energy}."
        )

    # --------------------------------------------------------
    # Caption 8
    # --------------------------------------------------------

    captions.append(
        f"Tạo một giai điệu {mood} "
        f"với {instrument}, "
        f"tiết tấu {tempo} "
        f"và phần nhạc {density}."
    )

    return captions


# ============================================================
# CLEAN CAPTIONS
# ============================================================

def clean_captions(
    captions
):
    """
    Loại bỏ caption trùng nhau.

    Args:
        captions: list[str]

    Returns:
        list[str]
    """

    cleaned = []

    seen = set()

    for caption in captions:

        caption = (
            caption
            .strip()
        )

        if not caption:
            continue

        key = caption.lower()

        if key in seen:
            continue

        seen.add(key)

        cleaned.append(
            caption
        )

    return cleaned


# ============================================================
# CREATE CAPTIONS
# ============================================================

def generate_captions(
    feature
):
    """
    Tạo caption cho một MIDI.

    Args:
        feature: Music feature dictionary.

    Returns:
        tuple:
            attributes,
            captions
    """

    attributes = extract_attributes(
        feature
    )

    # Sinh nhiều lần để thay đổi
    # các lựa chọn từ vựng.
    all_captions = []

    for _ in range(5):

        captions = build_caption_templates(
            attributes
        )

        all_captions.extend(
            captions
        )

    all_captions = clean_captions(
        all_captions
    )

    # Lấy số lượng caption yêu cầu
    all_captions = all_captions[
        :CAPTIONS_PER_MIDI
    ]

    return (
        attributes,
        all_captions
    )


# ============================================================
# BUILD DATASET
# ============================================================

def build_caption_dataset(
    features
):
    """
    Xây dựng dataset Text → Music.

    Args:
        features: Danh sách music features.

    Returns:
        list[dict]
    """

    dataset = []

    total = len(features)

    for index, feature in enumerate(
        features,
        start=1
    ):

        file_name = feature.get(
            "file_name",
            "unknown.midi"
        )

        print(
            f"[{index}/{total}] "
            f"{file_name}"
        )

        try:

            attributes, captions = (
                generate_captions(
                    feature
                )
            )

            item = {

                "music_file": file_name,

                "music_path": feature.get(
                    "file_path"
                ),

                "attributes": attributes,

                "captions": captions,

                "music_features": feature
            }

            dataset.append(
                item
            )

        except Exception as error:

            print(
                f"    [ERROR] {error}"
            )

    return dataset


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    dataset,
    output_file
):
    """
    Lưu caption dataset.

    Args:
        dataset: Dataset.
        output_file: Output JSON.
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
            dataset,
            file,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# PRINT SAMPLE
# ============================================================

def print_samples(
    dataset,
    number=3
):
    """
    In một vài sample để kiểm tra.

    Args:
        dataset: Dataset.
        number: Số sample.
    """

    print()
    print("=" * 60)
    print("SAMPLE CAPTIONS")
    print("=" * 60)

    for index, item in enumerate(
        dataset[:number],
        start=1
    ):

        print()
        print(
            f"Sample {index}:"
        )

        print(
            f"Music: "
            f"{item['music_file']}"
        )

        print(
            f"Attributes: "
            f"{item['attributes']}"
        )

        print(
            "Captions:"
        )

        for caption in item[
            "captions"
        ]:

            print(
                f"  - {caption}"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Entry point của chương trình.
    """

    random.seed(
        RANDOM_SEED
    )

    print("=" * 60)
    print("CAPTION GENERATOR")
    print("=" * 60)

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    features = load_json(
        FEATURE_FILE
    )

    print(
        f"Đã load {len(features)} MIDI features."
    )

    # --------------------------------------------------------
    # Generate captions
    # --------------------------------------------------------

    dataset = build_caption_dataset(
        features
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(
        dataset,
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    print_samples(
        dataset,
        number=3
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_captions = sum(
        len(item["captions"])
        for item in dataset
    )

    print()
    print("=" * 60)
    print("HOÀN THÀNH")
    print("=" * 60)

    print(
        f"Số MIDI      : {len(dataset)}"
    )

    print(
        f"Số caption   : {total_captions}"
    )

    print(
        f"Trung bình   : "
        f"{total_captions / max(len(dataset), 1):.2f} "
        f"caption/MIDI"
    )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()