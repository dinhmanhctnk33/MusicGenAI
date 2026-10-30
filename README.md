# MusicGenAI

MusicGenAI là dự án Transformer encoder–decoder sinh chuỗi music token từ mô tả văn bản. Encoder xử lý caption, decoder autoregressive sinh token nhạc và kết quả được chuyển thành MIDI/audio.

## Thành phần repository

- `model.py`: kiến trúc Text-to-Music Transformer.
- `train.py`: chia dữ liệu theo MIDI, huấn luyện, validation và checkpoint.
- `generate.py`: sinh music token và xuất MIDI từ prompt.
- `midi_preprocess.py`: chuyển MIDI thành music token.
- `music_feature_extractor.py`: trích xuất đặc trưng MIDI.
- `caption_generator.py`: tạo caption từ đặc trưng.
- `text_dataset.py`: tạo vocabulary và dataset text-to-music.
- `processed_data/`: dataset đã xử lý và các vocabulary cần để train.

Dataset gốc MAESTRO, checkpoint, soundfont và file audio sinh ra không được commit vì dung lượng lớn. Dataset huấn luyện đã xử lý nằm tại `processed_data/text_music_dataset.json`.

## Yêu cầu

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- Git LFS (để tải các file dataset lớn)
- GPU NVIDIA được khuyến nghị nhưng không bắt buộc

## Cài đặt

```bash
git clone https://github.com/dinhmanhctnk33/MusicGenAI.git
cd MusicGenAI
git lfs pull
uv sync
```

Kích hoạt môi trường nếu không dùng `uv run` trực tiếp:

```powershell
.venv\Scripts\Activate.ps1
```

## Huấn luyện

Repository đã chứa dataset train, `text_vocab.json` và `music_vocab.json`:

```bash
uv run python train.py
```

Checkpoint được ghi vào:

- `checkpoints/best_model.pt`
- `checkpoints/last_model.pt`

## Sinh nhạc

Sau khi có `checkpoints/best_model.pt`:

```bash
uv run python generate.py
```

Các tham số prompt, sampling, đường dẫn checkpoint và output được cấu hình ở đầu `generate.py`.

Để chuyển MIDI thành audio, tải một General MIDI SoundFont, đặt file `.sf2` trong thư mục `soundfont/`, rồi cấu hình đường dẫn trong `midi_to_audio.py`.

## Tạo lại dataset

Pipeline đầy đủ:

```bash
uv run python 02_clean_midi.py
uv run python midi_preprocess.py
uv run python music_feature_extractor.py
uv run python caption_generator.py
uv run python text_dataset.py
```

Dataset gốc có thể tải từ [MAESTRO](https://magenta.tensorflow.org/datasets/maestro). Đặt dữ liệu vào `dataset/` trước khi chạy pipeline và kiểm tra các hằng đường dẫn ở đầu từng script.

Khi chuỗi nhạc vượt giới hạn 512 token, pipeline luôn giữ `EOS` ở vị trí cuối. Train/validation được chia theo `music_file` để caption của cùng một MIDI không bị rò rỉ giữa hai tập.

## Ghi chú tài nguyên

Self-attention trên chuỗi 511 token cần khá nhiều bộ nhớ. Nếu gặp lỗi hết VRAM, giảm `BATCH_SIZE` trong `train.py`. Mixed precision tự động được bật khi CUDA khả dụng.
