import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
import librosa
import numpy as np
import torch
import whisperx

SPEAKER_COLORS = ["#6B8AFF", "#F5A623", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4"]

def extract_audio_from_video(video_path: str, audio_output_path: str):
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_output_path,
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def classify_block_type(block_audio: np.ndarray, sr: int) -> str:
    """Classifica o bloco entre 'dialogue', 'music' e 'sfx'."""
    if len(block_audio) == 0:
        return "dialogue"

    total_energy = float(np.mean(block_audio**2))
    if total_energy < 1e-4:
        return "dialogue"

    flatness = float(np.mean(librosa.feature.spectral_flatness(y=block_audio)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=block_audio)))

    y_harmonic = librosa.effects.harmonic(block_audio)
    harmonic_energy = float(np.mean(y_harmonic**2))
    harmonic_ratio = harmonic_energy / (total_energy + 1e-6)

    if flatness > 0.25 or zcr > 0.22:
        return "sfx"
    elif harmonic_ratio > 0.70 and flatness < 0.06:
        return "music"
    else:
        return "dialogue"

def get_word_db(word_audio: np.ndarray) -> float:
    """Calcula o nível de energia em dB da palavra."""
    if len(word_audio) == 0:
        return -60.0

    frame_length = min(len(word_audio), 256)
    if frame_length < 64:
        rms = float(np.sqrt(np.mean(word_audio**2)))
    else:
        rms_frames = librosa.feature.rms(y=word_audio, frame_length=frame_length, hop_length=64)[0]
        rms = float(np.percentile(rms_frames, 80))

    return 20 * np.log10(max(rms, 1e-6))

def process_video_to_json(
    video_path: str,
    output_json_path: str,
    language: str = "pt",
    model_size: str = "small",
):
    temp_audio = "temp_extracted_audio.wav"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # 1. Extração de Áudio
        extract_audio_from_video(video_path, temp_audio)

        # 2. Leitura e Normalização do Áudio Completo
        y, sr = librosa.load(temp_audio, sr=16000)
        total_duration = float(librosa.get_duration(y=y, sr=sr))

        y = y - np.mean(y)
        y = librosa.util.normalize(y, norm=np.inf) * 0.90

        # Média Global de referência
        rms_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        active_rms = rms_frames[rms_frames > np.percentile(rms_frames, 20)]
        db_speech = 20 * np.log10(np.maximum(active_rms, 1e-6))
        global_mean_db = float(np.median(db_speech)) if len(db_speech) > 0 else -20.0

        # 3. Transcrição e Alinhamento
        whisper_model = whisperx.load_model(
            model_size, device, compute_type="float32", language=language
        )
        audio_data = whisperx.load_audio(temp_audio)
        result = whisper_model.transcribe(audio_data, batch_size=16)

        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        aligned_result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio_data,
            device,
            return_char_alignments=False,
        )

        captions = []
        mapped_speaker_id = "S0"

        # 4. Processamento dos Blocos com Normalização Suave
        for seg_idx, segment in enumerate(aligned_result["segments"]):
            words = segment.get("words", [])
            if not words:
                continue

            block_start = round(words[0]["start"], 2)
            block_end = round(words[-1]["end"], 2)

            start_sample_block = int(block_start * sr)
            end_sample_block = int(block_end * sr)
            block_audio = y[start_sample_block:end_sample_block]

            block_type = classify_block_type(block_audio, sr)

            # Média do bloco para o volumePercent
            if len(block_audio) > 0:
                block_rms_frames = librosa.feature.rms(y=block_audio, frame_length=1024, hop_length=256)[0]
                active_block_rms = block_rms_frames[block_rms_frames > 1e-4]
                block_mean_db = (
                    float(np.median(20 * np.log10(active_block_rms)))
                    if len(active_block_rms) > 0
                    else global_mean_db
                )
            else:
                block_mean_db = global_mean_db

            # Pré-calcular dB de cada palavra para extrair estatísticas locais
            block_words_info = []
            words_db_list = []

            for w in words:
                if "start" not in w or "end" not in w:
                    continue
                w_start = round(w["start"], 2)
                w_end = round(w["end"], 2)

                start_sample_word = int(w_start * sr)
                end_sample_word = int(w_end * sr)
                word_audio = y[start_sample_word:end_sample_word]

                w_db = get_word_db(word_audio)
                words_db_list.append(w_db)
                block_words_info.append({
                    "word": w["word"].strip(),
                    "start": w_start,
                    "end": w_end,
                    "db": w_db
                })

            if not block_words_info:
                continue

            # Estatísticas relativas do bloco
            median_db = float(np.median(words_db_list))
            max_db = float(np.max(words_db_list))
            min_db = float(np.min(words_db_list))

            # Mapeamento suave de volumePercent e weight
            words_data = []
            for item in block_words_info:
                w_db = item["db"]

                # A. Cálculo do volumePercent (escala moderada de 4x por dB)
                delta_ref = (0.7 * (w_db - block_mean_db)) + (0.3 * (w_db - global_mean_db))
                raw_vol = 50.0 + (delta_ref * 4.0)
                volume_percent = int(np.clip(round(raw_vol), 20, 95))

                # B. Normalização Suave da espessura com base na Mediana e Máximo
                if max_db == min_db:
                    weight = 400
                elif w_db >= median_db:
                    # Sobem da mediana (400) até o pico do bloco (700)
                    headroom = max(max_db - median_db, 1.5)
                    ratio = np.clip((w_db - median_db) / headroom, 0.0, 1.0)
                    raw_weight = 400.0 + (ratio * 300.0)
                    weight = int(round(raw_weight / 50.0) * 50)
                else:
                    # Caem da mediana (400) até o piso do bloco (300)
                    floorroom = max(median_db - min_db, 1.5)
                    ratio = np.clip((median_db - w_db) / floorroom, 0.0, 1.0)
                    raw_weight = 400.0 - (ratio * 100.0)
                    weight = int(round(raw_weight / 50.0) * 50)

                weight = int(np.clip(weight, 300, 800))

                words_data.append(
                    {
                        "text": item["word"],
                        "start": item["start"],
                        "end": item["end"],
                        "weight": weight,
                        "volumePercent": volume_percent,
                        "width": 78,
                        "italic": False,
                    }
                )

            captions.append(
                {
                    "id": f"block-{seg_idx + 1}-{block_type}-{mapped_speaker_id.lower()}",
                    "start": block_start,
                    "end": block_end,
                    "speaker_id": mapped_speaker_id,
                    "type": block_type,
                    "words": words_data,
                }
            )

        # 5. Elenco
        cast = [
            {
                "id": "S0",
                "name": "Speaker 0",
                "color": SPEAKER_COLORS[0],
            }
        ]

        # 6. JSON Final
        final_json = {
            "$schema": "https://opencaptions.tools/schema/cwi/1.0.json",
            "version": "1.0",
            "metadata": {
                "duration": round(total_duration, 1),
                "language": language,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "generator": "opencaptions/0.1.0",
                "extractor_backend": "audio-vision-v1",
            },
            "cast": cast,
            "captions": captions,
        }

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)

    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera legendas JSON OpenCaptions CWI 1.0 a partir de vídeo."
    )
    parser.add_argument("--video", required=True, help="Caminho do arquivo de vídeo de entrada")
    parser.add_argument(
        "--output", default="legendas.json", help="Caminho do JSON de saída"
    )
    parser.add_argument("--lang", default="pt", help="Código do idioma (ex: pt, en)")

    args = parser.parse_args()
    process_video_to_json(args.video, args.output, language=args.lang)