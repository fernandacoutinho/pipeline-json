import os
import sys
import warnings
import logging
import argparse
import json
import subprocess
from datetime import datetime, timezone
from contextlib import redirect_stdout, redirect_stderr

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.disable(logging.CRITICAL)

logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("whisperx").setLevel(logging.ERROR)

import librosa
import numpy as np
import torch
import whisperx

SPEAKER_COLORS = ["#6B8AFF", "#F5A623", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4"]


def extract_audio_from_video(video_path: str, audio_output_path: str):
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-af", "highpass=f=80,lowpass=f=8000,loudnorm",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_output_path,
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def classify_block_type(block_audio: np.ndarray, sr: int) -> str:
    if len(block_audio) == 0:
        return "dialogue"

    total_energy = float(np.mean(block_audio**2))
    if total_energy < 1e-4:
        return "dialogue"

    flatness = float(np.mean(librosa.feature.spectral_flatness(y=block_audio)))

    y_harmonic = librosa.effects.harmonic(block_audio)
    harmonic_energy = float(np.mean(y_harmonic**2))
    harmonic_ratio = harmonic_energy / (total_energy + 1e-6)

    if flatness > 0.45:
        return "sfx"
    elif harmonic_ratio > 0.75 and flatness < 0.05:
        return "music"
    else:
        return "dialogue"


def get_word_db(word_audio: np.ndarray) -> float:
    if len(word_audio) < 64:
        return -60.0

    frame_length = min(len(word_audio), 256)
    try:
        rms_frames = librosa.feature.rms(
            y=word_audio, frame_length=frame_length, hop_length=64
        )[0]
        rms = float(np.percentile(rms_frames, 80))
    except Exception:
        rms = float(np.sqrt(np.mean(word_audio**2)))

    if np.isnan(rms) or rms <= 0:
        return -60.0

    return float(20 * np.log10(max(rms, 1e-6)))


def process_video_to_json(
    video_path: str,
    output_json_path: str,
    language: str = None,
    model_size: str = "small",
    hf_token: str = None,
    min_speakers: int = None,
    max_speakers: int = None,
):
    # Se o token não for passado por parâmetro na função, busca do arquivo .env
    hf_token = hf_token or os.getenv("HF_TOKEN")

    temp_audio = "temp_extracted_audio.wav"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # 1. Extração de áudio limpo
        extract_audio_from_video(video_path, temp_audio)

        # 2. Leitura e normalização do áudio completo
        y, sr = librosa.load(temp_audio, sr=16000)
        total_duration = float(librosa.get_duration(y=y, sr=sr))

        y = y - np.mean(y)
        y = librosa.util.normalize(y, norm=np.inf) * 0.90

        # Média global de referência
        rms_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        active_rms = rms_frames[rms_frames > np.percentile(rms_frames, 20)]
        db_speech = 20 * np.log10(np.maximum(active_rms, 1e-6))
        global_mean_db = float(np.median(db_speech)) if len(db_speech) > 0 else -20.0

        # 3. Transcrição e Alinhamento
        vad_options = {
            "vad_onset": 0.20,
            "vad_offset": 0.15
        }

        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
            whisper_model = whisperx.load_model(
                model_size,
                device,
                compute_type="float32",
                language=language,
                vad_options=vad_options
            )
            audio_data = whisperx.load_audio(temp_audio)
            result = whisper_model.transcribe(audio_data, batch_size=16)

            detected_language = result["language"]

            align_model, metadata = whisperx.load_align_model(
                language_code=detected_language, device=device
            )
            aligned_result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio_data,
                device,
                return_char_alignments=False,
            )

            # 4. Diarização
            if hf_token:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token, 
                    device=device
                )
                diarize_segments = diarize_model(
                    audio_data, 
                    min_speakers=min_speakers, 
                    max_speakers=max_speakers
                )
                aligned_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)

        captions = []
        speaker_map = {}

        def get_mapped_speaker_id(raw_speaker: str) -> str:
            if not raw_speaker:
                raw_speaker = "SPEAKER_00"
            if raw_speaker not in speaker_map:
                speaker_map[raw_speaker] = f"S{len(speaker_map)}"
            return speaker_map[raw_speaker]

        # 5. Processamento mantendo tempos e falantes
        for seg_idx, segment in enumerate(aligned_result["segments"]):
            words = segment.get("words", [])
            valid_words = [w for w in words if "start" in w and "end" in w]
            if not valid_words:
                continue

            raw_speaker = segment.get("speaker", "SPEAKER_00")
            mapped_speaker_id = get_mapped_speaker_id(raw_speaker)

            block_start = round(float(valid_words[0]["start"]), 2)
            block_end = round(float(valid_words[-1]["end"]), 2)

            start_sample_block = max(0, int(block_start * sr))
            end_sample_block = min(len(y), int(block_end * sr))
            block_audio = y[start_sample_block:end_sample_block]

            block_type = classify_block_type(block_audio, sr)

            words_data = []
            for w in valid_words:
                w_start = float(w["start"])
                w_end = float(w["end"])

                start_sample_word = max(0, int(w_start * sr))
                end_sample_word = min(len(y), int(w_end * sr))

                if end_sample_word - start_sample_word < 512:
                    end_sample_word = min(len(y), start_sample_word + 512)

                word_audio = y[start_sample_word:end_sample_word]
                w_db = get_word_db(word_audio)

                db_diff = w_db - global_mean_db
                volume_percent = int(np.clip(50 + (db_diff * 3.5), 10, 100))

                if len(word_audio) >= 256:
                    peak_rms = float(np.max(np.abs(word_audio)))
                    try:
                        sc = librosa.feature.spectral_centroid(y=word_audio, sr=sr)
                        spectral_centroid = float(np.nanmean(sc))
                        if np.isnan(spectral_centroid):
                            spectral_centroid = 2000.0
                    except Exception:
                        spectral_centroid = 2000.0

                    emphasis_score = (peak_rms * 0.6) + ((spectral_centroid / 4000.0) * 0.4)
                else:
                    emphasis_score = 0.5

                if volume_percent < 25 and emphasis_score < 0.3:
                    word_type = "whisper"
                    weight = 200
                elif volume_percent > 75 and emphasis_score > 0.8:
                    word_type = "shout"
                    weight = 900
                else:
                    word_type = "normal"
                    raw_w = 300 + (emphasis_score * 500)
                    weight = int(round(raw_w / 100.0) * 100)
                    weight = int(np.clip(weight, 300, 800))

                words_data.append(
                    {
                        "text": w["word"].strip(),
                        "start": round(w_start, 2),
                        "end": round(w_end, 2),
                        "weight": weight,
                        "volumePercent": volume_percent,
                        "type": word_type,
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

        # Monta a lista de falantes para o JSON
        cast = []
        for raw_spk, mapped_id in speaker_map.items():
            spk_index = int(mapped_id.replace("S", ""))
            color = SPEAKER_COLORS[spk_index % len(SPEAKER_COLORS)]
            cast.append({
                "id": mapped_id,
                "name": f"Speaker {spk_index}",
                "color": color
            })

        if not cast:
            cast = [{"id": "S0", "name": "Speaker 0", "color": SPEAKER_COLORS[0]}]

        final_json = {
            "$schema": "https://opencaptions.tools/schema/cwi/1.0.json",
            "version": "1.0",
            "metadata": {
                "duration": round(total_duration, 1),
                "language": detected_language,
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
    parser.add_argument(
        "--lang", default=None, help="Código do idioma (ex: pt, en). Se omitido, detecta automaticamente."
    )
    parser.add_argument(
        "--hf_token", default=None, help="Token do Hugging Face (Sobrescreve a variável do .env)"
    )
    parser.add_argument(
        "--min_speakers", type=int, default=None, help="Número mínimo de falantes para diarização."
    )
    parser.add_argument(
        "--max_speakers", type=int, default=None, help="Número máximo de falantes para diarização."
    )

    args = parser.parse_args()
    process_video_to_json(
        args.video, 
        args.output, 
        language=args.lang,
        hf_token=args.hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers
    )