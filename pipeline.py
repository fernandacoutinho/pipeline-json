import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
import librosa
import numpy as np
import torch
import whisperx

# Cores atribuídas automaticamente aos falantes no elenco (cast)
SPEAKER_COLORS = ["#6B8AFF", "#F5A623", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4"]


def extract_audio_from_video(video_path: str, audio_output_path: str):
    """Extrai o áudio do vídeo convertendo para WAV 16kHz Mono."""
    print("🎬 Extraindo áudio do vídeo...")
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
    """Classifica um bloco de áudio entre 'dialogue', 'music' e 'sfx' usando análise acústica."""
    if len(block_audio) == 0:
        return "dialogue"

    # 1. Spectral Flatness (Planura Espectral - ruídos e SFX possuem alta planura)
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=block_audio)))

    # 2. Zero Crossing Rate (Taxa de Cruzamento por Zero - alta em ruídos e cliques)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=block_audio)))

    # 3. Análise Harmônica vs Percussiva (Música costuma ter forte componente harmônico contínuo)
    y_harmonic = librosa.effects.harmonic(block_audio)
    harmonic_energy = float(np.mean(y_harmonic**2))
    total_energy = float(np.mean(block_audio**2)) + 1e-6
    harmonic_ratio = harmonic_energy / total_energy

    # Regras de classificação baseadas na física do áudio
    if flatness > 0.22 or zcr > 0.20:
        return "sfx"
    elif harmonic_ratio > 0.65 and flatness < 0.08:
        return "music"
    else:
        return "dialogue"


def calculate_word_metrics(word_audio: np.ndarray, sr: int, max_rms: float) -> dict:
    """Calcula volume relativo e weight baseado no pico global de áudio do vídeo."""
    if len(word_audio) == 0:
        return {"volumePercent": 50, "weight": 600, "width": 78, "italic": False}

    # 1. Energia RMS da palavra
    rms = float(np.sqrt(np.mean(word_audio**2)))

    # 2. Volume relativo normalizado de 0% a 100%
    volume_percent = int(min(1.0, (rms / max_rms)) * 100)

    # 3. Mapeamento de Font Weight pelo volume
    if volume_percent <= 20:
        weight = 300  # Sussurro (fonte leve/fina)
    elif volume_percent <= 45:
        weight = 400  # Fala suave
    elif volume_percent <= 75:
        weight = 600  # Fala normal / padrão firme
    elif volume_percent <= 88:
        weight = 800  # Destaque
    else:
        weight = 900  # Grito / Ênfase máxima

    return {
        "volumePercent": volume_percent,
        "weight": weight,
        "width": 78,
        "italic": False,  # Mantido sempre False
    }


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

        # 2. Leitura com Librosa para análise de sinal
        y, sr = librosa.load(temp_audio, sr=16000)
        total_duration = float(librosa.get_duration(y=y, sr=sr))

        rms_all = librosa.feature.rms(y=y)[0]
        max_rms = float(np.percentile(rms_all, 95)) + 1e-6

        # 3. Transcrição e Alinhamento Preciso de Palavras com WhisperX
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

        # 4. Estruturação dos Blocos e Palavras
        captions = []
        speaker_map = {}
        speaker_counter = 0

        for seg_idx, segment in enumerate(aligned_result["segments"]):
            words = segment.get("words", [])
            if not words:
                continue

            # Identificação do Falante (Usa a chave de diarização se existir, senão usa S0)
            speaker_id = segment.get("speaker")
            if not speaker_id:
                speaker_id = f"S{speaker_counter}"

            if speaker_id not in speaker_map:
                speaker_map[speaker_id] = f"S{len(speaker_map)}"

            mapped_speaker_id = speaker_map[speaker_id]

            block_start = round(words[0]["start"], 2)
            block_end = round(words[-1]["end"], 2)

            # Recorte do áudio do bloco para classificação (dialogue / music / sfx)
            start_sample_block = int(block_start * sr)
            end_sample_block = int(block_end * sr)
            block_type = classify_block_type(y[start_sample_block:end_sample_block], sr)

            words_data = []
            for w in words:
                if "start" not in w or "end" not in w:
                    continue

                w_start = round(w["start"], 2)
                w_end = round(w["end"], 2)

                # Amostra do áudio da palavra individual
                start_sample_word = int(w_start * sr)
                end_sample_word = int(w_end * sr)
                word_audio = y[start_sample_word:end_sample_word]

                metrics = calculate_word_metrics(word_audio, sr, max_rms)

                words_data.append(
                    {
                        "text": w["word"].strip(),
                        "start": w_start,
                        "end": w_end,
                        "weight": metrics["weight"],
                        "volumePercent": metrics["volumePercent"],
                        "width": metrics["width"],
                        "italic": metrics["italic"],
                    }
                )

            if words_data:
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

        # 5. Criação do Elenco (Cast)
        cast = []
        for idx, spk in enumerate(speaker_map.values()):
            cast.append(
                {
                    "id": spk,
                    "name": f"Speaker {idx}",
                    "color": SPEAKER_COLORS[idx % len(SPEAKER_COLORS)],
                }
            )

        # 6. Montagem do JSON final no Schema CWI 1.0
        final_json = {
            "$schema": "https://opencaptions.tools/schema/cwi/1.0.json",
            "version": "1.0",
            "metadata": {
                "duration": round(total_duration, 1),
                "language": language,
                # CORREÇÃO: Uso de datetime.now(timezone.utc)
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "generator": "opencaptions/0.1.0",
                "extractor_backend": "audio-vision-v1",
            },
            "cast": cast,
            "captions": captions,
        }

        # Salva o arquivo JSON
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)

    finally:
        # Limpa o áudio temporário extraído
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