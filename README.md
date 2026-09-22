# Pipeline JSON — Processador de Áudio e Gerador de Legendas OpenCaptions CWI 1.0

Aplicação em Python para extração de áudio, análise acústica e transcrição automatizada de mídias de vídeo. A ferramenta realiza o alinhamento temporal a nível de palavra via **WhisperX**, classifica os segmentos de áudio e analisa parâmetros dinâmicos da fala através de métricas espectrais e RMS com **Librosa**, exportando os resultados no formato padronizado **OpenCaptions CWI 1.0**.

---

## Arquitetura e Funcionalidades

- **Pré-processamento de Áudio**: Converte o fluxo de áudio do arquivo de vídeo para o formato PCM 16-bit mono a 16 kHz, aplicando filtros de passagem de banda (80 Hz a 8000 Hz) e normalização de volume via `ffmpeg`.
- **Transcrição e Alinhamento Temporal**: Realiza o reconhecimento de fala e o alinhamento forçado por palavra utilizando o modelo WhisperX.
- **Classificação Acústica de Blocos**: Avalia métricas de planicidade espectral e razão harmônica para categorizar os segmentos em `dialogue`, `music` ou `sfx`.
- **Análise Dinâmica da Fala**:
  - Medição do nível de volume relativo por palavra (`volumePercent`).
  - Classificação do modo de articulação em `whisper`, `normal` ou `shout`.
  - Cálculo de peso visual (`weight`) baseado no pico de energia RMS e no centroide espectral.
- **Estruturação OpenCaptions**: Serialização dos dados segundo a especificação JSON OpenCaptions CWI 1.0.

---

## Requisitos do Sistema

- **Python**: Versão 3.10 ou superior.
- **FFmpeg**: Utilitário de linha de comando para manipulação e codificação de áudio/vídeo.
- **Aceleração por Hardware (Recomendado)**: GPU NVIDIA com suporte a CUDA para execução otimizada do WhisperX.

---

## Instalação

### 1. Clonagem do Repositório

```bash
git clone [https://github.com/fernandacoutinho/pipeline-json.git](https://github.com/fernandacoutinho/pipeline-json.git)
cd pipeline-json
```

### 2. Instalação do FFmpeg

O binário do `ffmpeg` deve estar configurado no `PATH` do sistema operacional.

- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
- **macOS**:
  ```bash
  brew install ffmpeg
  ```
- **Windows**:
  ```powershell
  winget install FFmpeg
  ```

### 3. Configuração do Ambiente Virtual

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 4. Instalação das Dependências

Instale o PyTorch com suporte a GPU (ajuste a URL conforme a versão CUDA disponível na máquina):

```bash
pip install torch torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

Em seguida, instale as bibliotecas de processamento:

```bash
pip install git+[https://github.com/m-bain/whisperX.git](https://github.com/m-bain/whisperX.git)
pip install librosa numpy
```

---

## Instruções de Uso

A execução do pipeline é realizada via interface de linha de comando (CLI).

### Execução Básica (Detecção Automática de Idioma)

```bash
python main.py --video caminho/do/video.mp4
```

### Execução com Parâmetros Definidos

```bash
python main.py --video caminho/do/video.mp4 --output resultado.json --lang pt
```

---

## Opções da CLI

| Parâmetro | Tipo | Obrigatório | Padrão | Descrição |
| :--- | :---: | :---: | :---: | :--- |
| `--video` | String | **Sim** | — | Caminho do arquivo de vídeo de entrada. |
| `--output` | String | Não | `legendas.json` | Caminho do arquivo JSON de saída. |
| `--lang` | String | Não | `None` | Código ISO do idioma (ex: `pt`, `en`). Quando omitido, o idioma é detectado automaticamente. |

---

## Estrutura do JSON de Saída

Exemplo da estrutura do payload gerado no padrão **OpenCaptions CWI 1.0**:

```json
{
  "$schema": "[https://opencaptions.tools/schema/cwi/1.0.json](https://opencaptions.tools/schema/cwi/1.0.json)",
  "version": "1.0",
  "metadata": {
    "duration": 12.4,
    "language": "pt",
    "created_at": "2026-03-31T16:00:00.000Z",
    "generator": "opencaptions/0.1.0",
    "extractor_backend": "audio-vision-v1"
  },
  "cast": [
    {
      "id": "S0",
      "name": "Speaker 0",
      "color": "#6B8AFF"
    }
  ],
  "captions": [
    {
      "id": "block-1-dialogue-s0",
      "start": 0.12,
      "end": 1.85,
      "speaker_id": "S0",
      "type": "dialogue",
      "words": [
        {
          "text": "Exemplo",
          "start": 0.12,
          "end": 0.65,
          "weight": 500,
          "volumePercent": 60,
          "type": "normal",
          "width": 78,
          "italic": false
        }
      ]
    }
  ]
}
```
