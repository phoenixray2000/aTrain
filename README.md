<img src="https://github.com/BANDAS-Center/aTrain/blob/main/docs/images/logo.svg" width="300" alt="Logo">

## Accessible Transcription of Interviews
aTrain is a tool for automatically transcribing speech recordings utilizing state-of-the-art machine learning models without uploading any data. It was developed by researchers at the Business Analytics and Data Science-Center at the University of Graz and tested by researchers from the Know-Center Graz.


<p>
  <a href="https://flathub.org/apps/io.github.juergenfleiss.aTrain">
    <img height="58" alt="Get it on Flathub" src="https://flathub.org/api/badge?locale=en">
  </a>
  &nbsp;&nbsp;
  <a href="https://apps.microsoft.com/detail/9N15Q44SZNS2?mode=direct">
    <img width="220" alt="Get it from Microsoft" src="https://get.microsoft.com/images/en-us%20dark.svg">
  </a>
</p>


## Command-Line Interface (CLI)

aTrain also ships a Typer-based CLI for scripted transcription workflows. The CLI uses the same transcription engine and local model files as the GUI, but it does not open the GUI.

Use one of these entrypoints depending on how aTrain is installed:

```powershell
aTrain-cli --help
aTrain-cli.exe --help
python -m aTrain.cli --help
```

The CLI provides these commands:

```powershell
aTrain-cli init
aTrain-cli transcribe INPUT [OPTIONS]
aTrain-cli voiceprint enroll [OPTIONS]
```

`aTrain-cli init` downloads the default transcription model and speaker-detection model used by `transcribe`. `INPUT` can be a single audio/video file or a directory. Directory input scans only the top-level directory by default; pass `--recursive` to include subdirectories. Outputs are copied from a temporary transcription workspace into the selected output directory. Existing output files are kept by default; pass `--overwrite` to replace them.

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `--model` | string | `large-v3` | Whisper model name. |
| `--language` | string | `auto-detect` | Language code or `auto-detect`. |
| `--speaker-detection / --no-speaker-detection` | bool | `True` | Enables pyannote speaker detection. |
| `--speaker-count` | integer | `0` | `0` means auto-detect speaker count. |
| `--identify-speakers / --no-identify-speakers` | bool | `True` | Renames diarized `SPEAKER_xx` labels with enrolled voiceprints. Requires `--speaker-detection`; no-ops when no voiceprints are enrolled. |
| `--voiceprint-threshold` | float | `0.5` | Minimum cosine similarity required for a voiceprint match. |
| `--voiceprint-margin` | float | `0.05` | Minimum score gap over competing speaker/name assignments. |
| `--speaker-embeddings-output` | file | `None` | Writes captured per-speaker embeddings to an `.npz` file. Requires `--speaker-detection` and `--identify-speakers`; single-file input only. |
| `--device` | `cpu`, `gpu` | `gpu` | Hardware backend. |
| `--compute-type` | `int8`, `float16`, `float32` | `float32` | Model compute precision. |
| `--temperature` | float | `None` | Optional sampling temperature, `0.0` to `1.0`. |
| `--prompt` | string | `None` | Optional initial prompt for Whisper. |
| `--cpu-threads` | integer | `aTrain_core.globals.DEFAULT_CPU_THREADS` | `0` means automatic CPU thread selection. |
| `--recursive / --no-recursive` | bool | `False` | Applies only when `INPUT` is a directory. |
| `--formats` | CSV | `txt,timestamps` | Allowed values: `json`, `txt`, `timestamps`, `maxqda`, `srt`. |
| `--output` | directory | `./atrain-output` | Fallback output directory for all selected formats. |
| `--json-output` | directory | fallback to `--output` | Dedicated directory for JSON output. |
| `--txt-output` | directory | fallback to `--output` | Dedicated directory for plain text output. |
| `--timestamps-output` | directory | fallback to `--output` | Dedicated directory for timestamped text output. |
| `--maxqda-output` | directory | fallback to `--output` | Dedicated directory for MAXQDA output. |
| `--srt-output` | directory | fallback to `--output` | Dedicated directory for SRT output. |
| `--overwrite / --no-overwrite` | bool | `False` | Existing target files are kept by default; use `--overwrite` to replace them. |

### Output Contract

Output filenames are derived from the input file stem. For an input file named `interview01.wav`, the selected formats are written as:

| Format | Output filename |
| --- | --- |
| `json` | `interview01.json` |
| `txt` | `interview01.txt` |
| `timestamps` | `interview01_timestamps.txt` |
| `maxqda` | `interview01_maxqda.txt` |
| `srt` | `interview01.srt` |

For recursive directory input, the input folder's relative subdirectory structure is preserved below each output directory. This prevents collisions when different subdirectories contain files with the same stem. Top-level directory input without `--recursive` writes all selected files directly into the chosen output directories.

### Model Initialization

Use `init` to download models for both CLI and GUI use:

```powershell
aTrain-cli init large-v3
aTrain-cli init speaker-detection
aTrain-cli init all
```

Because `transcribe` defaults to `--model large-v3` and `--speaker-detection`, a fresh environment needs both `large-v3` and `speaker-detection` before the default transcription command can run. A model is treated as available when its model directory exists and contains at least one `.bin` file, including nested `.bin` files.

### Speaker Voiceprints

The GUI provides a `Voiceprints` page for enrolling and managing persistent speaker profiles. Each profile is stored as a JSON file below the local aTrain data directory's `voiceprints` folder. Enrollment uses the local `speaker-detection/embedding` model; it does not upload reference audio.

### CLI voiceprint enrollment

The CLI can create or update local voiceprint profiles. Profiles are stored in the same local voiceprint directory used by the GUI, and `transcribe --identify-speakers` consumes those profiles during later transcription runs.

Enroll from a direct audio sample:

```powershell
aTrain-cli voiceprint enroll --name "李想" --audio "D:\samples\li-xiang.wav" --update
```

Enroll from a captured speaker embedding exported during transcription:

```powershell
aTrain-cli transcribe "D:\input\meeting.wav" --speaker-detection --identify-speakers --speaker-embeddings-output "D:\out\meeting.speaker-embeddings.npz"
aTrain-cli voiceprint enroll --name "李想" --speaker-embeddings "D:\out\meeting.speaker-embeddings.npz" --speaker SPEAKER_01 --update
```

If a diarized speaker matches an enrolled profile above `--voiceprint-threshold` and above the competing-match `--voiceprint-margin`, output speaker fields are rewritten from labels such as `SPEAKER_00` to the enrolled name. Low-confidence matches remain as `SPEAKER_xx`; tune `--voiceprint-threshold` and `--voiceprint-margin` when needed.

### CLI Examples

Transcribe one file with the default outputs:

```powershell
aTrain-cli transcribe "D:\media\interview01.wav" --output "D:\transcripts"
```

Use `aTrain-cli transcribe --help` for model, language, speaker detection, device, compute type, format, and output-directory options.


## About aTrain

aTrain offers the following benefits:
\
\
**Fast and accurate 🚀**
\
aTrain provides a user friendly access to the [faster-whisper](https://github.com/guillaumekln/faster-whisper) implementation of OpenAI’s [Whisper model](https://github.com/openai/whisper), ensuring best in class transcription quality (see [Wollin-Geiring et al. 2023](https://www.static.tu.berlin/fileadmin/www/10005401/Publikationen_sos/Wollin-Giering_et_al_2023_Automatic_transcription.pdf)) paired with higher speeds on your local computer. Transcription when selecting the highest-quality model takes only around three times the audio length on current mobile CPUs typically found in middle-class business notebooks (e.g., Core i5 12th Gen, Ryzen Series 6000).
\
\
**Speaker detection 🗣️**
\
aTrain has a speaker detection mode based on [pyannote.audio](https://github.com/pyannote/pyannote-audio) and can analyze each text segment to determine which speaker it belongs to.
The GUI can also enroll local speaker voiceprints from reference audio and use them to rename diarized speaker labels when a confident match is found.
\
\
**Privacy Preservation and GDPR compliance 🔒**
\
aTrain processes the provided speech recordings completely offline on your own device and does not send recordings or transcriptions to the internet. This helps researchers to maintain data privacy requirements arising from ethical guidelines or to comply with legal requirements such as the GDPR.
\
\
**Multi-language support 🌍**
\
aTrain-core can process speech recordings a total of 99 languages, including Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh. A full list can be found [here](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py). Note that transcription quality varies with language; word error rates for the different languages can be found [here](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages).
\
\
**MAXQDA, ATLAS.ti and nVivo compatible output 📄**
\
aTrain-core provides transcription files that are seamlessly importable into the most popular tools for qualitative analysis, ATLAS.ti, MAXQDA and nVivo. This allows you to directly play audio for the corresponding text segment by clicking on its timestamp. Go to the [tutorial](https://github.com/BANDAS-Center/aTrain/wiki/Tutorials) for MAXQDA.
\
\
**Nvidia GPU support 🖥️**
\
aTrain can either run on the CPU or an NVIDIA GPU (CUDA toolkit installation required). A [CUDA-enabled NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) significantly improves the speed of transcriptions and speaker detection, reducing transcription time to 20% of audio length on current entry-level gaming notebooks.

| Screenshot 1 | Screenshot 2 |
| --- | --- |
| ![Screenshot1](docs/images/screenshot_1.webp) | ![Screenshot2](docs/images/screenshot_2.webp) |

## Benchmarks
For testing the processing time of aTrain-core we transcribe a [conversation between Christine Lagarde and Andrea Enria at the Fifth ECB Forum on Banking Supervision 2023](https://www.youtube.com/watch?v=kd7e3OXkajY) published on YouTube by the European Central Bank under a Creative Commons license , downloaded as 320p MP4 video file. The file has a duration of exactly 22 minutes and was transcribed on different computing devices with speaker detection enabled. The figure below shows the processing time of each transcription.

Transcription Time (incl. speaker detection) for 00:22:00 File:

| Computing Device       |  large-v3   | Distil large-v3   | large-v3-turbo |
| ---                    | ---         | ---               | ---            |
| CPU: Ryzen 6850U       | 00:26:12    | 00:13:30          | 00:18:30       |
| CPU: Apple M1          | 00:33:15    | 00:21:40          | 00:??:??       |
| CPU: Intel i9-10940X   | 00:10:25    | 00:04:36          | 00:??:??       |
| CPU: Intel i7-8750H    | 00:??:??    | 00:??:??          | 00:19:16       |
| GPU: RTX 2080 Ti       | 00:01:44    | 00:01:06          | 00:??:??       |
| GPU: RTX 2070 Max-Q    | 00:05:59    | 00:??:??          | 00:04:37       |


## Roadmap and Upcoming Features

Planned in the near future.
- Batch Processing, allowing to have files queued for transcription
- Add options for more verbatim output
- Make adding custom models more easy
- MacOS installers
- Somehow getting that flatpak package to work **Published 1.4.1 on Flathub.
- Customization of output naming
- Allowing users to setting the output directory
- Allow for saving settings and defaults (currently resets after each transcription)  **Implemented in v1.4.0

## For contributors

See [CONTRIBUTING.md](CONTRIBUTING.md) for local development setup. aTrain
uses [uv](https://docs.astral.sh/uv/) as its recommended package manager.

## Attribution
The GIFs and Icons in aTrain are from [tenor](https://tenor.com/) and [flaticon](https://www.flaticon.com/).
