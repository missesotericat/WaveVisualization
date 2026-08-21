# WaveVisualization
multiple tracks, audio management, sprite/audio control, audio-reactive visual, lyric synchronization

## Lyric aligner

The local lyric aligner uses `faster-whisper` on the CPU and matches its
timestamped words against supplied lyrics. It requires Python 3.11 and FFmpeg.

### Windows setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\tools\lyric-aligner\requirements.txt
```

On first use, faster-whisper downloads the selected model from Hugging Face.
The default `small` model works on CPU; use `--model medium` for better
accuracy if the machine has enough memory.

Use an existing UTF-8 lyrics JSON document. Lines are read from
`sections[].lines[]`; section headings, styles, metadata, and all other fields
are preserved in the output. Each line receives `start`, `end`, `duration`, and
`confidence`, while the top-level `duration` and `timingStatus` are updated.
For example, `lyrics/zombie-society.json` can be synchronized with:

```powershell
python .\tools\lyric-aligner\align.py `
	--audio .\audio\zombie-society.mp3 `
	--lyrics .\lyrics\zombie-society.json `
	--output .\output\zombie-society.json
```

The output directory is created automatically. No TXT conversion or external
API is used; transcription and matching run locally on the CPU.
