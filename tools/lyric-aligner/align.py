"""Align a plain-text lyric file to an audio file with faster-whisper."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess

from faster_whisper import WhisperModel

try:
	from exporter import write_alignment
	from matcher import align_lyrics, reconstruct_missing_lines
except ImportError:
	from .exporter import write_alignment
	from .matcher import TranscriptWord, align_lyrics


def audio_duration(path: Path) -> float:
	result = subprocess.run(
		["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
		check=True,
		capture_output=True,
		text=True,
	)
	return float(result.stdout.strip())


def collect_line_objects(sections: list) -> list[dict]:
	line_objects: list[dict] = []
	for section_index, section in enumerate(sections):
		if not isinstance(section, dict):
			raise ValueError(f"sections[{section_index}] must be an object")
		lines = section.get("lines", [])
		if not isinstance(lines, list):
			raise ValueError(f"sections[{section_index}].lines must be an array")
		for line_index, line in enumerate(lines):
			if not isinstance(line, dict) or not isinstance(line.get("text"), str):
				raise ValueError(f"sections[{section_index}].lines[{line_index}] must contain text")
			line_objects.append(line)
		line_objects.extend(collect_line_objects(section.get("sections", [])))
	return line_objects


def load_lyrics_document(path: Path) -> tuple[dict, list[dict]]:
	try:
		document = json.loads(path.read_text(encoding="utf-8-sig"))
	except (OSError, json.JSONDecodeError) as error:
		raise ValueError(f"Invalid lyrics JSON: {error}") from error
	if not isinstance(document, dict):
		raise ValueError("Lyrics JSON must contain an object at the top level")
	sections = document.get("sections")
	if not isinstance(sections, list):
		raise ValueError('Lyrics JSON must contain a "sections" array')

	return document, collect_line_objects(sections)


def main() -> None:
    import sys

    # =========================================================
    # FORCE UTF-8 OUTPUT ON WINDOWS
    # =========================================================

    try:
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
        )

        sys.stderr.reconfigure(
            encoding="utf-8",
            errors="replace",
        )
    except AttributeError:
        pass

    # =========================================================
    # ARGUMENT PARSER
    # =========================================================

    parser = argparse.ArgumentParser(
        description="Align a lyrics JSON document to an audio recording."
    )

    parser.add_argument(
        "--audio",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--lyrics",
        required=True,
        type=Path,
        help="UTF-8 lyrics JSON with sections[].lines[].text",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--model",
        default="small",
        help="faster-whisper model name, e.g. small or medium",
    )

    args = parser.parse_args()

    # =========================================================
    # VALIDATE INPUT FILES
    # =========================================================

    if not args.audio.is_file():
        parser.error(
            f"Audio file not found: {args.audio}"
        )

    if not args.lyrics.is_file():
        parser.error(
            f"Lyrics file not found: {args.lyrics}"
        )

    # =========================================================
    # LOAD LYRICS
    # =========================================================

    try:
        document, line_objects = load_lyrics_document(
            args.lyrics
        )
    except ValueError as error:
        parser.error(str(error))

    # =========================================================
    # AUDIO DURATION
    # =========================================================

    duration = audio_duration(
        args.audio
    )

    print(
        f"\nAudio duration: {duration:.2f}s\n"
    )

    # =========================================================
    # LOAD WHISPER MODEL
    # =========================================================

    print(
        f"Loading Whisper model: {args.model}\n"
    )

    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
    )

    # =========================================================
    # TRANSCRIBE AUDIO
    # =========================================================

    print(
        "Transcribing audio...\n"
    )

    segments, info = model.transcribe(
        str(args.audio),
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
        beam_size=5,
    )

    # faster-whisper returns a generator.
    # Convert it to a list so the complete transcript
    # is materialized before extracting words.
    segments = list(segments)

    # =========================================================
    # WHISPER INFO
    # =========================================================

    print(
        "\n========== WHISPER INFO ==========\n"
    )

    print(
        f"Audio duration reported by Whisper: "
        f"{info.duration:.2f}s"
    )

    print(
        f"Language: "
        f"{info.language}"
    )

    print(
        f"Language probability: "
        f"{info.language_probability:.4f}"
    )

    print(
        f"Number of segments: "
        f"{len(segments)}"
    )

    if segments:
        print(
            f"First segment: "
            f"{segments[0].start:.2f} -> "
            f"{segments[0].end:.2f}"
        )

        print(
            f"Last segment: "
            f"{segments[-1].start:.2f} -> "
            f"{segments[-1].end:.2f}"
        )
    else:
        print(
            "No Whisper segments were returned."
        )

    print(
        "\n========== END WHISPER INFO ==========\n"
    )

    # =========================================================
    # SEGMENT SUMMARY
    # =========================================================

    print(
        "\n========== SEGMENT SUMMARY ==========\n"
    )

    for segment in segments:
        segment_text = (
            segment.text.strip()
        )

        print(
            f"{segment.start:8.2f} -> "
            f"{segment.end:8.2f} | "
            f"{segment_text}"
        )

    print(
        "\n========== END SEGMENT SUMMARY ==========\n"
    )

    # =========================================================
    # EXTRACT WHISPER WORD TIMESTAMPS
    # =========================================================

    words = []

    for segment in segments:
        for word in (segment.words or []):
            word_text = word.word.strip()

            if not word_text:
                continue

            words.append(
                TranscriptWord(
                    word_text,
                    word.start,
                    word.end,
                )
            )

    # =========================================================
    # WHISPER WORD DEBUG
    # =========================================================

    print(
        "\n========== WHISPER WORDS ==========\n"
    )

    for word in words:
        print(
            f"{word.start:8.2f} -> "
            f"{word.end:8.2f} | "
            f"{word.text}"
        )

    print(
        "\n========== END WHISPER WORDS ==========\n"
    )

    print(
        f"\nTotal Whisper words: {len(words)}"
    )

    if words:
        print(
            f"First word timestamp: "
            f"{words[0].start:.2f}s"
        )

        print(
            f"Last word timestamp: "
            f"{words[-1].end:.2f}s"
        )

    # =========================================================
    # ALIGN LYRICS
    # =========================================================

    lyric_lines = [
        line["text"]
        for line in line_objects
    ]

    aligned_lines = align_lyrics(
        lyric_lines,
        words,
        duration,
    )

    # =========================================================
    # BUILD OUTPUT DOCUMENT
    # =========================================================

    output_document = copy.deepcopy(
        document
    )

    output_lines = collect_line_objects(
        output_document["sections"]
    )

    for output_line, timing in zip(
        output_lines,
        aligned_lines,
    ):
        output_line["start"] = (
            timing["start"]
        )

        output_line["end"] = (
            timing["end"]
        )

        output_line["duration"] = (
            timing["duration"]
        )

        output_line["confidence"] = (
            timing["confidence"]
        )

    # =========================================================
    # WRITE OUTPUT
    # =========================================================

    write_alignment(
        args.output,
        output_document,
        duration,
    )

    print(
        f"\nWrote "
        f"{len(output_lines)} "
        f"synchronized lines "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()

def reconstruct_missing_lines(
    results: list[dict],
    audio_duration: float,
) -> list[dict]:
    """
    Reconstruct timestamps for consecutive lyric lines
    that Whisper could not confidently match.

    Only lines with confidence == 0 are reconstructed.

    The function works in blocks:
        known line
        missing
        missing
        missing
        known line

    The time gap between the two known anchors is divided
    across the missing lines.

    Existing Whisper matches are never modified.
    """

    if not results:
        return results

    total = len(results)

    # ---------------------------------------------------------
    # Mark existing alignment method
    # ---------------------------------------------------------

    for item in results:
        confidence = float(
            item.get("confidence", 0.0)
        )

        if confidence > 0:
            item["alignment_method"] = "whisper"
        else:
            item["alignment_method"] = "pending"

    # ---------------------------------------------------------
    # Find consecutive missing blocks
    # ---------------------------------------------------------

    index = 0

    while index < total:

        current = results[index]

        confidence = float(
            current.get("confidence", 0.0)
        )

        # Already matched.
        if confidence > 0:
            index += 1
            continue

        # -----------------------------------------------------
        # Start of missing block
        # -----------------------------------------------------

        block_start = index

        while index < total:
            confidence = float(
                results[index].get(
                    "confidence",
                    0.0,
                )
            )

            if confidence > 0:
                break

            index += 1

        block_end = index - 1

        # -----------------------------------------------------
        # Find previous known anchor
        # -----------------------------------------------------

        previous = None

        for position in range(
            block_start - 1,
            -1,
            -1,
        ):
            candidate = results[position]

            if candidate.get("start") is not None:
                previous = candidate
                break

        # -----------------------------------------------------
        # Find following known anchor
        # -----------------------------------------------------

        following = None

        for position in range(
            block_end + 1,
            total,
        ):
            candidate = results[position]

            if candidate.get("start") is not None:
                following = candidate
                break

        # -----------------------------------------------------
        # Determine reconstruction boundaries
        # -----------------------------------------------------

        if previous is not None:
            lower = float(
                previous["end"]
            )
        else:
            lower = 0.0

        if following is not None:
            upper = float(
                following["start"]
            )
        else:
            upper = float(
                audio_duration
            )

        # Safety.
        lower = max(
            0.0,
            lower,
        )

        upper = min(
            float(audio_duration),
            max(
                lower,
                upper,
            ),
        )

        missing_count = (
            block_end
            - block_start
            + 1
        )

        gap = upper - lower

        # -----------------------------------------------------
        # No usable gap
        # -----------------------------------------------------

        if gap <= 0:
            for position in range(
                block_start,
                block_end + 1,
            ):
                item = results[position]

                item["start"] = round(
                    lower,
                    3,
                )

                item["end"] = round(
                    lower,
                    3,
                )

                item["duration"] = 0.0

                item["confidence"] = 0.0

                item["alignment_method"] = (
                    "reconstructed_zero_gap"
                )

            continue

        # -----------------------------------------------------
        # Divide the gap across missing lines
        # -----------------------------------------------------

        interval = (
            gap / missing_count
        )

        for offset, position in enumerate(
            range(
                block_start,
                block_end + 1,
            )
        ):

            item = results[position]

            start = (
                lower
                + interval * offset
            )

            end = (
                lower
                + interval * (offset + 1)
            )

            item["start"] = round(
                start,
                3,
            )

            item["end"] = round(
                end,
                3,
            )

            item["duration"] = round(
                max(
                    0.0,
                    end - start,
                ),
                3,
            )

            item["confidence"] = 0.0

            item["alignment_method"] = (
                "reconstructed"
            )

    # ---------------------------------------------------------
    # Final monotonic cleanup
    # ---------------------------------------------------------

    previous_end = 0.0

    for item in results:

        start = float(
            item.get("start", 0.0)
        )

        end = float(
            item.get("end", start)
        )

        start = max(
            previous_end,
            start,
        )

        end = max(
            start,
            end,
        )

        end = min(
            float(audio_duration),
            end,
        )

        item["start"] = round(
            start,
            3,
        )

        item["end"] = round(
            end,
            3,
        )

        item["duration"] = round(
            end - start,
            3,
        )

        previous_end = end

    return results    