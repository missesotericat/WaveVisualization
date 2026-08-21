"""Match supplied lyric lines to timestamped Whisper words."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import unicodedata


@dataclass(frozen=True)
class TranscriptWord:
	text: str
	start: float
	end: float


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy comparison without changing exported lyrics."""
    text = unicodedata.normalize("NFKC", text).casefold()

    # Python's \w already supports Unicode characters, including Vietnamese.
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)

    return " ".join(tokens)


def _score(lyric: str, words: list[TranscriptWord]) -> float:
    """
    Score lyric against a candidate Whisper word window.

    Combines:
    - character similarity
    - fuzzy token similarity
    - token coverage
    - exact token overlap
    """

    if not lyric or not words:
        return 0.0

    lyric_norm = normalize_text(lyric)
    candidate_norm = normalize_text(
        " ".join(word.text for word in words)
    )

    if not lyric_norm or not candidate_norm:
        return 0.0

    lyric_tokens = lyric_norm.split()
    candidate_tokens = candidate_norm.split()

    # Character-level similarity
    char_score = SequenceMatcher(
        None,
        lyric_norm,
        candidate_norm,
    ).ratio()

    # Best fuzzy match for every lyric token
    token_scores: list[float] = []

    for lyric_token in lyric_tokens:
        best = max(
            (
                SequenceMatcher(
                    None,
                    lyric_token,
                    candidate_token,
                ).ratio()
                for candidate_token in candidate_tokens
            ),
            default=0.0,
        )

        token_scores.append(best)

    token_score = (
        sum(token_scores) / len(token_scores)
        if token_scores
        else 0.0
    )

    # Token coverage
    matched_tokens = sum(
        1
        for score in token_scores
        if score >= 0.60
    )

    coverage = matched_tokens / max(
        1,
        len(lyric_tokens),
    )

    # Exact token overlap
    lyric_set = set(lyric_tokens)
    candidate_set = set(candidate_tokens)

    overlap = len(
        lyric_set & candidate_set
    ) / max(
        1,
        len(lyric_set),
    )

    # Weighted score
    score = (
        char_score * 0.30
        + token_score * 0.35
        + overlap * 0.15
        + coverage * 0.20
    )

    return score


def align_lyrics(
    lyric_lines: list[str],
    transcript_words: list[TranscriptWord],
    audio_duration: float,
    minimum_score: float = 0.42,
) -> list[dict]:
    """
    Timeline-aware lyric alignment.

    Core idea:

    - Strong Whisper/lyric matches become anchors.
    - Weak matches are not allowed to jump arbitrarily far.
    - Matching is monotonic.
    - Large gaps between anchors are treated as missing-transcript
      regions and lyrics are distributed inside those gaps.
    - A lyric line cannot consume an unreasonable amount of time.
    """

    lines = [
        line.rstrip("\r")
        for line in lyric_lines
    ]

    results = [
        {
            "index": index,
            "text": text,
            "start": None,
            "end": None,
            "confidence": 0.0,
        }
        for index, text in enumerate(lines)
    ]

    if not lines:
        return results

    if not transcript_words:
        interval = (
            audio_duration
            / max(1, len(results))
        )

        for index, item in enumerate(results):
            item["start"] = index * interval
            item["end"] = (index + 1) * interval
            item["confidence"] = 0.0
            item["duration"] = round(
                interval,
                3,
            )

        return results

    # =========================================================
    # CONFIG
    # =========================================================

    MAX_SEARCH_WORDS = 60

    MAX_WINDOW_WORDS = 16

    MAX_MATCH_DURATION = 7.0

    STRONG_SCORE = 0.68

    WEAK_SCORE = minimum_score

    # Maximum amount of real time a fuzzy matcher is allowed
    # to jump forward while searching for the next lyric.
    MAX_TIME_JUMP = 14.0

    # =========================================================
    # NORMALIZE LYRICS
    # =========================================================

    normalized_lines = [
        normalize_text(text)
        for text in lines
    ]

    # =========================================================
    # FIND ANCHORS
    # =========================================================

    cursor = 0

    anchors: list[dict] = []

    for lyric_index, lyric in enumerate(
        normalized_lines
    ):

        if not lyric:
            continue

        lyric_tokens = lyric.split()

        lyric_word_count = max(
            1,
            len(lyric_tokens),
        )

        if cursor >= len(
            transcript_words
        ):
            break

        search_end = min(
            len(transcript_words),
            cursor + MAX_SEARCH_WORDS,
        )

        best_strong = None
        best_weak = None

        for start_index in range(
            cursor,
            search_end,
        ):

            candidate_start_time = (
                transcript_words[
                    start_index
                ].start
            )

            # -------------------------------------------------
            # IMPORTANT:
            # Don't allow a fuzzy match to jump too far.
            # -------------------------------------------------

            cursor_time = (
                transcript_words[
                    cursor
                ].start
            )

            if (
                candidate_start_time
                - cursor_time
                > MAX_TIME_JUMP
            ):
                break

            min_window = max(
                1,
                lyric_word_count - 2,
            )

            max_window = min(
                MAX_WINDOW_WORDS,
                max(
                    min_window,
                    lyric_word_count * 2 + 3,
                ),
            )

            for window_size in range(
                min_window,
                max_window + 1,
            ):

                end_index = (
                    start_index
                    + window_size
                )

                if end_index > len(
                    transcript_words
                ):
                    break

                candidate_words = (
                    transcript_words[
                        start_index:end_index
                    ]
                )

                if not candidate_words:
                    continue

                candidate_start = (
                    candidate_words[0].start
                )

                candidate_end = (
                    candidate_words[-1].end
                )

                candidate_duration = (
                    candidate_end
                    - candidate_start
                )

                if (
                    candidate_duration
                    > MAX_MATCH_DURATION
                ):
                    continue

                score = _score(
                    lyric,
                    candidate_words,
                )

                # -------------------------------------------------
                # Prefer candidate windows whose word count
                # resembles the lyric.
                # -------------------------------------------------

                word_distance = abs(
                    len(candidate_words)
                    - lyric_word_count
                )

                length_penalty = (
                    word_distance
                    / max(
                        1,
                        lyric_word_count,
                    )
                )

                adjusted_score = (
                    score
                    - min(
                        0.12,
                        length_penalty
                        * 0.08,
                    )
                )

                candidate = (
                    adjusted_score,
                    score,
                    start_index,
                    end_index,
                )

                if (
                    best_weak is None
                    or candidate[0]
                    > best_weak[0]
                ):
                    best_weak = candidate

                if score >= STRONG_SCORE:

                    if (
                        best_strong is None
                        or candidate[0]
                        > best_strong[0]
                    ):
                        best_strong = candidate

        # =====================================================
        # SELECT MATCH
        # =====================================================

        selected = None

        # Strong match always wins.
        if best_strong is not None:

            selected = best_strong

        # Weak match only allowed when reasonably close.
        elif best_weak is not None:

            weak_score = best_weak[1]

            if weak_score >= WEAK_SCORE:
                selected = best_weak

        if selected is None:
            continue

        adjusted_score, raw_score, match_start, match_end = (
            selected
        )

        start = transcript_words[
            match_start
        ].start

        end = transcript_words[
            match_end - 1
        ].end

        duration = end - start

        if duration <= 0:
            continue

        if duration > MAX_MATCH_DURATION:
            continue

        # =====================================================
        # STORE ANCHOR
        # =====================================================

        anchors.append(
            {
                "lyric_index": lyric_index,
                "start": start,
                "end": end,
                "confidence": round(
                    raw_score,
                    3,
                ),
                "match_start": match_start,
                "match_end": match_end,
            }
        )

        cursor = max(
            cursor,
            match_end,
        )

    # =========================================================
    # REMOVE DUPLICATE / BACKWARD ANCHORS
    # =========================================================

    clean_anchors: list[dict] = []

    previous_lyric_index = -1
    previous_end = 0.0

    for anchor in anchors:

        if (
            anchor["lyric_index"]
            <= previous_lyric_index
        ):
            continue

        if (
            anchor["start"]
            < previous_end
        ):
            continue

        clean_anchors.append(
            anchor
        )

        previous_lyric_index = (
            anchor["lyric_index"]
        )

        previous_end = anchor["end"]

    anchors = clean_anchors

    # =========================================================
    # APPLY ANCHORS
    # =========================================================

    for anchor in anchors:

        index = anchor[
            "lyric_index"
        ]

        results[index]["start"] = (
            anchor["start"]
        )

        results[index]["end"] = (
            anchor["end"]
        )

        results[index]["confidence"] = (
            anchor["confidence"]
        )

    # =========================================================
    # FILL BEFORE FIRST ANCHOR
    # =========================================================

    if anchors:

        first = anchors[0]

        first_index = first[
            "lyric_index"
        ]

        if first_index > 0:

            upper = first["start"]

            count = first_index

            interval = (
                upper
                / max(
                    1,
                    count,
                )
            )

            for index in range(
                first_index
            ):

                results[index]["start"] = (
                    index * interval
                )

                results[index]["end"] = (
                    (index + 1) * interval
                )

                results[index]["confidence"] = 0.0

    # =========================================================
    # FILL GAPS BETWEEN ANCHORS
    # =========================================================

    for anchor_position in range(
        len(anchors) - 1
    ):

        left = anchors[
            anchor_position
        ]

        right = anchors[
            anchor_position + 1
        ]

        left_index = left[
            "lyric_index"
        ]

        right_index = right[
            "lyric_index"
        ]

        missing_count = (
            right_index
            - left_index
            - 1
        )

        if missing_count <= 0:
            continue

        lower = left["end"]

        upper = right["start"]

        gap = max(
            0.0,
            upper - lower,
        )

        interval = (
            gap
            / (missing_count + 1)
        )

        for offset in range(
            1,
            missing_count + 1,
        ):

            index = (
                left_index
                + offset
            )

            start = (
                lower
                + interval
                * (offset - 1)
            )

            end = (
                lower
                + interval
                * offset
            )

            results[index]["start"] = (
                start
            )

            results[index]["end"] = (
                end
            )

            results[index]["confidence"] = 0.0

    # =========================================================
    # FILL AFTER LAST ANCHOR
    # =========================================================

    if anchors:

        last = anchors[-1]

        last_index = last[
            "lyric_index"
        ]

        if last_index < len(results) - 1:

            lower = last["end"]

            remaining_count = (
                len(results)
                - last_index
                - 1
            )

            remaining_time = max(
                0.0,
                audio_duration
                - lower,
            )

            interval = (
                remaining_time
                / max(
                    1,
                    remaining_count,
                )
            )

            for offset in range(
                1,
                remaining_count + 1,
            ):

                index = (
                    last_index
                    + offset
                )

                start = (
                    lower
                    + interval
                    * (offset - 1)
                )

                end = (
                    lower
                    + interval
                    * offset
                )

                results[index]["start"] = (
                    start
                )

                results[index]["end"] = (
                    end
                )

                results[index]["confidence"] = 0.0

    # =========================================================
    # FALLBACK IF NO ANCHORS
    # =========================================================

    else:

        interval = (
            audio_duration
            / max(
                1,
                len(results),
            )
        )

        for index, item in enumerate(
            results
        ):

            item["start"] = (
                index * interval
            )

            item["end"] = (
                (index + 1) * interval
            )

            item["confidence"] = 0.0

    # =========================================================
    # FINAL CLEANUP
    # =========================================================

    previous_end = 0.0

    for item in results:

        start = float(
            item["start"]
            if item["start"] is not None
            else previous_end
        )

        end = float(
            item["end"]
            if item["end"] is not None
            else start
        )

        start = max(
            previous_end,
            start,
        )

        start = min(
            audio_duration,
            start,
        )

        end = max(
            start,
            end,
        )

        end = min(
            audio_duration,
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

def _is_valid_timestamp(value) -> bool:
    return value is not None


def _find_previous_known(results: list[dict], index: int):
    for position in range(index - 1, -1, -1):
        item = results[position]

        if _is_valid_timestamp(item.get("start")):
            return item

    return None


def _find_next_known(results: list[dict], index: int):
    for position in range(index + 1, len(results)):
        item = results[position]

        if _is_valid_timestamp(item.get("start")):
            return item

    return None


def _get_section_id(line_id: str) -> str | None:
    """
    Extract section prefix from IDs such as:

        c1-01
        c1-02
        v2-03
        pc-01
        intro-01
    """

    if not line_id or "-" not in line_id:
        return None

    return line_id.rsplit("-", 1)[0]


def _build_section_groups(
    line_objects: list[dict],
) -> dict[str, list[int]]:
    """
    Group lyric line indexes by section prefix.

    Example:

        c1-01
        c1-02
        c1-03

    becomes:

        {
            "c1": [10, 11, 12]
        }
    """

    groups: dict[str, list[int]] = {}

    for index, line in enumerate(line_objects):
        line_id = str(line.get("id", ""))
        section_id = _get_section_id(line_id)

        if section_id is None:
            continue

        groups.setdefault(section_id, [])
        groups[section_id].append(index)

    return groups


def _interpolate_missing_local(
    results: list[dict],
    start_index: int,
    end_index: int,
) -> None:
    """
    Interpolate ONLY inside a local missing block.

    Example:

        known
        missing
        missing
        known

    The missing lines are distributed between
    the two actual anchors.

    This function deliberately does not touch
    timestamps outside the requested block.
    """

    if start_index > end_index:
        return

    previous = _find_previous_known(
        results,
        start_index,
    )

    following = _find_next_known(
        results,
        end_index,
    )

    if previous is None:
        lower = 0.0
    else:
        lower = float(previous["end"])

    if following is None:
        upper = lower

        # If there is no following anchor,
        # use the last known end as the lower bound.
        for item in results:
            if _is_valid_timestamp(item.get("end")):
                upper = max(
                    upper,
                    float(item["end"]),
                )
    else:
        upper = float(following["start"])

    count = end_index - start_index + 1

    if upper < lower:
        upper = lower

    interval = (
        (upper - lower) / count
        if count > 0
        else 0.0
    )

    for offset, index in enumerate(
        range(start_index, end_index + 1)
    ):
        item = results[index]

        item["start"] = (
            lower
            + interval * offset
        )

        item["end"] = (
            lower
            + interval * (offset + 1)
        )

        item["confidence"] = 0.0
        item["alignment_method"] = "local_interpolation"


def reconstruct_missing_lines(
    results: list[dict],
) -> list[dict]:
    """
    Reconstruct missing lyric timestamps.

    Strategy:

    1. Preserve all Whisper anchors.
    2. Detect contiguous missing blocks.
    3. Interpolate only inside each local block.
    4. Never perform a global interpolation across
       unrelated sections.
    """

    if not results:
        return results

    index = 0

    while index < len(results):

        item = results[index]

        if _is_valid_timestamp(item.get("start")):
            index += 1
            continue

        block_start = index

        while (
            index < len(results)
            and not _is_valid_timestamp(
                results[index].get("start")
            )
        ):
            index += 1

        block_end = index - 1

        _interpolate_missing_local(
            results,
            block_start,
            block_end,
        )

    return results    