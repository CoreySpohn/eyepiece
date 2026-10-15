"""Match written words to the tokens a speech model timed.

This module imports only the standard library.
"""


def _norm(text):
    return "".join(ch for ch in text if ch.isalnum()).lower()


def align(words, tokens):
    """Give each written word the span of the tokens that spell it.

    Kokoro splits text into tokens of its own (punctuation apart, some
    words in pieces). A word takes every token up to the one that completes
    its letters, and spans from the first token's start to the last one's
    end. A token with no time borrows its neighbor's. A word with no letters
    or digits (a free-standing dash or ampersand) is not spoken as a token of
    its own: it takes no token and gets zero width at the end of the word
    before it, or at the start of the word after it when it opens the line.

    Args:
        words: The written words, with punctuation.
        tokens: ``(text, start_s, end_s)`` for each Kokoro token, where a
            time may be None.

    Returns:
        A list of ``(start_s, end_s)``, one per word.

    Raises:
        ValueError: If the tokens do not spell the words.
    """
    spans = []
    pos = 0
    for word in words:
        want = _norm(word)
        if not want:
            spans.append(None)
            continue
        have = ""
        first = last = None
        while pos < len(tokens) and have != want:
            text, start, end = tokens[pos]
            pos += 1
            piece = _norm(text)
            if not piece:
                continue
            have += piece
            if not want.startswith(have):
                raise ValueError(f"tokens do not spell {word!r} (reached {have!r})")
            first = (start, end) if first is None else first
            last = (start, end)
        if have != want:
            raise ValueError(f"ran out of tokens at {word!r}")
        spans.append((first[0], last[1]))
    filled = []
    previous_end = None
    for span in spans:
        if span is None:
            filled.append(None)
            continue
        start, end = span
        start = (
            (0.0 if previous_end is None else previous_end) if start is None else start
        )
        end = start if end is None else end
        filled.append((start, end))
        previous_end = end
    anchor = next((s for s in filled if s is not None), (0.0, 0.0))[0]
    out = []
    for span in filled:
        if span is None:
            out.append((anchor, anchor))
        else:
            out.append(span)
            anchor = span[1]
    return out
