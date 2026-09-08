"""Pure extension-faithful parser, extracted unchanged from eval_noop_fp."""
import re

MARKER_LINE = re.compile(
    r"^\s*(<<<<<<<\s*CURRENT|=======|>>>>>>>\s*UPDATED|<\[fim-(middle|prefix|"
    r"suffix)\]>|<\|user_cursor\|>|<\|outline\|>)\s*$")

def parse_prediction(text: str) -> list[str]:
    """extension.parsePrediction, line-for-line."""
    if ">>>>>>>" in text:
        text = text.split(">>>>>>>")[0]
    text = text.replace("<|user_cursor|>", "")
    lines = [l.rstrip("\r") for l in text.split("\n")]
    lines = [l for l in lines if not MARKER_LINE.match(l)]
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    for i in range(2, len(lines)):
        if lines[i] == lines[i - 1] == lines[i - 2]:
            lines = lines[:i]
            break
    return lines
