"""Normalize known spellings of families already admitted by the frozen policy.

This module does not authorize new license families or override package-level
restrictions. Preserve the original license text in provenance records.
"""
import re

ALIASES = {
    "BSD_2_clause": "BSD-2-clause",
    "BSD_3_clause": "BSD-3-clause",
    "GNU General Public License": "GPL",
    "Mozilla Public License 2.0": "MPL-2.0",
    "Mozilla Public License Version 2.0": "MPL-2.0",
}

def normalize_recognized_alias(value):
    for original, canonical in sorted(ALIASES.items(), key=lambda item: -len(item[0])):
        pattern = r"(?<![\w-])" + re.escape(original) + r"(?=$|[ +,(])"
        value = re.sub(pattern, canonical, value)
    return value
