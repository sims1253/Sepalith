"""Two-stage R detector for Modotte/CodeX-7M-Non-Thinking.

Stage 1 (cheap regex prefilter on input+output):
  - an exact R-ish fence info string (```r / ```R / rlang / rscript / splus), OR
  - any strong R token anywhere, OR
  - at least one '<-' occurrence.

Stage 2 (confirmation): extract fenced code blocks, classify each block's
language (fence tag if known, else token scoring), compare R mass vs
other-language mass, and require a positive R token score.
"""
import re

# ---------------------------------------------------------------- stage 1

R_FENCE_TAGS = {"r", "rlang", "rscript", "rsplus", "splus", "rcode", "rmarkdown", "rd"}
R_FENCE_RE = re.compile(r"^```+[ \t]*([A-Za-z0-9+#._-]{0,20})[ \t]*$", re.M)

_STAGE1_TOKEN_RE = re.compile(
    r"\blibrary\(|\brequire\(|%>%|%in%|\bdata\.frame\(|\bstopifnot\(|\bread\.csv\("
    r"|\bwrite\.csv\(|\bsaveRDS\(|\breadRDS\(|\bRcpp::|<-"
)


def stage1(text: str) -> bool:
    for tag in R_FENCE_RE.findall(text):
        if tag.lower() in R_FENCE_TAGS:
            return True
    return bool(_STAGE1_TOKEN_RE.search(text))


# ---------------------------------------------------------------- stage 2

_FENCE_BLOCK_RE = re.compile(r"^```+[ \t]*([A-Za-z0-9+#._-]{0,20})[ \t]*\n(.*?)^```+[ \t]*$", re.M | re.S)

_OTHER_FENCE = {
    "python", "py", "java", "javascript", "js", "ts", "typescript", "cpp", "c++", "c",
    "csharp", "cs", "go", "golang", "rust", "ruby", "php", "swift", "kotlin", "scala",
    "haskell", "clojure", "elixir", "erlang", "perl", "lua", "dart", "bash", "sh", "shell",
    "zsh", "powershell", "ps1", "sql", "html", "xml", "css", "json", "yaml", "yml",
    "fortran", "verilog", "vhdl", "assembly", "asm", "matlab", "octave", "julia", "raku",
    "perl6", "objective", "objectivec", "makefile", "cmake", "dockerfile", "text", "plaintext",
}

_R_MARKERS = [
    (re.compile(r"\blibrary\("), 3, 2),
    (re.compile(r"\brequire\("), 3, 1),
    (re.compile(r"%>%"), 2, 3),
    (re.compile(r"%in%"), 2, 2),
    (re.compile(r"\|\>"), 1, 2),
    (re.compile(r"\bdata\.frame\("), 3, 2),
    (re.compile(r"\bstopifnot\("), 3, 1),
    (re.compile(r"\bread\.csv\(|\bread_csv\(|\bwrite\.csv\(|\bwrite_csv\("), 3, 1),
    (re.compile(r"\bsaveRDS\(|\breadRDS\(|\bread\.rds\("), 3, 1),
    (re.compile(r"\bggplot\(|\bggplot2::|\bdplyr::|\btidyverse|\btidyr::|\bstringr::|\bpurrr::|\bforcats::|\btibble\("), 3, 2),
    (re.compile(r"<-"), 1, 8),
    (re.compile(r"^#'[ \t]", re.M), 2, 2),
    (re.compile(r"\bfunction\s*\("), 1, 2),
    (re.compile(r"\bmtcars\b|\biris\b|\blm\(|\baov\(|\bglm\("), 1, 2),
    (re.compile(r"\bcat\(|\bhead\(|\bstr\(|\bsummary\("), 0.5, 2),
]

_OTHER_MARKERS = [
    (re.compile(r"\bdef\s+\w+\s*\("), 3, 2),          # python/ruby(def..end)
    (re.compile(r"\bimport\s+(numpy|pandas|torch|tensorflow|matplotlib|scipy|sklearn|os|sys|re|json|random)\b"), 4, 2),
    (re.compile(r"\bself\b|\belif\b|\bf\"|f'"), 2, 2),
    (re.compile(r"=>|\bconsole\.log\(|\blet\s+\w+\s*=|\bconst\s+\w+\s*=|\bvar\s+\w+\s*="), 2, 2),
    (re.compile(r"#include|\bstd::|\bint\s+main\(|\bprintf\("), 4, 2),
    (re.compile(r"\bpublic\s+class\b|\bSystem\.out|\bpublic\s+static\b"), 4, 2),
    (re.compile(r"\busing\s+\w+|\bprintln\(|\bstruct\s+\w+"), 2, 2),   # julia
    (re.compile(r"\bfn\s+\w+\s*\(|\blet\s+mut\b|\bimpl\b"), 4, 1),    # rust
    (re.compile(r"\bfunc\s+\w+\s*\("), 4, 1),                          # go
    (re.compile(r":="), 2, 2),                                         # go/pseudocode
    (re.compile(r"\bputs\s+|\bend\s*$", re.M), 1.5, 4),                # ruby/julia/matlab end
    (re.compile(r"\bdisp\(|\bfprintf\(|\bend\s*;"), 2, 2),             # matlab
]


def _token_score(code: str):
    r = 0.0
    for rx, w, cap in _R_MARKERS:
        n = len(rx.findall(code))
        if n:
            r += w * min(n, cap)
    o = 0.0
    for rx, w, cap in _OTHER_MARKERS:
        n = len(rx.findall(code))
        if n:
            o += w * min(n, cap)
    return r, o


def stage2(inp: str, out: str):
    """Return (is_r: bool, info: dict with scores for tuning)."""
    text = (inp or "") + "\n" + (out or "")
    r_fenced_mass = 0
    other_fenced_mass = 0
    other_fenced_langs = set()
    unfenced = [text]
    blocks = _FENCE_BLOCK_RE.findall(out or "")
    if blocks:
        unfenced = []
        for tag, body in blocks:
            t = tag.lower()
            if t in R_FENCE_TAGS:
                r_fenced_mass += len(body)
            elif t in _OTHER_FENCE:
                other_fenced_mass += len(body)
                other_fenced_langs.add(t)
            else:
                # unknown tag: treat as unfenced code, token-score it below
                unfenced.append(body)
        # prose between fences still token-scores via unfenced text
    r_score_out, o_score_out = _token_score(out or "")
    r_score_in, o_score_in = _token_score(inp or "")
    r_score = r_score_out + 0.3 * r_score_in
    o_score = o_score_out + 0.3 * o_score_in

    has_r_fence = r_fenced_mass > 0
    fenced_total = r_fenced_mass + other_fenced_mass
    r_frac = (r_fenced_mass / fenced_total) if fenced_total else None

    accept = False
    reason = ""
    if has_r_fence and (r_frac is None or r_frac >= 0.35) and r_score >= 4 and r_score >= o_score:
        accept = True
        reason = "r_fence_dominant"
    elif r_score >= 6 and r_score >= 1.5 * max(o_score, 1) and not (fenced_total and r_frac == 0 and fenced_total > 800):
        # token path: reject if output is fenced with clearly non-R code and no R block
        accept = True
        reason = "token_dominant"

    return accept, {
        "accept": accept, "reason": reason, "r_score": round(r_score, 1), "o_score": round(o_score, 1),
        "has_r_fence": has_r_fence, "r_frac": round(r_frac, 2) if r_frac is not None else None,
        "other_fenced_langs": sorted(other_fenced_langs),
    }
