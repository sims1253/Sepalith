#!/usr/bin/env python3
"""eval_noop_fp.py — the NO-OP / FALSE-SUGGESTION benchmark (product annoyance metric).

Measures how often the SERVED model proposes an edit when NONE is
warranted: the extension's ghost-text popping up while the user merely
moved the cursor or paused in complete code. Joins the permanent battery
(scenarios / intent / drafting / cache / this).

CASE CLASSES (constructed; n >= 120; no training contamination):
  a_after_close_brace  complete function, cursor at end of the closing
                       `}` line — correct action: NO proposal.
  a_file_end           cursor at the very end of the file window, all code
                       complete — correct action: NO proposal.
  b_mid_typing         cursor at the end of an inserted INCOMPLETE partial
                       line (identifier/arg/pipe prefixes) — judgment class:
                       likely-continue vs no-op is genuinely ambiguous, so
                       the proposal rate is INFORMATIONAL, not scored as FP.
  c_stmt_line_end      cursor-moved-only: cursor at the end of a COMPLETE
                       interior statement line — NO proposal.
  c_mid_identifier     cursor-moved-only: cursor in the MIDDLE of a word on
                       a complete line — NO proposal.
  c_blank_in_fn        cursor-moved-only: cursor on a blank line inside a
                       complete function — NO proposal.
  d_tempt_close_brace  complete function CONTAINING a plausible-but-
                       unrequested refactor (accumulation for-loop /
                       mean-no-na.rm / 1:length), cursor at the closing
                       brace — NO proposal (the temptation cases).
  d_tempt_stmt_end     same, cursor at the END of the refactorable
                       statement itself — NO proposal.

CONTAMINATION CONTROL: (1) AUTHORED functions — written for this script,
never in any mixture (guaranteed fresh); (2) corpus functions drawn ONLY
from the finish_block family's own EVAL package split (re-derived here
with the exact assemble_sft_v5.load_finish_block procedure: seed-11
shuffle over the family file's packages, first 5% -> eval), so their
finish_block rows were held out of every sft train split; authored+eval
only, no train-side package is sampled.

PROMPT + ACCEPTANCE SEMANTICS: byte-faithful ports of the PRODUCT
extension (extensions/vscode-sepalith/src):
  - build_prompt()     <- context_build.buildScopedPrompt(scope=null),
                         the byte-identical v0.0.6 render (<[fim-suffix]>
                         head, suffix = rest-of-cursor-line + file below,
                         6000-char budget, prefix truncated from its START)
  - parse_prediction() <- extension.parsePrediction (cut at ">>>>>>>",
                         marker-line drop, blank strip, degenerate-
                         repetition cut at the 3rd identical line)
  - STOPS / max_tokens 320 / temperature 0 <- extension.postCompletion
A PROPOSAL = parse_prediction(completion) non-empty — exactly what the
extension turns into an InlineCompletionItem (ghost text the user sees).

SERVING: --external uses an ALREADY-ANSWERING server (readiness POST,
never spawn, never kill); otherwise spawns a tracked llama-server child
(CPU by default: -ngl 0) and tears down ONLY that PID. Teardown never
touches servers we did not start (tracked-PID-only policy).

Usage:
  python3 experiments/eval/eval_noop_fp.py --external --port 18103 \
      --model /home/m0hawk/Documents/Sepalith/experiments/models/sft_v7_minicpm5-Q8_0.gguf
  python3 experiments/eval/eval_noop_fp.py --spawn-cpu --port 18095 \
      --model .../sft_v6_minicpm5-Q8_0.gguf --parallel 4 --workers 4

Writes results_noop_fp_<model>.jsonl next to this script (resume by id)
and prints the per-class aggregate + VERDICT LAST.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "experiments" / "synthetic-data"))   # scenarios

import scenarios as S                                               # noqa: E402
from scenarios import node_text                                     # noqa: E402

DEFAULT_SERVER = HERE.parent / "bin" / "llama" / "llama-b10453" / "llama-server"
FINISH_SRC = REPO / "experiments" / "synthetic-data" / "finish_block_sample.jsonl"
NORMALIZED = Path("/mnt/h/sepalith/normalized")

# --- the extension's constants, verbatim -------------------------------
MAX_PREFIX_SUFFIX_CHARS = 6000          # context_build.ts
EXT_STOPS = [">>>>>>> UPDATED", "<<<<<<< CURRENT", "=======", "<[fim-middle]>",
             "<[fim-suffix]>", "<[fim-prefix]>", "<|outline|>"]
EXT_MAX_TOKENS = 320
MARKER_LINE = re.compile(
    r"^\s*(<<<<<<<\s*CURRENT|=======|>>>>>>>\s*UPDATED|<\[fim-(middle|prefix|"
    r"suffix)\]>|<\|user_cursor\|>|<\|outline\|>)\s*$")

# mid-typing partials (class b): typed-prefix conventions the mixture trains
# (expect_ partials, pipe operators, half-named args). Index-seeded draw.
PARTIALS = [
    "  out <- lapp",
    "  m <- mean(x, na.r",
    "  res <- x %>%",
    "  plot(x, co",
    "  if (n > ",
    "  vals <- c(vals, x[[i]]$",
]

# temptation (class d) detectors: plausible-but-UNREQUESTED refactors the
# product could volunteer (the loop_to_apply / na_rm / seq_along shapes)
TEMPT_PATTERNS = (
    re.compile(r"for\s*\(\s*\w+\s+in\s+seq_along\("),
    re.compile(r"for\s*\(\s*\w+\s+in\s+\d+L?\s*:"),          # 1:n loop
    re.compile(r"\b(?:mean|sd|var|median)\s*\((?![^()]*na\.rm)"),
    re.compile(r"\b1\s*:\s*length\s*\("),
    re.compile(r"\b1\s*:\s*nrow\s*\("),
    re.compile(r"\bsapply\s*\("),                            # vapply-able
    re.compile(r"\bif\s*\(\s*length\s*\(\s*\w+\s*\)\s*==\s*0"),
    re.compile(r"repeat\s*\{"),
)


# ---------------------------------------------------------------------------
# the extension render/parse, ported byte-faithfully (documented at top)
# ---------------------------------------------------------------------------

def build_prompt(lines: list[str], cursor_line: int, cursor_char: int,
                 rel_path: str) -> tuple[str, int]:
    """context_build.buildScopedPrompt(scope === null): the byte-identical
    v0.0.6 prompt. Returns (prompt, prefix_lines_truncated)."""
    line = lines[cursor_line] if cursor_line < len(lines) else ""
    before, after = line[:cursor_char], line[cursor_char:]
    region_old = [before + "<|user_cursor|>"]
    suffix = [after] + lines[cursor_line + 1:]
    suffix_chars = sum(len(l) + 1 for l in suffix)
    budget = max(0, MAX_PREFIX_SUFFIX_CHARS - suffix_chars)
    prefix = lines[:cursor_line]
    keep = used = 0
    for i in range(len(prefix) - 1, -1, -1):
        if used + len(prefix[i]) + 1 > budget:
            break
        used += len(prefix[i]) + 1
        keep += 1
    truncated = len(prefix) - keep
    prompt = "\n".join(
        ["<[fim-suffix]>"] + suffix
        + [f"<[fim-prefix]><filename>{rel_path}"]
        + prefix[truncated:]
        + ["<<<<<<< CURRENT"] + region_old
        + ["=======", "<[fim-middle]>"])
    return prompt, truncated


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


# ---------------------------------------------------------------------------
# case model
# ---------------------------------------------------------------------------

class Case:
    __slots__ = ("id", "cls", "kind", "source", "package", "fn", "lines",
                 "cursor_line", "cursor_char", "rel_path", "expectation")

    def __init__(self, cls, kind, source, package, fn, lines, cursor_line,
                 cursor_char, rel_path):
        self.cls, self.kind, self.source = cls, kind, source
        self.package, self.fn = package, fn
        self.lines, self.cursor_line, self.cursor_char = lines, cursor_line, cursor_char
        self.rel_path = rel_path
        self.expectation = "no_proposal" if cls[0] in "acd" else "judgment"
        self.id = hashlib.sha1(
            (f"{kind}\x00{rel_path}\x00{cursor_line}\x00{cursor_char}\x00"
             + "\n".join(lines[-8:])).encode("utf-8", "replace")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# authored functions (zero contamination by construction)
# ---------------------------------------------------------------------------

AUTHORED = [
    # stats / simulation
    b"""#' Rolling window standard deviation
#'
#' @param x numeric vector
#' @param w integer window width
#' @return numeric vector of the same length as x
roll_sd <- function(x, w = 3L) {
  n <- length(x)
  out <- rep(NA_real_, n)
  for (i in seq_len(n)) {
    lo <- max(1L, i - w + 1L)
    out[i] <- sd(x[lo:i])
  }
  out
}
""",
    b"""#' Weighted mean with zero-mass guard
#'
#' @param x numeric vector of values
#' @param w numeric vector of weights
weighted_mean <- function(x, w) {
  stopifnot(length(x) == length(w))
  if (sum(w) == 0) {
    return(NA_real_)
  }
  sum(x * w) / sum(w)
}
""",
    b"""#' Bootstrap confidence interval for a mean
#'
#' @param x numeric sample
#' @param B number of replicates
#' @param probs tail probabilities
boot_ci <- function(x, B = 2000L, probs = c(0.025, 0.975)) {
  n <- length(x)
  boots <- vapply(seq_len(B), function(i) {
    mean(sample(x, n, replace = TRUE))
  }, numeric(1))
  stats::quantile(boots, probs = probs)
}
""",
    # data munging
    b"""#' Split a data frame by a factor and summarise each chunk
#'
#' @param df data frame
#' @param by grouping column name
#' @param value value column name
chunk_summary <- function(df, by, value) {
  splits <- split(df[[value]], df[[by]])
  out <- lapply(names(splits), function(nm) {
    v <- splits[[nm]]
    data.frame(group = nm, mean = mean(v), n = length(v))
  })
  do.call(rbind, out)
}
""",
    b"""#' Recode missings across selected columns
#'
#' @param df data frame
#' @param cols character vector of column names
#' @param from value to replace
#' @param to replacement value
recode_missing <- function(df, cols, from = -99, to = NA) {
  for (col in cols) {
    hits <- df[[col]] == from & !is.na(df[[col]])
    df[[col]][hits] <- to
  }
  df
}
""",
    b"""#' Long-to-wide pivot for a three-key frame
#'
#' @param df data frame with key, item, value
pivot_simple <- function(df) {
  keys <- unique(df$key)
  items <- unique(df$item)
  out <- list()
  for (k in keys) {
    row <- list(key = k)
    for (it in items) {
      hit <- df$key == k & df$item == it
      row[[it]] <- if (any(hit)) df$value[hit][1] else NA
    }
    out[[length(out) + 1L]] <- as.data.frame(row, stringsAsFactors = FALSE)
  }
  do.call(rbind, out)
}
""",
    # strings / IO
    b"""#' Normalise a vector of file paths
#'
#' @param paths character vector
#' @param winslash separator style to normalise away
normalise_paths <- function(paths, winslash = "\\\\") {
  p <- gsub(winslash, "/", paths)
  p <- sub("/+$", "", p)
  tolower(p)
}
""",
    b"""#' Read a header-less numeric table safely
#'
#' @param path file path
#' @param sep field separator
read_matrix_safe <- function(path, sep = ",") {
  lines <- readLines(path)
  lines <- lines[nzchar(trimws(lines))]
  rows <- strsplit(lines, sep, fixed = TRUE)
  m <- do.call(rbind, lapply(rows, as.numeric))
  if (any(is.na(m))) {
    warning("non-numeric cells coerced to NA")
  }
  m
}
""",
    b"""#' Timestamped log line writer
#'
#' @param ... pasted message parts
#' @param con connection
log_line <- function(..., con = stderr()) {
  stamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  paste0(stamp, " ", paste0(..., collapse = " "), "\\n") |>
    cat(file = con)
  invisible(NULL)
}
""",
    # dates
    b"""#' Month-start dates for a vector of years
#'
#' @param years integer vector of years
month_starts <- function(years) {
  starts <- as.Date(sprintf("%d-01-01", years))
  ends <- as.Date(sprintf("%d-03-31", years))
  out <- vector("list", length(starts))
  for (i in seq_along(starts)) {
    out[[i]] <- seq(starts[i], ends[i], by = "month")
  }
  as.Date(unlist(out), origin = "1970-01-01")
}
""",
    b"""#' Age in whole years from birth dates
#'
#' @param dob vector of birth Dates
#' @param ref reference Date
age_years <- function(dob, ref = Sys.Date()) {
  if (any(dob > ref, na.rm = TRUE)) {
    stop("birth dates after reference date")
  }
  anniv <- as.Date(sprintf("%s-%s-%s", format(ref, "%Y"),
                           format(dob, "%m"), format(dob), "%d"))
  anniv <- as.Date(ifelse(is.na(anniv),
                          as.Date(sprintf("%s-02-28", format(ref, "%Y"))),
                          anniv), origin = "1970-01-01")
  as.integer(ref - anniv) %/% 365L
}
""",
    # text / factors
    b"""#' Tidy a free-text column into a factor
#'
#' @param x character vector
#' @param keep_top how many levels to keep before lumping
tidy_factor <- function(x, keep_top = 8L) {
  x <- trimws(tolower(x))
  x[!nzchar(x)] <- NA_character_
  tab <- sort(table(x), decreasing = TRUE)
  top <- names(tab)[seq_len(min(keep_top, length(tab)))]
  x[!x %in% top & !is.na(x)] <- "other"
  factor(x)
}
""",
    b"""#' Extract the first regex group as a character vector
#'
#' @param x character input
#' @param pattern regex with one group
first_group <- function(x, pattern) {
  m <- regmatches(x, regexec(pattern, x))
  vapply(m, function(g) if (length(g) >= 2L) g[2] else NA_character_,
         character(1))
}
""",
    # numerics
    b"""#' Row-wise z-score standardisation
#'
#' @param m numeric matrix
row_zscore <- function(m) {
  mu <- rowMeans(m)
  s <- apply(m, 1L, sd)
  safe <- ifelse(s == 0, 1, s)
  sweep(m, 1L, mu) / safe
}
""",
    b"""#' Pretty number formatting with a compact suffix
#'
#' @param x numeric vector
#' @param digits decimals for small values
compact_num <- function(x, digits = 2L) {
  out <- character(length(x))
  for (i in seq_along(x)) {
    v <- abs(x[i])
    if (is.na(v)) {
      out[i] <- "NA"
    } else if (v >= 1e9) {
      out[i] <- paste0(round(x[i] / 1e9, 1), "B")
    } else if (v >= 1e6) {
      out[i] <- paste0(round(x[i] / 1e6, 1), "M")
    } else if (v >= 1e3) {
      out[i] <- paste0(round(x[i] / 1e3, 1), "k")
    } else {
      out[i] <- as.character(round(x[i], digits))
    }
  }
  out
}
""",
    # modelling-ish
    b"""#' Leave-one-out prediction error for a linear fit
#'
#' @param df data frame
#' @param formula model formula
loo_mse <- function(df, formula) {
  n <- nrow(df)
  errs <- numeric(n)
  for (i in seq_len(n)) {
    fit <- lm(formula, data = df[-i, ])
    pred <- predict(fit, newdata = df[i, ])
    errs[i] <- (df[[all.vars(formula)[1]]][i] - pred)^2
  }
  mean(errs)
}
""",
    b"""#' Grid search over a one-parameter objective
#'
#' @param f function of one numeric
#' @param grid numeric candidate vector
grid_argmin <- function(f, grid) {
  vals <- vapply(grid, function(g) f(g), numeric(1))
  grid[which.min(vals)]
}
""",
    # plotting
    b"""#' Base histogram with a density overlay
#'
#' @param x numeric vector
#' @param main plot title
#' @param col histogram fill
hist_density <- function(x, main = "", col = "grey90") {
  dens <- density(x, na.rm = TRUE)
  hx <- hist(x, plot = FALSE)
  graphics::hist(x, col = col, main = main, freq = FALSE,
                 ylim = c(0, max(max(hx$density), max(dens$y)) * 1.05))
  lines(dens, lwd = 2)
  invisible(hx)
}
""",
    b"""#' Save the current device to a dated PNG
#'
#' @param name file stem
#' @param dir output directory
save_dated_png <- function(name, dir = "figures") {
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
  stamp <- format(Sys.Date(), "%Y%m%d")
  path <- file.path(dir, sprintf("%s_%s.png", stamp, name))
  grDevices::png(path, width = 1200, height = 800, res = 150)
  print(path)
  path
}
""",
    b"""#' Simple reference class counter with a cap
#'
#' @param cap maximum value before reset
new_counter <- function(cap = 100L) {
  count <- 0L
  list(
    bump = function() {
      count <<- count + 1L
      if (count > cap) {
        count <<- 0L
      }
      count
    },
    value = function() count
  )
}
""",
    b"""#' Retry an IO call with exponential backoff
#'
#' @param expr the call to retry
#' @param times max attempts
#' @param base initial sleep seconds
retry_io <- function(expr, times = 3L, base = 1) {
  for (attempt in seq_len(times)) {
    ok <- tryCatch({
      assign("ans", force(expr), envir = parent.frame())
      TRUE
    }, error = function(e) FALSE)
    if (ok) {
      return(get("ans", envir = parent.frame()))
    }
    Sys.sleep(base * 2^(attempt - 1))
  }
  stop("all attempts failed")
}
""",
    # temptation carriers (class d): complete, working code that a rewrite-
    # happy assistant would want to touch (loops, sapply, missing na.rm)
    b"""#' Column means of a numeric frame
#'
#' @param df data frame
#' @param cols columns to average
col_means <- function(df, cols) {
  out <- numeric(length(cols))
  for (j in 1:length(cols)) {
    out[j] <- mean(df[[cols[j]]])
  }
  out
}
""",
    b"""#' Widths of every string in a list of lines
#'
#' @param lines list of character vectors
line_widths <- function(lines) {
  sapply(lines, function(l) {
    sum(nchar(l))
  })
}
""",
    b"""#' Scale each matrix column to a given range
#'
#' @param m numeric matrix
#' @param lo target minimum
#' @param hi target maximum
rescale_cols <- function(m, lo = 0, hi = 1) {
  mins <- apply(m, 2, min)
  maxs <- apply(m, 2, max)
  span <- maxs - mins
  if (length(span) == 0) {
    return(m)
  }
  scaled <- sweep(m, 2, mins, "-")
  sweep(scaled, 2, span, "/") * (hi - lo) + lo
}
""",
]


# ---------------------------------------------------------------------------
# corpus source: finish_block EVAL-split packages (contamination control)
# ---------------------------------------------------------------------------

def eval_split_packages() -> list[str]:
    """The exact assemble_sft_v5.load_finish_block package split: seed-11
    shuffle over the family file's packages, first max(1, len//20) = eval
    (5%). These packages' finish_block rows are held out of every sft
    TRAIN split, so corpus cases built from them cannot be memorised
    finish_block targets."""
    pkgs = sorted({json.loads(l)["package"]
                   for l in open(FINISH_SRC)})
    rng = random.Random(11)
    rng.shuffle(pkgs)
    return pkgs[:max(1, len(pkgs) // 20)]


def _lhs_name(b, fn) -> str | None:
    parent = fn.parent
    if parent is None or parent.type != "binary_operator" or not parent.children:
        return None
    return node_text(b.src, parent.children[0]).decode("utf-8", "replace").strip()


ROXY_RE = re.compile(r"^\s*#'(?:\s|$)")


def corpus_functions(want: int, seed: int = 20260820):
    """(rel_path, lines, fn_name, body_r0, body_r1, package) windows from
    HELD-OUT (eval-split) packages: braced bodies of 6..30 non-blank rows,
    named `f <- function(...)`; window = roxygen block above + function +
    2 rows below. Caps: 2 functions per package."""
    from build_astfim import pick_version_dir, src_root_for
    from cases.corpus import _fn_body
    import cases.validators as V
    rng = random.Random(seed)
    eval_pkgs = list(eval_split_packages())
    rng.shuffle(eval_pkgs)
    out: list[dict] = []
    per_pkg: dict[str, int] = {}
    versions: dict[str, Path | None] = {}
    t0 = time.time()
    for pkg in eval_pkgs:
        if len(out) >= want or time.time() - t0 > 600:
            break
        if per_pkg.get(pkg, 0) >= 2:
            continue
        if pkg not in versions:
            versions[pkg] = pick_version_dir(NORMALIZED / pkg)
        vd = versions[pkg]
        if vd is None:
            continue
        rdir = src_root_for(Path(vd), pkg)
        if rdir is None:
            continue
        try:
            files = sorted(list(rdir.glob("*.R")) + list(rdir.glob("*.r")))
        except OSError:
            continue
        for f in files:
            if len(out) >= want or per_pkg.get(pkg, 0) >= 2:
                break
            try:
                src = f.read_bytes()
            except OSError:
                continue
            if not src or len(src) > S.MAX_FILE_BYTES:
                continue
            b = S.Bundle(pkg, f"R/{f.name}", src)
            for fn in (n for n in V._walk(b.tree.root_node)
                       if n.type == "function_definition"):
                geom = _fn_body(b, fn)
                if geom is None:
                    continue
                _body, head_row, r0, r1, nb = geom
                if not 6 <= len(nb) <= 30:
                    continue
                name = _lhs_name(b, fn)
                if not name:
                    continue
                # window: roxygen above (if any) + function + 2 rows below
                top = head_row
                while top - 1 >= 0 and ROXY_RE.match(b.line_str(top - 1)):
                    top -= 1
                bottom = min(b.nlines() - 1, r1 + 2)
                lines = [b.line_str(r) for r in range(top, bottom + 1)]
                out.append(dict(
                    rel_path=f"{pkg}/{b.rel}", lines=lines, fn=name,
                    r0=r0 - top, r1=r1 - top, package=pkg))
                per_pkg[pkg] = per_pkg.get(pkg, 0) + 1
                if len(out) >= want:
                    break
    return out


# ---------------------------------------------------------------------------
# case construction
# ---------------------------------------------------------------------------

def _tempt_lines(lines: list[str], r0: int, r1: int) -> list[int]:
    """Rows that carry a plausible-but-unrequested refactor pattern."""
    return [r for r in range(r0 + 1, r1)
            for pat in TEMPT_PATTERNS if pat.search(lines[r])] or \
           [r for r in range(len(lines))
            for pat in TEMPT_PATTERNS if pat.search(lines[r])]


def _mid_identifier_pos(line: str) -> tuple[int, int] | None:
    """(row_cursor_char) in the middle of the first identifier-like word."""
    m = re.search(r"[A-Za-z.][A-Za-z0-9._]{5,}", line)
    if m is None:
        return None
    start = m.start()
    return start + max(3, (m.end() - start) // 2)


def build_cases(corpus_n: int = 30, seed: int = 7) -> list[Case]:
    rng = random.Random(seed)
    cases: list[Case] = []
    seen_ids: set[str] = set()

    def add(cls, kind, source, fn, lines, cl, cc, rel):
        c = Case(cls, kind, source, source, fn, lines, cl, cc, rel)
        if c.id in seen_ids:
            return
        seen_ids.add(c.id)
        cases.append(c)

    # ---- sources -----------------------------------------------------
    authored = []
    for i, src in enumerate(AUTHORED):
        lines = src.decode("utf-8").split("\n")
        while lines and not lines[-1]:
            lines.pop()
        # locate the function's brace rows
        r0 = next(r for r, l in enumerate(lines) if l.rstrip().endswith("{"))
        depth = 0
        r1 = r0
        for r in range(r0, len(lines)):
            depth += lines[r].count("{") - lines[r].count("}")
            if depth == 0 and r > r0:
                r1 = r
                break
        authored.append(dict(rel_path=f"authored/authored_{i + 1:02d}.R",
                             lines=lines, fn=f"authored_{i + 1:02d}",
                             r0=r0, r1=r1, package="authored"))
    corpus = corpus_functions(corpus_n)
    sources = authored + corpus

    # ---- class a: complete function, cursor at close brace ------------
    for s in sources:
        add("a_after_close_brace", "after_close_brace", s["package"],
            s["fn"], s["lines"], s["r1"],
            len(s["lines"][s["r1"]]), s["rel_path"])
    for s in sources[::2]:                       # every other -> file end
        last = len(s["lines"]) - 1
        add("a_file_end", "file_end", s["package"],
            s["fn"], s["lines"], last, len(s["lines"][last]), s["rel_path"])

    # ---- class b: mid-typing partials (judgment) ----------------------
    for i, s in enumerate(sources):
        partial = PARTIALS[i % len(PARTIALS)]
        body_rows = [r for r in range(s["r0"] + 1, s["r1"])
                     if s["lines"][r].strip()]
        if not body_rows:
            continue
        row = body_rows[min(2, len(body_rows) - 1)]
        lines = s["lines"][:row] + [partial] + s["lines"][row:]
        add("b_mid_typing", "mid_typing_" + partial.strip().replace(" ", "")[:12],
            s["package"], s["fn"], lines, row, len(partial), s["rel_path"])

    # ---- class c: cursor-moved-only -----------------------------------
    for s in sources:
        body = [r for r in range(s["r0"] + 1, s["r1"])
                if s["lines"][r].strip()]
        if not body:
            continue
        row = body[len(body) // 2]
        add("c_stmt_line_end", "stmt_line_end", s["package"],
            s["fn"], s["lines"], row, len(s["lines"][row]), s["rel_path"])
        pos = _mid_identifier_pos(s["lines"][row])
        if pos is not None:
            add("c_mid_identifier", "mid_identifier", s["package"],
                s["fn"], s["lines"], row, pos, s["rel_path"])
        blanks = [r for r in range(s["r0"] + 1, s["r1"])
                  if not s["lines"][r].strip()]
        if blanks:
            add("c_blank_in_fn", "blank_in_fn", s["package"], s["fn"],
                s["lines"], blanks[len(blanks) // 2], 0, s["rel_path"])

    # ---- class d: temptation (plausible but unrequested refactor) -----
    tempted = [s for s in sources if _tempt_lines(s["lines"], s["r0"], s["r1"])]
    for i, s in enumerate(tempted):
        trows = sorted(set(_tempt_lines(s["lines"], s["r0"], s["r1"])))
        add("d_tempt_close_brace", "tempt_close_brace", s["package"],
            s["fn"], s["lines"], s["r1"], len(s["lines"][s["r1"]]),
            s["rel_path"])
        t = trows[i % len(trows)]
        add("d_tempt_stmt_end", "tempt_stmt_end", s["package"], s["fn"],
            s["lines"], t, len(s["lines"][t]), s["rel_path"])

    rng.shuffle(cases)
    return cases


# ---------------------------------------------------------------------------
# serving (tracked-PID-only policy; external mode never spawns/kills)
# ---------------------------------------------------------------------------

def complete(port: int, prompt: str, max_tokens: int, stop: list[str]):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stop": stop,
                       "stream": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return (data["choices"][0]["text"],
            (data.get("usage") or {}).get("completion_tokens", 0),
            time.time() - t0)


def port_answers(port: int, timeout: float = 2.0) -> bool:
    try:
        body = json.dumps({"prompt": "readiness", "max_tokens": 1,
                           "temperature": 0}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            json.JSONDecodeError):
        return False


class Server:
    """Own llama-server child (readiness = a real completion POST), or an
    EXTERNAL already-answering server (never spawn, never kill)."""

    def __init__(self, args):
        self.args = args
        self.proc = None
        self.external = args.external

    def start(self, ready_timeout=1800):
        if self.external:
            if not port_answers(self.args.port, timeout=10):
                raise RuntimeError(
                    f"external server on :{self.args.port} not answering")
            print(f"[serve] external server :{self.args.port} ready "
                  f"(not ours — never killed)", flush=True)
            return
        if port_answers(self.args.port, timeout=1.5):
            raise RuntimeError(
                f"port {self.args.port} already answering; refusing to "
                f"spawn (use --external to use it)")
        cmd = [str(self.args.server_bin), "-m", str(self.args.model),
               "--port", str(self.args.port), "--host", "127.0.0.1",
               "-t", str(self.args.threads), "--parallel",
               str(self.args.parallel), "-c", str(self.args.ctx),
               "-ngl", str(self.args.ngl)]
        log = open(self.args.log, "ab")
        log.write(f"\n==== {time.strftime('%F %T')} {' '.join(cmd)}\n".encode())
        log.flush()
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,
                                     start_new_session=True)
        print(f"[serve] spawned pid {self.proc.pid}: {' '.join(cmd)}",
              flush=True)
        t0 = time.time()
        while time.time() - t0 < ready_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited rc={self.proc.returncode}; log: "
                    f"{Path(self.args.log).read_text(errors='replace')[-1200:]}")
            if port_answers(self.args.port, timeout=30):
                print(f"[serve] ready after {time.time()-t0:.0f}s "
                      f"(pid {self.proc.pid})", flush=True)
                return
            time.sleep(2)
        raise RuntimeError(f"not ready in {ready_timeout}s")

    def stop(self):
        if self.proc is None:
            return
        pid = self.proc.pid
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(75):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.2)
            if self.proc.poll() is None:
                os.kill(pid, signal.SIGKILL)
            self.proc.wait(10)
        except (ProcessLookupError, PermissionError):
            pass          # already gone; never signal anything but this pid
        self.proc = None
        print(f"[serve] torn down pid {pid}", flush=True)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run(args, cases: list[Case]):
    model_name = Path(args.model).stem
    if model_name.endswith("-Q8_0"):
        model_name = model_name[:-len("-Q8_0")]
    out_path = HERE / f"results_noop_fp_{model_name}.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line).get("id"))
            except (ValueError, AttributeError):
                pass
        print(f"resume: {len(done)} case(s) already scored", flush=True)

    server = Server(args)
    server.start()

    def one(case: Case) -> dict:
        prompt, truncated = build_prompt(case.lines, case.cursor_line,
                                         case.cursor_char, case.rel_path)
        rec = dict(id=case.id, cls=case.cls, kind=case.kind,
                   source=case.package, fn=case.fn, model=model_name,
                   expectation=case.expectation,
                   prompt_chars=len(prompt), truncated_prefix=truncated)
        try:
            text, ntok, dt = complete(args.port, prompt, EXT_MAX_TOKENS,
                                      EXT_STOPS)
            pred = parse_prediction(text)
            rec.update(latency_s=round(dt, 3), completion_tokens=ntok,
                       proposal=int(len(pred) > 0), n_pred_lines=len(pred),
                       pred_chars=len("\n".join(pred)),
                       pred_first=(pred[0][:120] if pred else ""),
                       raw_head=text[:160])
        except Exception as e:                     # noqa: BLE001 — scored
            rec.update(latency_s=0.0, completion_tokens=0, proposal=0,
                       n_pred_lines=0, pred_chars=0, pred_first="",
                       raw_head="", error=str(e)[:160])
        return rec

    try:
        todo = [c for c in cases if c.id not in done]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for rec in pool.map(one, todo):
                with open(out_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(json.dumps({k: rec[k] for k in
                                  ("id", "cls", "kind", "proposal",
                                   "n_pred_lines", "latency_s")}), flush=True)
    finally:
        server.stop()

    # ---- aggregate over ALL rows in the file (incl. resumed) ----------
    rows = [json.loads(l) for l in open(out_path)]
    agg: dict = {}
    for cls in sorted({r["cls"] for r in rows}):
        rs = [r for r in rows if r["cls"] == cls]
        lat = sorted(r["latency_s"] for r in rs)
        prop = [r for r in rs if r.get("proposal")]
        agg[cls] = dict(
            n=len(rs),
            proposal_rate=round(len(prop) / len(rs), 4),
            scored_as_fp=(rs[0]["expectation"] == "no_proposal"),
            mean_pred_chars=round(sum(r["pred_chars"] for r in prop)
                                  / max(1, len(prop)), 1),
            mean_completion_tokens=round(
                sum(r["completion_tokens"] for r in rs) / len(rs), 1),
            p50_latency_s=lat[len(lat) // 2],
            p95_latency_s=lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))],
            sample_proposal=(prop[0]["pred_first"] if prop else ""),
        )
    fp_rows = [r for r in rows if r["expectation"] == "no_proposal"]
    fp = sum(1 for r in fp_rows if r.get("proposal"))
    verdict = dict(
        model=model_name, gguf=str(Path(args.model).resolve()),
        port=args.port, external=args.external, workers=args.workers,
        temperature=0, max_tokens=EXT_MAX_TOKENS, stop=EXT_STOPS,
        render="extension v0.0.6 (context_build.buildScopedPrompt scope=null)",
        acceptance="extension.parsePrediction non-empty == ghost text shown",
        cases_total=len(rows),
        no_op_cases=len(fp_rows),
        FALSE_POSITIVE_RATE=round(fp / max(1, len(fp_rows)), 4),
        by_class=agg,
    )
    print(json.dumps(dict(VERDICT=verdict), indent=1))   # LAST, always
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="GGUF path")
    ap.add_argument("--port", type=int, default=18095)
    ap.add_argument("--external", action="store_true",
                    help="use an already-answering server; never spawn/kill")
    ap.add_argument("--spawn-cpu", action="store_true",
                    help="spawn own server CPU-only (-ngl 0, default)")
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER)
    ap.add_argument("--corpus-functions", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--log", type=Path,
                    default=HERE / "llama-server-noop-fp.log")
    args = ap.parse_args()
    if not args.external and not Path(args.model).exists():
        sys.exit(f"missing model: {args.model}")
    if not Path(args.server_bin).exists() and not args.external:
        sys.exit(f"missing server binary: {args.server_bin}")

    t0 = time.time()
    cases = build_cases(args.corpus_functions, seed=args.seed)
    per = {}
    for c in cases:
        per[c.cls] = per.get(c.cls, 0) + 1
    print(json.dumps(dict(cases=len(cases), per_class=per,
                          authored_sources=len(AUTHORED),
                          elapsed_s=round(time.time() - t0, 1))), flush=True)
    if len(cases) < 120:
        sys.exit(f"only {len(cases)} cases (< 120)")
    run(args, cases)


if __name__ == "__main__":
    main()
