"""Coverage grid for synthetic R data — input diversity lives HERE, by construction;
the generator (and any DSPy optimizer) only optimizes output quality on this
fixed distribution."""
import random

DOMAINS = [
    "clinical trial efficacy summary (ADaM ADTTE/ADVS/ADSL datasets)",
    "safety data review (ADaM ADAE/ADLB, shift tables, adverse event rates)",
    "pharmacokinetic data processing and summary (non-compartmental flavor)",
    "survival analysis for oncology endpoints (Kaplan-Meier, Cox)",
    "laboratory data QC and outlier flagging",
    "survey / observational data weighting and tabulation",
    "genomics exploratory analysis (Bioconductor-style SummarizedExperiment)",
    "time series forecasting for operational data",
    "statistical report assembly (Rmd/Quarto-style code chunks)",
    "interactive summary dashboard logic (shiny server functions)",
]
PACKAGES = [
    "tidyverse (dplyr/tidyr/purrr)",
    "data.table",
    "dplyr + gtsummary",
    "survival + broom",
    "ggplot2 + scales",
    "tidyr + forcats + stringr",
    "broom + purrr (many-models pattern)",
    "data.table + ggplot2",
]
CONSTRUCTS = [
    "import and clean, handling NA/missing codes",
    "grouped summarization with multiple stats",
    "derive a new variable from existing columns (dplyr::case_when flavor)",
    "fit a model and tidy the results",
    "produce a publication-style plot with facets",
    "build a summary table object",
    "write a reusable function with argument checks",
    "reshaping long/wide and joining two tables",
]
REAL_ROXYGEN_HINTS = [
    "keep base-R idioms where natural", "prefer the native pipe |>",
    "use tidyverse pipes consistently", "include on.exit() cleanup where relevant",
]


def cell(rng):
    return {
        "domain": rng.choice(DOMAINS),
        "packages": rng.choice(PACKAGES),
        "construct": rng.choice(CONSTRUCTS),
        "style": rng.choice(REAL_ROXYGEN_HINTS),
        "line_target": rng.choice([8, 12, 16, 20, 25]),
    }


ANALYST_PROMPT = """You write realistic R code for pharmaceutical/biostatistical analysis teams.

Task: generate ONE authentic analyst-style R code snippet.

Domain: {domain}
Primary packages: {packages}
What the code should do: {construct}
Style: {style}
Target length: ~{line_target} lines of code.

Realism requirements (mimic real analyst code):
- Use plausible data-frame column names in the domain's conventions (e.g. USUBJID, TRTP, PARAMCD, AVISIT, AVAL, BASE for clinical; other conventions for other domains).
- Read data from a plausible source (csv, or assume a data.frame already in scope with a realistic name).
- Include the small imperfections of real work: a comment or two, a filter for missing values, sensible fallbacks.
- ONLY use functions that actually exist in base R or the named packages. No invented function names.

Respond ONLY with a JSON object:
{{"intent": "<one sentence describing what the snippet does>",
  "code": "<the R code, as a single string with \\n line breaks>",
  "packages_used": ["<pkg>", ...]}}"""

FINISH_PROMPT = """You are completing R package code. Given the documented signature below,
write a realistic function body.

{roxygen}
{signature} {{
  # ... body
}}

Requirements:
- The body must implement exactly what the documentation describes.
- Follow the package's conventions visible in the signature (rlang/tidyverse style if applicable).
- Keep it idiomatic and of realistic length ({line_target} lines max).
- ONLY use functions that actually exist. No invented names.

Respond ONLY with a JSON object:
{{"intent": "<one sentence on what the function does>",
  "body": "<the R body lines as a single string with \\n line breaks, no outer braces>",
  "packages_used": ["<pkg>", ...]}}"""
