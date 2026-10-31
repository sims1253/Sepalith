# SYNTH-v3 audit sample

Bound to `/mnt/e/sepalith/campaign-20260915/data-work/SYNTH-v3/manifest.json` sha256 `48e9a9192fc40ca686d99a3d7f06ff1b763c4de8e27177a0f0663f6ad9bd28bb` (build-summary.json `762e697da4f520ea57c59f121049e9107b2620881e33aa865db8a7ce753547de`).

Stratified random sample (seed 20261022) of 60 of 1834 cases. Strata: {"finish_block": {"constrained_short": 22, "sibling:short": 6, "sibling:medium": 16, "sibling:long": 4}, "roxygen_drafting": {"roxy_full": 12}}.

Each case shows the edited window as a diff (prefix + region + suffix, before and after the gold edit),
the gold target, the verification record and the exact rendered prompt of the drop-in TRAIN row.

## 1. `84117570a1059f6795f97951` finish_block / constrained_short

- package `synth:gaugemark`, path `R/convert.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": true, "novel_tokens": ["UseMethod"], "novelty": 1, "rule": "fb_statement", "seed": "b28-s3/gaugemark/R/convert.R", "signature_region": true, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "convertUnit", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 25, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 228, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -6,3 +6,5 @@
 #' @return An object like `x`, expressed in `to`.
 #' @export
 convertUnit <- function(x, to, ...) {
+  UseMethod("convertUnit")
+}
```

**Target (gold region)**

```r
convertUnit <- function(x, to, ...) {
  UseMethod("convertUnit")
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/convert.R
#' @title Convert lengths to another unit
#' @description Generic with methods for readings, ranges and plain numbers.
#' @param x An object to convert.
#' @param to The target unit, one of `"mm"`, `"cm"`, `"m"`, `"km"` or `"in"`.
#' @param ... Passed to methods.
#' @return An object like `x`, expressed in `to`.
#' @export
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
convertUnit <- function(x, to, ...) {<|user_cursor|>
=======
<[fim-middle]>
````

</details>

## 2. `e767704fc6c398e550a92a32` finish_block / constrained_short

- package `synth:fluxunit`, path `R/convert.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["factor"], "novelty": 1, "rule": "fb_statement", "seed": "b40-science/fluxunit/R/convert.R", "signature_region": false, "structural_support": 0.0, "target_lines": 8, "test_file": false, "unit": "pretty_unit", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 44, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 220, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -9,3 +9,11 @@
   unit <- if (length(fits) == 0L) {
     names(factors)[1]
   } else {
+    names(fits)[length(fits)]
+  }
+  list(
+    unit = unit,
+    factor = factors[[unit]],
+    value = x / factors[[unit]]
+  )
+}
```

**Target (gold region)**

```r
    names(fits)[length(fits)]
  }
  list(
    unit = unit,
    factor = factors[[unit]],
    value = x / factors[[unit]]
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/convert.R
pretty_unit <- function(x, quantity = "length", candidates = NULL) {
  factors <- unit_factors(quantity)
  if (!is.null(candidates)) {
    factors <- factors[intersect(candidates, names(factors))]
  }
  factors <- sort(factors)
  typical <- stats::median(abs(x), na.rm = TRUE)
  fits <- factors[factors <= typical]
  unit <- if (length(fits) == 0L) {
    names(factors)[1]
  } else {
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 3. `675928edd27b641c09cdfb9e` finish_block / constrained_short

- package `synth:relaydesk`, path `R/responses.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["error"], "novelty": 1, "rule": "fb_statement", "seed": "b18-web/relaydesk/R/responses.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "sendOk", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 17, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 131, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -1,2 +1,4 @@
 sendOk <- function(res, data, status = 200L) {
   res$status <- status
+  list(data = data, error = NULL)
+}
```

**Target (gold region)**

```r
  list(data = data, error = NULL)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/responses.R
sendOk <- function(res, data, status = 200L) {
  res$status <- status
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 4. `397ccae08f4affd41bb9df09` finish_block / constrained_short

- package `synth:gaugemark`, path `R/display.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["trimws"], "novelty": 1, "rule": "fb_statement", "seed": "b28-s3/gaugemark/R/display.R", "signature_region": false, "structural_support": 0.0, "target_lines": 4, "test_file": false, "unit": "format.gm_range", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 54, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 474, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -37,3 +37,7 @@
       upper <- formatC(bounds$upper, digits = digits, format = "e")
     }
   )
+  out <- paste0(trimws(lower), "-", trimws(upper), " ", unit)
+  out[is.na(bounds$lower) | is.na(bounds$upper)] <- "NA"
+  out
+}
```

**Target (gold region)**

```r
  out <- paste0(trimws(lower), "-", trimws(upper), " ", unit)
  out[is.na(bounds$lower) | is.na(bounds$upper)] <- "NA"
  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/display.R
# ---- diagnostics (static) --------------------
# typed statements: 4 of 7
# args not yet referenced: none
# unclosed braces in typed code: 1
# ----------------------------------------------
#' @title Format ranges
#' @description Each range is rendered as `lower-upper unit`; ranges with a
#'   missing bound as `NA`. `notation` works as in [format.gm_reading()] and
#'   applies to both bounds.
#' @param x An object of class `gm_range`.
#' @param digits Number of digits, interpreted per `notation`.
#' @param notation One of `"plain"`, `"fixed"` or `"scientific"`.
#' @param ... Unused.
#' @return A character vector with one element per range.
#' @export
format.gm_range <- function(
  x,
  digits = 3,
  notation = c("plain", "fixed", "scientific"),
  ...
) {
  notation <- match.arg(notation)
  unit <- attr(x, "unit")
  bounds <- unclass(x)
  switch(
    notation,
    plain = {
      lower <- as.character(signif(bounds$lower, digits))
      upper <- as.character(signif(bounds$upper, digits))
    },
    fixed = {
      lower <- formatC(bounds$lower, digits = digits, format = "f")
      upper <- formatC(bounds$upper, digits = digits, format = "f")
    },
    scientific = {
      lower <- formatC(bounds$lower, digits = digits, format = "e")
      upper <- formatC(bounds$upper, digits = digits, format = "e")
    }
  )
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 5. `0c340b4c843bd974f1179098` finish_block / constrained_short

- package `synth:tidemark`, path `R/series.R`
- generator: `{"behaviour": "unit_not_traceable", "bucket": "short", "determinacy": "constrained_short", "full_body": true, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b29-s4/tidemark/R/series.R", "signature_region": true, "structural_support": 0.667, "target_lines": 2, "test_file": false, "unit": "setMethod", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 45, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 475, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -36,3 +36,5 @@
 
 #' @export
 setMethod("Arith", signature("TideSeries", "TideSeries"), function(e1, e2) {
+  seriesSeriesOp(e1, e2, .Generic)
+})
```

**Target (gold region)**

```r
setMethod("Arith", signature("TideSeries", "TideSeries"), function(e1, e2) {
  seriesSeriesOp(e1, e2, .Generic)
})
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/series.R
#' Arithmetic between two TideSeries
#'
#' @param e1,e2 `TideSeries` objects on the same time grid.
#' @param op The operator name, as found in `.Generic`.
#' @return A `TideSeries` in the units of `e1`.
#' @keywords internal
seriesSeriesOp <- function(e1, e2, op) {
  if (!identical(e1@times, e2@times)) {
    stop("Both series must share the same `times`.", call. = FALSE)
  }
  e2 <- convertHeights(e2, e1@units)
  if (!op %in% c("+", "-")) {
    stop(
      sprintf("Operator `%s` is not supported for TideSeries.", op),
      call. = FALSE
    )
  }
  station <- if (identical(e1@station, e2@station)) e1@station else "combined"
  tideSeries(
    station,
    e1@times,
    seriesArith(e1@heights, e2, op)@heights,
    e1@units
  )
}

#' @export
setMethod("Arith", signature("TideSeries", "numeric"), function(e1, e2) {
  seriesArith(e1, e2, .Generic)
})

#' @export
setMethod("Arith", signature("numeric", "TideSeries"), function(e1, e2) {
  seriesArith(e1, e2, .Generic)
})

#' @export
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
setMethod("Arith", signature("TideSeries", "TideSeries"), function(e1, e2) {<|user_cursor|>
=======
<[fim-middle]>
````

</details>

## 6. `543bddce84f77389c3c27529` finish_block / constrained_short

- package `synth:itemwise`, path `R/norms.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": true, "novel_tokens": ["num:10", "num:50"], "novelty": 2, "rule": "fb_statement", "seed": "b15-psych-trials/itemwise/R/norms.R", "signature_region": true, "structural_support": 0.0, "target_lines": 3, "test_file": false, "unit": "to_t_score", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 45, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 343, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -18,3 +18,6 @@
 #' @rdname to_z_score
 #' @export
 to_t_score <- function(x, ref_mean, ref_sd) {
+  z <- to_z_score(x, ref_mean, ref_sd)
+  50 + 10 * z
+}
```

**Target (gold region)**

```r
to_t_score <- function(x, ref_mean, ref_sd) {
  z <- to_z_score(x, ref_mean, ref_sd)
  50 + 10 * z
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/norms.R
#' Convert raw scores to standard norm scales
#'
#' `to_z_score()` standardises raw scores against a reference mean and
#' standard deviation, `to_t_score()` rescales them to a mean of 50 and a
#' standard deviation of 10, and `to_stanine()` maps them onto the nine-point
#' stanine scale, clamped to 1-9.
#'
#' @param x A numeric vector of raw scale scores.
#' @param ref_mean Mean of the reference (norm) sample.
#' @param ref_sd Standard deviation of the reference (norm) sample.
#' @return A numeric vector of z scores, T scores or (integer) stanines.
#' @export
to_z_score <- function(x, ref_mean, ref_sd) {
  stopifnot(ref_sd > 0)
  (x - ref_mean) / ref_sd
}

#' @rdname to_z_score
#' @export
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
to_t_score <- function(x, ref_mean, ref_sd) {<|user_cursor|>
=======
<[fim-middle]>
````

</details>

## 7. `4e7133996ecc12ffbdfc6c98` finish_block / constrained_short

- package `synth:calendrix`, path `R/parse.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["parsed", "which.max"], "novelty": 2, "rule": "fb_statement", "seed": "b32-dates/calendrix/R/parse.R", "signature_region": false, "structural_support": 0.0, "target_lines": 5, "test_file": false, "unit": "guess_date_format", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 47, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 135, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -1,2 +1,7 @@
 guess_date_format <- function(x, candidates = names(DATE_PATTERNS)) {
   report <- date_format_report(x, candidates)
+  if (max(report$parsed) == 0L) {
+    return(NA_character_)
+  }
+  return(report$format[which.max(report$parsed)])
+}
```

**Target (gold region)**

```r
  if (max(report$parsed) == 0L) {
    return(NA_character_)
  }
  return(report$format[which.max(report$parsed)])
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/parse.R
guess_date_format <- function(x, candidates = names(DATE_PATTERNS)) {
  report <- date_format_report(x, candidates)
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 8. `51015d13c2af9f249ef85cf6` finish_block / constrained_short

- package `synth:fitwright`, path `R/bootstrap.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b02-base-stats/fitwright/R/bootstrap.R", "signature_region": false, "structural_support": 0.556, "target_lines": 4, "test_file": false, "unit": "boot_ci_basic", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 36, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 647, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -58,3 +58,7 @@
     na.rm = TRUE
   )
   # Reflect the quantiles around the original estimate.
+  out <- 2 * boot$t0 - t(q)
+  colnames(out) <- c("lower", "upper")
+  out
+}
```

**Target (gold region)**

```r
  out <- 2 * boot$t0 - t(q)
  colnames(out) <- c("lower", "upper")
  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/bootstrap.R
#' @param strata Optional vector of stratum labels, one per observation.
#' @return An object of class `fw_boot`.
#' @export
boot_stat <- function(data, statistic, R = 999L, strata = NULL) {
  statistic <- match.fun(statistic)
  is_df <- is.data.frame(data)
  n <- if (is_df) nrow(data) else length(data)

  t0 <- statistic(data)
  if (is.null(names(t0))) {
    names(t0) <- paste0("t", seq_along(t0))
  }

  idx <- boot_indices(n, R, strata)
  t <- vapply(
    seq_len(R),
    function(r) {
      rows <- idx[r, ]
      unname(statistic(if (is_df) data[rows, , drop = FALSE] else data[rows]))
    },
    numeric(length(t0))
  )
  # vapply() returns one column per resample, or a plain vector for scalars.
  t <- if (length(t0) == 1L) matrix(t, ncol = 1L) else t(t)
  new_fw_boot(t0, t, match.call())
}

#' @title Bootstrap interval helpers
#' @description Percentile, basic (reflected percentile) and bias-corrected
#'   normal-approximation intervals from the replicates of an `fw_boot`.
#' @param boot An object of class `fw_boot`.
#' @param level Confidence level of the interval.
#' @return A matrix with columns `lower` and `upper`, one row per statistic.
#' @export
boot_ci_percentile <- function(boot, level = 0.95) {
  alpha <- (1 - level) / 2
  q <- apply(
    boot$t,
    2,
    stats::quantile,
    probs = c(alpha, 1 - alpha),
    na.rm = TRUE
  )
  out <- t(q)
  colnames(out) <- c("lower", "upper")
  out
}

#' @rdname boot_ci_percentile
#' @export
boot_ci_basic <- function(boot, level = 0.95) {
  alpha <- (1 - level) / 2
  q <- apply(
    boot$t,
    2,
    stats::quantile,
    probs = c(1 - alpha, alpha),
    na.rm = TRUE
  )
  # Reflect the quantiles around the original estimate.
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 9. `0054b5c578707ae1dcdb2306` finish_block / constrained_short

- package `synth:pkgsmith`, path `R/semver.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b22-devtools/pkgsmith/R/semver.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "bump_patch", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 13, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 409, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -33,3 +33,5 @@
   v$patch <- v$patch + 1L
   v$dev <- 0L
   v$pre <- NA_character_
+  semver_format(v)
+}
```

**Target (gold region)**

```r
  semver_format(v)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/semver.R
#' @title Bump one component of a version
#' @description Each bump resets the components below it and drops any pre-
#'   release tag. `bump_dev()` starts the development series at `9000` when the
#'   version has none.
#' @param x A single version string.
#' @return The bumped version string.
#' @export
bump_major <- function(x) {
  v <- semver_parse(x)
  v$major <- v$major + 1L
  v$minor <- 0L
  v$patch <- 0L
  v$dev <- 0L
  v$pre <- NA_character_
  semver_format(v)
}

#' @rdname bump_major
#' @export
bump_minor <- function(x) {
  v <- semver_parse(x)
  v$minor <- v$minor + 1L
  v$patch <- 0L
  v$dev <- 0L
  v$pre <- NA_character_
  semver_format(v)
}

#' @rdname bump_major
#' @export
bump_patch <- function(x) {
  v <- semver_parse(x)
  v$patch <- v$patch + 1L
  v$dev <- 0L
  v$pre <- NA_character_
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 10. `d97dbf9be58da91c43213ba3` finish_block / constrained_short

- package `synth:gradeledger`, path `R/rubric.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["num:5"], "novelty": 1, "rule": "fb_comment_lead", "seed": "b44-analytics/gradeledger/R/rubric.R", "signature_region": false, "structural_support": 0.894, "target_lines": 8, "test_file": false, "unit": "rubricAgreement", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 80, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 497, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -33,3 +33,11 @@
   out$exact[4] <- mean(d == 0)
   out$adjacent[4] <- mean(abs(d) <= 1)
   out$meanDiff[4] <- mean(d)
+  # Agreement on `originality`
+  d <- raterA$originality - raterB$originality
+  out$exact[5] <- mean(d == 0)
+  out$adjacent[5] <- mean(abs(d) <= 1)
+  out$meanDiff[5] <- mean(d)
+
+  return(out)
+}
```

**Target (gold region)**

```r
  # Agreement on `originality`
  d <- raterA$originality - raterB$originality
  out$exact[5] <- mean(d == 0)
  out$adjacent[5] <- mean(abs(d) <= 1)
  out$meanDiff[5] <- mean(d)

  return(out)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/rubric.R
rubricAgreement <- function(raterA, raterB) {
  if (nrow(raterA) != nrow(raterB)) {
    stop(
      "`raterA` and `raterB` must have the same number of rows.",
      call. = FALSE
    )
  }
  criteria <- c("clarity", "evidence", "structure", "mechanics", "originality")
  out <- data.frame(
    criterion = criteria,
    exact = NA_real_,
    adjacent = NA_real_,
    meanDiff = NA_real_
  )

  # Agreement on `clarity`
  d <- raterA$clarity - raterB$clarity
  out$exact[1] <- mean(d == 0)
  out$adjacent[1] <- mean(abs(d) <= 1)
  out$meanDiff[1] <- mean(d)
  # Agreement on `evidence`
  d <- raterA$evidence - raterB$evidence
  out$exact[2] <- mean(d == 0)
  out$adjacent[2] <- mean(abs(d) <= 1)
  out$meanDiff[2] <- mean(d)
  # Agreement on `structure`
  d <- raterA$structure - raterB$structure
  out$exact[3] <- mean(d == 0)
  out$adjacent[3] <- mean(abs(d) <= 1)
  out$meanDiff[3] <- mean(d)
  # Agreement on `mechanics`
  d <- raterA$mechanics - raterB$mechanics
  out$exact[4] <- mean(d == 0)
  out$adjacent[4] <- mean(abs(d) <= 1)
  out$meanDiff[4] <- mean(d)
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 11. `2add0dbdf1397df015e48253` finish_block / constrained_short

- package `synth:rowcast`, path `R/profile.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_comment_lead", "seed": "b45-sql/rowcast/R/profile.R", "signature_region": false, "structural_support": 0.111, "target_lines": 6, "test_file": false, "unit": "column.profile", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 57, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 498, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -37,3 +37,9 @@
   if ("max" %in% stats) {
     out$max <- if (ordered && length(present) > 0) max(present) else NA
   }
+  # Statistic `mean`
+  if ("mean" %in% stats) {
+    out$mean <- if (is.numeric(x) && length(present) > 0) mean(present) else NA
+  }
+  out
+}
```

**Target (gold region)**

```r
  # Statistic `mean`
  if ("mean" %in% stats) {
    out$mean <- if (is.numeric(x) && length(present) > 0) mean(present) else NA
  }
  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/profile.R
#' Profile one result column
#'
#' Computes the requested statistics; `min`, `max` and `mean` are `NA` when they
#' do not apply to the column type.
#'
#' @param x A result column.
#' @param stats Any of `"n"`, `"n_missing"`, `"n_distinct"`, `"min"`, `"max"` and `"mean"`.
#' @return A named list with one element per requested statistic.
#' @export
column.profile <- function(
  x,
  stats = c("n", "n_missing", "n_distinct", "min", "max", "mean")
) {
  stats <- match.arg(stats, several.ok = TRUE)
  ordered <- is.numeric(x) ||
    inherits(x, c("Date", "POSIXct")) ||
    is.character(x)
  present <- x[!is.na(x)]
  out <- list()
  # Statistic `n`
  if ("n" %in% stats) {
    out$n <- length(x)
  }
  # Statistic `n_missing`
  if ("n_missing" %in% stats) {
    out$n_missing <- sum(is.na(x))
  }
  # Statistic `n_distinct`
  if ("n_distinct" %in% stats) {
    out$n_distinct <- length(unique(present))
  }
  # Statistic `min`
  if ("min" %in% stats) {
    out$min <- if (ordered && length(present) > 0) min(present) else NA
  }
  # Statistic `max`
  if ("max" %in% stats) {
    out$max <- if (ordered && length(present) > 0) max(present) else NA
  }
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 12. `10cb9c7dc504782a8ef8016b` finish_block / constrained_short

- package `synth:querywright`, path `R/report.R`
- generator: `{"behaviour": "no_check", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b18-web/querywright/R/report.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "save_report", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 11, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 167, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -5,3 +5,5 @@
     dir.create(dir, recursive = TRUE)
   }
   save_html(page, file = path)
+  invisible(path)
+}
```

**Target (gold region)**

```r
  invisible(path)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/report.R
save_report <- function(page, path) {
  # Make sure the target directory exists before writing.
  dir <- dirname(path)
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
  save_html(page, file = path)
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 13. `431487a812cfb95cd5163580` finish_block / constrained_short

- package `synth:siteledger`, path `R/site-meta.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["lon", "num:180"], "novelty": 2, "rule": "fb_statement", "seed": "b40-science/siteledger/R/site-meta.R", "signature_region": false, "structural_support": 0.558, "target_lines": 5, "test_file": false, "unit": "site.bbox", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 59, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 275, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -15,3 +15,8 @@
   north <- min(max(meta$lat) + pad, 90)
 
   # Longitude
+  west <- max(min(meta$lon) - pad, -180)
+  east <- min(max(meta$lon) + pad, 180)
+
+  c(south = south, north = north, west = west, east = east)
+}
```

**Target (gold region)**

```r
  west <- max(min(meta$lon) - pad, -180)
  east <- min(max(meta$lon) + pad, 180)

  c(south = south, north = north, west = west, east = east)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/site-meta.R
#' Bounding box of a set of sites
#'
#' @param meta A site metadata table from [site.register()].
#' @param pad Margin in degrees added on every side.
#' @export
#' @return A named numeric vector with elements `south`, `north`, `west`
#'   and `east`, clipped to valid coordinates.
site.bbox <- function(meta, pad = 0) {
  if (nrow(meta) == 0L) {
    stop("`meta` has no sites.", call. = FALSE)
  }

  # Latitude
  south <- max(min(meta$lat) - pad, -90)
  north <- min(max(meta$lat) + pad, 90)

  # Longitude
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 14. `405b0d8ae5d0ba436af508e7` finish_block / constrained_short

- package `synth:terraslab`, path `R/tiles.R`
- generator: `{"behaviour": "no_check", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["validCells"], "novelty": 1, "rule": "fb_nested_close", "seed": "b13-geo/terraslab/R/tiles.R", "signature_region": false, "structural_support": 0.0, "target_lines": 5, "test_file": false, "unit": "tileValidCounts", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 23, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 211, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -6,3 +6,8 @@
   for (i in seq_len(nrow(scheme))) {
     piece <- terra::crop(r[[1]], tileExtent(r, scheme[i, ]))
     counts[i] <- as.integer(terra::global(!is.na(piece), "sum")[1, 1])
+  }
+
+  scheme$validCells <- counts
+  return(scheme)
+}
```

**Target (gold region)**

```r
  }

  scheme$validCells <- counts
  return(scheme)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/tiles.R
tileValidCounts <- function(r, ...) {
  scheme <- tileScheme(r, ...)
  counts <- integer(nrow(scheme))

  # Only the first layer is inspected; layers share the same footprint
  for (i in seq_len(nrow(scheme))) {
    piece <- terra::crop(r[[1]], tileExtent(r, scheme[i, ]))
    counts[i] <- as.integer(terra::global(!is.na(piece), "sum")[1, 1])
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 15. `71a37f2117038c6840d8e3fd` finish_block / constrained_short

- package `synth:trophicstan`, path `R/diversity.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b08-rcpp-science/trophicstan/R/diversity.R", "signature_region": false, "structural_support": 0.0, "target_lines": 5, "test_file": false, "unit": "richness_chao1", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 66, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 221, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -11,3 +11,8 @@
   n <- sum(counts)
 
   if (f2 > 0) {
+    s_obs + (n - 1) / n * f1^2 / (2 * f2)
+  } else {
+    s_obs + (n - 1) / n * f1 * (f1 - 1) / 2
+  }
+}
```

**Target (gold region)**

```r
    s_obs + (n - 1) / n * f1^2 / (2 * f2)
  } else {
    s_obs + (n - 1) / n * f1 * (f1 - 1) / 2
  }
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/diversity.R
richness_chao1 <- function(counts) {
  counts <- round(counts[!is.na(counts) & counts > 0])
  s_obs <- length(counts)
  if (s_obs == 0L) {
    return(0)
  }

  # Singletons and doubletons drive the estimate of unseen species
  f1 <- sum(counts == 1)
  f2 <- sum(counts == 2)
  n <- sum(counts)

  if (f2 > 0) {
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 16. `fc22df81957cf6a43b8dd794` finish_block / constrained_short

- package `synth:tokentrellis`, path `R/trim.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["strrep"], "novelty": 1, "rule": "fb_statement", "seed": "b31-text/tokentrellis/R/trim.R", "signature_region": false, "structural_support": 0.0, "target_lines": 6, "test_file": false, "unit": "padText", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 42, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 219, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -13,3 +13,9 @@
     },
     both = {
       leftPad <- missing %/% 2L
+      rightPad <- missing - leftPad
+    }
+  )
+
+  paste0(strrep(pad, leftPad), x, strrep(pad, rightPad))
+}
```

**Target (gold region)**

```r
      rightPad <- missing - leftPad
    }
  )

  paste0(strrep(pad, leftPad), x, strrep(pad, rightPad))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/trim.R
padText <- function(x, width, side = c("left", "right", "both"), pad = " ") {
  side <- match.arg(side)
  missing <- pmax(width - nchar(x), 0L)
  switch(
    side,
    left = {
      leftPad <- missing
      rightPad <- 0L
    },
    right = {
      leftPad <- 0L
      rightPad <- missing
    },
    both = {
      leftPad <- missing %/% 2L
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 17. `20cfadc2c1b016afcf1bd0bb` finish_block / constrained_short

- package `synth:inkplate`, path `R/table.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["dimnames", "matrix"], "novelty": 2, "rule": "fb_statement", "seed": "b10-text/inkplate/R/table.R", "signature_region": false, "structural_support": 0.0, "target_lines": 8, "test_file": false, "unit": "format.cells", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 46, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 248, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -15,3 +15,11 @@
   )
 
   # explicit ncol: vapply() collapses to a bare vector when there are no rows
+  cells <- matrix(
+    cells,
+    nrow = nrow(data),
+    ncol = length(data),
+    dimnames = list(NULL, names(data))
+  )
+  cells
+}
```

**Target (gold region)**

```r
  cells <- matrix(
    cells,
    nrow = nrow(data),
    ncol = length(data),
    dimnames = list(NULL, names(data))
  )
  cells
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/table.R
format.cells <- function(data, digits = 2L, na = "") {
  cells <- vapply(
    data,
    function(col) {
      # integers keep their natural form; only doubles get fixed decimals
      out <- if (is.numeric(col) && !is.integer(col)) {
        formatC(col, format = "f", digits = digits, big.mark = ",")
      } else {
        as.character(col)
      }
      out[is.na(col)] <- na
      out
    },
    character(nrow(data))
  )

  # explicit ncol: vapply() collapses to a bare vector when there are no rows
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 18. `308cedd1ab70bd443619017b` finish_block / constrained_short

- package `synth:foliokit`, path `R/ledger.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": true, "novel_tokens": ["is.data.frame"], "novelty": 1, "rule": "fb_statement", "seed": "b05-oop/foliokit/R/ledger.R", "signature_region": true, "structural_support": 0.0, "target_lines": 6, "test_file": false, "unit": "new_ledger", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 60, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 250, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -9,3 +9,9 @@
 #' @return A `foliokit_ledger` object.
 #' @keywords internal
 new_ledger <- function(entries, currency = "USD") {
+  stopifnot(is.data.frame(entries))
+  structure(
+    list(entries = entries, currency = currency),
+    class = "foliokit_ledger"
+  )
+}
```

**Target (gold region)**

```r
new_ledger <- function(entries, currency = "USD") {
  stopifnot(is.data.frame(entries))
  structure(
    list(entries = entries, currency = currency),
    class = "foliokit_ledger"
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/ledger.R
#' Construct a ledger
#'
#' Low-level constructor for the `foliokit_ledger` class. Assumes its input
#' is already well-formed; use [ledger()] for user-facing input.
#'
#' @param entries A data frame with columns `date`, `type`, `symbol`,
#'   `quantity`, `price` and `amount`.
#' @param currency A three-letter ISO currency code.
#' @return A `foliokit_ledger` object.
#' @keywords internal
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
new_ledger <- function(entries, currency = "USD") {<|user_cursor|>
=======
<[fim-middle]>
````

</details>

## 19. `c5f725bced3996388064f410` finish_block / constrained_short

- package `synth:quantia`, path `R/accessors.R`
- generator: `{"behaviour": "unit_not_traceable", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["unit", "validObject"], "novelty": 2, "rule": "fb_nested_close", "seed": "b05-oop/quantia/R/accessors.R", "signature_region": false, "structural_support": 0.0, "target_lines": 5, "test_file": false, "unit": "setReplaceMethod", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 24, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 141, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -1,3 +1,8 @@
 setReplaceMethod("unitOf", "Quantity", function(object, value) {
   if (is.character(value)) {
     value <- lookupUnit(value)
+  }
+  object@unit <- value
+  validObject(object)
+  object
+})
```

**Target (gold region)**

```r
  }
  object@unit <- value
  validObject(object)
  object
})
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/accessors.R
setReplaceMethod("unitOf", "Quantity", function(object, value) {
  if (is.character(value)) {
    value <- lookupUnit(value)
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 20. `23f173701449657cd6d751fc` finish_block / constrained_short

- package `synth:inkplate`, path `R/escape.R`
- generator: `{"behaviour": "file_checked_not_entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b10-text/inkplate/R/escape.R", "signature_region": false, "structural_support": 0.0, "target_lines": 5, "test_file": false, "unit": "unescape.html", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 30, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 480, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -29,3 +29,8 @@
         as.integer(body)
       }
       out[i] <- stringi::stri_replace_all_fixed(out[i], ref, intToUtf8(num))
+    }
+  }
+
+  stringi::stri_replace_all_fixed(out, "\001", "&")
+}
```

**Target (gold region)**

```r
    }
  }

  stringi::stri_replace_all_fixed(out, "\001", "&")
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/escape.R
#' Decode common HTML entities
#'
#' Handles the named entities produced by [escape.attr()] plus decimal and
#' hexadecimal numeric references.
#'
#' @param x A character vector of HTML-escaped text.
#' @return A character vector the same length as `x`.
#' @export
unescape.html <- function(x) {
  # "&amp;" is protected first and decoded last, so "&amp;lt;" stays "&lt;"
  # and a decoded "&#38;" cannot start a second round of decoding
  out <- stringi::stri_replace_all_fixed(x, "&amp;", "\001")
  out <- out %>%
    stringi::stri_replace_all_fixed(
      c("&lt;", "&gt;", "&quot;", "&#39;", "&nbsp;"),
      c("<", ">", "\"", "'", " "),
      vectorize_all = FALSE
    )

  refs <- stringi::stri_extract_all_regex(out, "&#(x[0-9A-Fa-f]+|[0-9]+);")
  for (i in seq_along(out)) {
    codes <- unique(refs[[i]])
    codes <- codes[!is.na(codes)]
    for (ref in codes) {
      body <- stringi::stri_sub(ref, 3, -2)
      num <- if (startsWith(body, "x")) {
        strtoi(substring(body, 2), 16L)
      } else {
        as.integer(body)
      }
      out[i] <- stringi::stri_replace_all_fixed(out[i], ref, intToUtf8(num))
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 21. `490e324239bf67f0e039ffea` finish_block / constrained_short

- package `synth:pipewren`, path `R/reshape.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b34-tidyverse/pipewren/R/reshape.R", "signature_region": false, "structural_support": 0.0, "target_lines": 8, "test_file": false, "unit": "splitColumn", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 55, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 377, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -30,3 +30,11 @@
       }
     }
     p
+  })
+
+  data[[col]] <- NULL
+  for (i in seq_len(k)) {
+    data[[into[i]]] <- vapply(parts, `[`, character(1), i)
+  }
+  data
+}
```

**Target (gold region)**

```r
  })

  data[[col]] <- NULL
  for (i in seq_len(k)) {
    data[[into[i]]] <- vapply(parts, `[`, character(1), i)
  }
  data
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/reshape.R
splitColumn <- function(
  data,
  col,
  into,
  sep = "_",
  extra = c("drop", "merge"),
  fill = c("right", "left")
) {
  extra <- match.arg(extra)
  fill <- match.arg(fill)
  if (!col %in% names(data)) {
    stop("Unknown columns: ", col, call. = FALSE)
  }
  k <- length(into)
  parts <- strsplit(as.character(data[[col]]), sep, fixed = TRUE)

  # Pad or trim every row to `length(into)` pieces
  parts <- lapply(parts, function(p) {
    if (length(p) > k) {
      if (extra == "merge") {
        p <- c(p[seq_len(k - 1L)], paste(p[seq(k, length(p))], collapse = sep))
      } else {
        p <- p[seq_len(k)]
      }
    } else if (length(p) < k) {
      if (fill == "right") {
        p <- c(p, rep(NA_character_, k - length(p)))
      } else {
        p <- c(rep(NA_character_, k - length(p)), p)
      }
    }
    p
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 22. `eb3501537718e2a480fc023b` finish_block / constrained_short

- package `synth:pursekeep`, path `R/projection.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b04-shiny/pursekeep/R/projection.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "summarizeProjection", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 10, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 311, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -17,3 +17,5 @@
     c("month", "contributed", "interest", "balance")
   ]
   names(out) <- c("Month", "Contributed", "Interest earned", "Balance")
+  out
+}
```

**Target (gold region)**

```r
  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/projection.R
#' Summarise a projection at yearly milestones
#'
#' Picks the end of each year, plus the final month if it is not already a
#' year end.
#'
#' @keywords internal
#' @param projection A data frame as returned by [projectBalance()].
#' @return A data frame with one row per milestone.
summarizeProjection <- function(projection) {
  last <- max(projection$month)
  # seq(12, last, by = 12) fails for plans shorter than a year
  keep <- unique(c(seq_len(last %/% 12) * 12, last))
  keep <- keep[keep >= 1 & keep <= last]

  out <- projection[
    projection$month %in% keep,
    c("month", "contributed", "interest", "balance")
  ]
  names(out) <- c("Month", "Contributed", "Interest earned", "Balance")
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 23. `fb75d29f64c5e1e18a28a17b` finish_block / sibling:short

- package `synth:shiftplan`, path `R/packing.R`
- generator: `{"behaviour": "file_checked_not_entered", "bucket": "short", "determinacy": "sibling", "full_body": false, "novel_tokens": [".5", "num:0.5", "solution"], "novelty": 3, "rule": "fb_statement", "seed": "b19-graphs-or/shiftplan/R/packing.R", "signature_region": false, "structural_support": 0.741, "target_lines": 7, "test_file": false, "unit": "knapsack.lp", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 48, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 633, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -58,3 +58,10 @@
     stop("lpSolve failed with status ", sol$status, call. = FALSE)
   }
 
+  chosen <- sol$solution > 0.5
+  list(
+    chosen = chosen,
+    value = sum(value[chosen]),
+    weight = sum(weight[chosen])
+  )
+}
```

**Target (gold region)**

```r
  chosen <- sol$solution > 0.5
  list(
    chosen = chosen,
    value = sum(value[chosen]),
    weight = sum(weight[chosen])
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/packing.R
#' Solve a 0/1 knapsack greedily by value density
#'
#' Takes items in order of decreasing value per unit weight, then compares
#' against the single best item that fits, which bounds the error by half.
#'
#' @param value Numeric vector of item values.
#' @param weight Integer vector of item weights.
#' @param capacity Integer knapsack capacity.
#' @return A list with logical `chosen`, numeric `value` and integer `weight`.
#' @export
knapsack.greedy <- function(value, weight, capacity) {
  stopifnot(length(value) == length(weight), capacity >= 0)
  n <- length(value)
  chosen <- logical(n)
  load <- 0

  for (i in order(value / weight, decreasing = TRUE)) {
    if (load + weight[i] <= capacity) {
      chosen[i] <- TRUE
      load <- load + weight[i]
    }
  }

  # Density order alone can be arbitrarily bad; the best single item fixes that
  fits <- which(weight <= capacity)
  if (length(fits) > 0) {
    best.single <- fits[which.max(value[fits])]
    if (value[best.single] > sum(value[chosen])) {
      chosen <- logical(n)
      chosen[best.single] <- TRUE
    }
  }
  list(
    chosen = chosen,
    value = sum(value[chosen]),
    weight = sum(weight[chosen])
  )
}

#' Solve a 0/1 knapsack with lpSolve
#'
#' @param value Numeric vector of item values.
#' @param weight Integer vector of item weights.
#' @param capacity Integer knapsack capacity.
#' @return A list with logical `chosen`, numeric `value` and integer `weight`.
#' @export
knapsack.lp <- function(value, weight, capacity) {
  stopifnot(length(value) == length(weight), capacity >= 0)
  sol <- lpSolve::lp(
    direction = "max",
    objective.in = value,
    const.mat = matrix(weight, nrow = 1),
    const.dir = "<=",
    const.rhs = capacity,
    all.bin = TRUE
  )
  if (sol$status != 0) {
    stop("lpSolve failed with status ", sol$status, call. = FALSE)
  }

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 24. `018fb473488c5debafacc3c8` finish_block / sibling:short

- package `synth:dosewise`, path `R/patient.R`
- generator: `{"behaviour": "file_checked_not_entered", "bucket": "short", "determinacy": "sibling", "full_body": false, "novel_tokens": ["num:2.2", "num:2.7", "num:48"], "novelty": 3, "rule": "fb_statement", "seed": "b29-s4/dosewise/R/patient.R", "signature_region": false, "structural_support": 0.867, "target_lines": 8, "test_file": false, "unit": "ideal.body.weight", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 47, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 317, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -29,3 +29,11 @@
       }
     },
     hamwi = {
+      if (male) {
+        48 + 2.7 * inches.over
+      } else {
+        45.5 + 2.2 * inches.over
+      }
+    }
+  )
+}
```

**Target (gold region)**

```r
      if (male) {
        48 + 2.7 * inches.over
      } else {
        45.5 + 2.2 * inches.over
      }
    }
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/patient.R
ideal.body.weight <- function(
  x,
  method = c("devine", "robinson", "miller", "hamwi")
) {
  method <- match.arg(method)
  inches.over <- max(x@height.cm / 2.54 - 60, 0)
  male <- x@sex == "male"
  switch(
    method,
    devine = {
      if (male) {
        50 + 2.3 * inches.over
      } else {
        45.5 + 2.3 * inches.over
      }
    },
    robinson = {
      if (male) {
        52 + 1.9 * inches.over
      } else {
        49 + 1.7 * inches.over
      }
    },
    miller = {
      if (male) {
        56.2 + 1.41 * inches.over
      } else {
        53.1 + 1.36 * inches.over
      }
    },
    hamwi = {
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 25. `20418b9c4fd233164d972ad0` finish_block / sibling:medium

- package `synth:fieldpanel`, path `R/formatters.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b36-shiny/fieldpanel/R/formatters.R", "signature_region": false, "structural_support": 0.911, "target_lines": 12, "test_file": false, "unit": "formatWaveSummary", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 103, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 367, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -32,3 +32,15 @@
     labels[["targetN"]],
     formatCells(summary$targetN, "count")$text
   )
+  lines[["completes"]] <- sprintf(
+    "%s: %s",
+    labels[["completes"]],
+    formatCells(summary$completes, "count")$text
+  )
+  lines[["responseRate"]] <- sprintf(
+    "%s: %s",
+    labels[["responseRate"]],
+    formatCells(summary$responseRate, "rate")$text
+  )
+  return(lines)
+}
```

**Target (gold region)**

```r
  lines[["completes"]] <- sprintf(
    "%s: %s",
    labels[["completes"]],
    formatCells(summary$completes, "count")$text
  )
  lines[["responseRate"]] <- sprintf(
    "%s: %s",
    labels[["responseRate"]],
    formatCells(summary$responseRate, "rate")$text
  )
  return(lines)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/formatters.R
formatWaveSummary <- function(summary) {
  labels <- c(
    waveId = "Wave",
    startDate = "Start",
    endDate = "End",
    targetN = "Target",
    completes = "Completes",
    responseRate = "Response rate"
  )
  lines <- character(0)

  # Identifier and dates
  lines[["waveId"]] <- sprintf(
    "%s: %s",
    labels[["waveId"]],
    formatCells(summary$waveId, "text")$text
  )
  lines[["startDate"]] <- sprintf(
    "%s: %s",
    labels[["startDate"]],
    formatCells(summary$startDate, "date")$text
  )
  lines[["endDate"]] <- sprintf(
    "%s: %s",
    labels[["endDate"]],
    formatCells(summary$endDate, "date")$text
  )

  # Counts and rates
  lines[["targetN"]] <- sprintf(
    "%s: %s",
    labels[["targetN"]],
    formatCells(summary$targetN, "count")$text
  )
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 26. `7d160b90d0b718968bea0009` finish_block / sibling:medium

- package `synth:scalekit`, path `R/reverse.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["attr"], "novelty": 1, "rule": "fb_statement", "seed": "b48-psych/scalekit/R/reverse.R", "signature_region": false, "structural_support": 0.857, "target_lines": 17, "test_file": false, "unit": "apply_keys", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 127, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 649, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -67,3 +67,20 @@
   # Reverse `binary` items
   if (length(binary) > 0) {
     bounds <- ranges[["binary"]]
+    for (item in binary) {
+      if (
+        any(data[[item]] < bounds[1] | data[[item]] > bounds[2], na.rm = TRUE)
+      ) {
+        stop(
+          sprintf("Item `%s` is outside the `%s` range.", item, "binary"),
+          call. = FALSE
+        )
+      }
+      data[[item]] <- sum(bounds) - data[[item]]
+    }
+    reversed <- c(reversed, binary)
+  }
+
+  attr(data, "reversed") <- reversed
+  return(data)
+}
```

**Target (gold region)**

```r
    for (item in binary) {
      if (
        any(data[[item]] < bounds[1] | data[[item]] > bounds[2], na.rm = TRUE)
      ) {
        stop(
          sprintf("Item `%s` is outside the `%s` range.", item, "binary"),
          call. = FALSE
        )
      }
      data[[item]] <- sum(bounds) - data[[item]]
    }
    reversed <- c(reversed, binary)
  }

  attr(data, "reversed") <- reversed
  return(data)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/reverse.R
apply_keys <- function(
  data,
  agree5 = character(),
  agree7 = character(),
  freq4 = character(),
  binary = character()
) {
  ranges <- list(
    agree5 = c(1, 5),
    agree7 = c(1, 7),
    freq4 = c(0, 3),
    binary = c(0, 1)
  )
  reversed <- character()

  # Reverse `agree5` items
  if (length(agree5) > 0) {
    bounds <- ranges[["agree5"]]
    for (item in agree5) {
      if (
        any(data[[item]] < bounds[1] | data[[item]] > bounds[2], na.rm = TRUE)
      ) {
        stop(
          sprintf("Item `%s` is outside the `%s` range.", item, "agree5"),
          call. = FALSE
        )
      }
      data[[item]] <- sum(bounds) - data[[item]]
    }
    reversed <- c(reversed, agree5)
  }

  # Reverse `agree7` items
  if (length(agree7) > 0) {
    bounds <- ranges[["agree7"]]
    for (item in agree7) {
      if (
        any(data[[item]] < bounds[1] | data[[item]] > bounds[2], na.rm = TRUE)
      ) {
        stop(
          sprintf("Item `%s` is outside the `%s` range.", item, "agree7"),
          call. = FALSE
        )
      }
      data[[item]] <- sum(bounds) - data[[item]]
    }
    reversed <- c(reversed, agree7)
  }

  # Reverse `freq4` items
  if (length(freq4) > 0) {
    bounds <- ranges[["freq4"]]
    for (item in freq4) {
      if (
        any(data[[item]] < bounds[1] | data[[item]] > bounds[2], na.rm = TRUE)
      ) {
        stop(
          sprintf("Item `%s` is outside the `%s` range.", item, "freq4"),
          call. = FALSE
        )
      }
      data[[item]] <- sum(bounds) - data[[item]]
    }
    reversed <- c(reversed, freq4)
  }

  # Reverse `binary` items
  if (length(binary) > 0) {
    bounds <- ranges[["binary"]]
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 27. `2b12b58deb802835b6daa02c` finish_block / sibling:medium

- package `synth:dosewise`, path `R/patient.R`
- generator: `{"behaviour": "file_checked_not_entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["num:2.2", "num:2.7", "num:48"], "novelty": 3, "rule": "fb_nested_close", "seed": "b29-s4/dosewise/R/patient.R", "signature_region": false, "structural_support": 0.905, "target_lines": 11, "test_file": false, "unit": "ideal.body.weight", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 56, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 488, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -37,3 +37,14 @@
         56.2 + 1.41 * inches.over
       } else {
         53.1 + 1.36 * inches.over
+      }
+    },
+    hamwi = {
+      if (male) {
+        48 + 2.7 * inches.over
+      } else {
+        45.5 + 2.2 * inches.over
+      }
+    }
+  )
+}
```

**Target (gold region)**

```r
      }
    },
    hamwi = {
      if (male) {
        48 + 2.7 * inches.over
      } else {
        45.5 + 2.2 * inches.over
      }
    }
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/patient.R
#' Ideal body weight
#'
#' @param x A `Patient` object.
#' @param method Formula to use.
#' @details All formulas add a fixed weight per inch of height above five
#'   feet to a base weight, with separate constants for men and women:
#'   Devine 50 + 2.3 / 45.5 + 2.3, Robinson 52 + 1.9 / 49 + 1.7, Miller
#'   56.2 + 1.41 / 53.1 + 1.36 and Hamwi 48 + 2.7 / 45.5 + 2.2 (male / female,
#'   kg). Patients shorter than five feet get the base weight.
#' @return Ideal body weight in kg.
#' @export
ideal.body.weight <- function(
  x,
  method = c("devine", "robinson", "miller", "hamwi")
) {
  method <- match.arg(method)
  inches.over <- max(x@height.cm / 2.54 - 60, 0)
  male <- x@sex == "male"
  switch(
    method,
    devine = {
      if (male) {
        50 + 2.3 * inches.over
      } else {
        45.5 + 2.3 * inches.over
      }
    },
    robinson = {
      if (male) {
        52 + 1.9 * inches.over
      } else {
        49 + 1.7 * inches.over
      }
    },
    miller = {
      if (male) {
        56.2 + 1.41 * inches.over
      } else {
        53.1 + 1.36 * inches.over
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 28. `7749b1b8cf1dcb1d3bbc999c` finish_block / sibling:medium

- package `synth:queuewright`, path `R/jobs_input.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b47-ops/queuewright/R/jobs_input.R", "signature_region": false, "structural_support": 0.868, "target_lines": 10, "test_file": false, "unit": "validate_jobs", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 81, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 681, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -63,3 +63,13 @@
   }
   # weight
   if (!is.numeric(jobs$weight) || anyNA(jobs$weight)) {
+    stop(
+      sprintf("Column `%s` must be numeric without missing values.", "weight"),
+      call. = FALSE
+    )
+  }
+  if (any(jobs$weight <= 0)) {
+    stop(sprintf("Column `%s` must be positive.", "weight"), call. = FALSE)
+  }
+  invisible(jobs)
+}
```

**Target (gold region)**

```r
    stop(
      sprintf("Column `%s` must be numeric without missing values.", "weight"),
      call. = FALSE
    )
  }
  if (any(jobs$weight <= 0)) {
    stop(sprintf("Column `%s` must be positive.", "weight"), call. = FALSE)
  }
  invisible(jobs)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/jobs_input.R
#' Validate a job table
#'
#' @param jobs A data frame with columns `id`, `release`, `proc_time`, `due`
#'   and `weight`.
#' @return `jobs`, invisibly, when every column passes its checks.
#' @keywords internal
validate_jobs <- function(jobs) {
  if (!is.data.frame(jobs)) {
    stop("`jobs` must be a data frame.", call. = FALSE)
  }
  required <- c("id", "release", "proc_time", "due", "weight")
  absent <- setdiff(required, names(jobs))
  if (length(absent) > 0) {
    stop(
      "`jobs` is missing columns: ",
      paste(absent, collapse = ", "),
      call. = FALSE
    )
  }

  # id
  if (anyNA(jobs$id) || anyDuplicated(jobs$id) > 0) {
    stop(
      sprintf("Column `%s` must be unique and not missing.", "id"),
      call. = FALSE
    )
  }
  # release
  if (!is.numeric(jobs$release) || anyNA(jobs$release)) {
    stop(
      sprintf("Column `%s` must be numeric without missing values.", "release"),
      call. = FALSE
    )
  }
  if (any(jobs$release < 0)) {
    stop(sprintf("Column `%s` must be non-negative.", "release"), call. = FALSE)
  }
  # proc_time
  if (!is.numeric(jobs$proc_time) || anyNA(jobs$proc_time)) {
    stop(
      sprintf(
        "Column `%s` must be numeric without missing values.",
        "proc_time"
      ),
      call. = FALSE
    )
  }
  if (any(jobs$proc_time <= 0)) {
    stop(sprintf("Column `%s` must be positive.", "proc_time"), call. = FALSE)
  }
  # due
  if (!is.numeric(jobs$due) || anyNA(jobs$due)) {
    stop(
      sprintf("Column `%s` must be numeric without missing values.", "due"),
      call. = FALSE
    )
  }
  if (any(jobs$due < jobs$release)) {
    stop(
      sprintf("Column `%s` must not precede `%s`.", "due", "release"),
      call. = FALSE
    )
  }
  # weight
  if (!is.numeric(jobs$weight) || anyNA(jobs$weight)) {
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 29. `af52b445d1325440db720655` finish_block / sibling:medium

- package `synth:rowcast`, path `R/validate.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["rownames", "toupper"], "novelty": 2, "rule": "fb_nested_close", "seed": "b45-sql/rowcast/R/validate.R", "signature_region": false, "structural_support": 0.59, "target_lines": 12, "test_file": false, "unit": "check.identifiers", "variant": "outline"}`
- verification: `{"air_idempotent": true, "completion_tokens": 80, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 619, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -61,3 +61,15 @@
       failures,
       data.frame(name = names[bad], rule = "lower_case")
     )
+  }
+  # Rule `not_reserved`
+  bad <- toupper(names) %in% reserved
+  if (any(bad)) {
+    failures <- rbind(
+      failures,
+      data.frame(name = names[bad], rule = "not_reserved")
+    )
+  }
+  rownames(failures) <- NULL
+  failures
+}
```

**Target (gold region)**

```r
  }
  # Rule `not_reserved`
  bad <- toupper(names) %in% reserved
  if (any(bad)) {
    failures <- rbind(
      failures,
      data.frame(name = names[bad], rule = "not_reserved")
    )
  }
  rownames(failures) <- NULL
  failures
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/validate.R
# ---- outline --------------------------------
# fn: check.identifiers(names, dialect, reserved)
# plan: 15 top-level statements; 10 typed above the cursor
# docs: yes
# ----------------------------------------------
#' Check identifiers against naming rules
#'
#' Applies each naming rule to every identifier and reports the failures.
#'
#' @param names A character vector of identifiers.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`; sets the length limit.
#' @param reserved Upper-case reserved words that must not be used bare.
#' @return A data frame with columns `name` and `rule`, one row per failure.
#' @export
check.identifiers <- function(
  names,
  dialect = c("postgres", "mysql", "sqlite"),
  reserved = c(
    "SELECT",
    "FROM",
    "WHERE",
    "GROUP",
    "ORDER",
    "USER",
    "TABLE",
    "INDEX",
    "KEY"
  )
) {
  dialect <- match.arg(dialect)
  max.length <- c(postgres = 63L, mysql = 64L, sqlite = 1024L)[[dialect]]
  failures <- data.frame(name = character(), rule = character())
  # Rule `max_length`
  bad <- nchar(names) > max.length
  if (any(bad)) {
    failures <- rbind(
      failures,
      data.frame(name = names[bad], rule = "max_length")
    )
  }
  # Rule `leading_letter`
  bad <- !grepl("^[A-Za-z_]", names)
  if (any(bad)) {
    failures <- rbind(
      failures,
      data.frame(name = names[bad], rule = "leading_letter")
    )
  }
  # Rule `plain_characters`
  bad <- grepl("[^A-Za-z0-9_]", names)
  if (any(bad)) {
    failures <- rbind(
      failures,
      data.frame(name = names[bad], rule = "plain_characters")
    )
  }
  # Rule `lower_case`
  bad <- names != tolower(names)
  if (any(bad)) {
    failures <- rbind(
      failures,
      data.frame(name = names[bad], rule = "lower_case")
    )
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 30. `10179c9b50cbf621603eb6d8` finish_block / sibling:medium

- package `synth:shiftloom`, path `R/coverage.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["attr"], "novelty": 1, "rule": "fb_comment_lead", "seed": "b47-ops/shiftloom/R/coverage.R", "signature_region": false, "structural_support": 0.807, "target_lines": 11, "test_file": false, "unit": "fillGaps", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 100, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 422, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -25,3 +25,14 @@
       roster[who, d] <- "L"
       added <- added + length(who)
     }
+    # Night shift, any spare staff member may take it
+    missing <- night - sum(roster[, d] == "N")
+    if (missing > 0) {
+      who <- pickSpare(d, missing, blocked = character(0))
+      roster[who, d] <- "N"
+      added <- added + length(who)
+    }
+  }
+  attr(roster, "added") <- added
+  roster
+}
```

**Target (gold region)**

```r
    # Night shift, any spare staff member may take it
    missing <- night - sum(roster[, d] == "N")
    if (missing > 0) {
      who <- pickSpare(d, missing, blocked = character(0))
      roster[who, d] <- "N"
      added <- added + length(who)
    }
  }
  attr(roster, "added") <- added
  roster
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/coverage.R
fillGaps <- function(roster, early = 2, late = 2, night = 1, maxShifts = Inf) {
  added <- 0
  pickSpare <- function(day, needed, blocked) {
    spare <- roster[, day] == "O" & rowSums(roster != "O") < maxShifts
    if (day > 1) {
      spare <- spare & !(roster[, day - 1] %in% blocked)
    }
    candidates <- which(spare)
    candidates <- candidates[order(rowSums(roster != "O")[candidates])]
    utils::head(candidates, needed)
  }

  for (d in seq_len(ncol(roster))) {
    # Early shift, never straight after a night
    missing <- early - sum(roster[, d] == "E")
    if (missing > 0) {
      who <- pickSpare(d, missing, blocked = "N")
      roster[who, d] <- "E"
      added <- added + length(who)
    }
    # Late shift, any spare staff member may take it
    missing <- late - sum(roster[, d] == "L")
    if (missing > 0) {
      who <- pickSpare(d, missing, blocked = character(0))
      roster[who, d] <- "L"
      added <- added + length(who)
    }
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 31. `10f0bd46f7e65f25e973d7a8` finish_block / sibling:medium

- package `synth:cellfmt`, path `R/numbers.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b26-reporting/cellfmt/R/numbers.R", "signature_region": false, "structural_support": 0.541, "target_lines": 9, "test_file": false, "unit": "fmt_signed", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 84, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 225, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -8,3 +8,12 @@
     out[pos] <- paste0(plus, body)
   }
 
+  if (any(neg)) {
+    body <- formatC(abs(x[neg]), format = "f", digits = digits)
+    out[neg] <- paste0(minus, body)
+  }
+
+  out[!is.na(x) & x == 0] <- zero
+  out[is.na(x)] <- NA_character_
+  return(out)
+}
```

**Target (gold region)**

```r
  if (any(neg)) {
    body <- formatC(abs(x[neg]), format = "f", digits = digits)
    out[neg] <- paste0(minus, body)
  }

  out[!is.na(x) & x == 0] <- zero
  out[is.na(x)] <- NA_character_
  return(out)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/numbers.R
fmt_signed <- function(x, digits = 2, plus = "+", minus = "-", zero = "0") {
  check_number_args(x, digits, ",", zero)
  out <- character(length(x))
  pos <- !is.na(x) & x > 0
  neg <- !is.na(x) & x < 0
  if (any(pos)) {
    body <- formatC(abs(x[pos]), format = "f", digits = digits)
    out[pos] <- paste0(plus, body)
  }

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 32. `2c342923dd0a073c09da33fd` finish_block / sibling:medium

- package `synth:calendrix`, path `R/boundaries.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["num:6L"], "novelty": 1, "rule": "fb_statement", "seed": "b32-dates/calendrix/R/boundaries.R", "signature_region": false, "structural_support": 0.667, "target_lines": 12, "test_file": false, "unit": "start_of_period", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 108, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 351, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -26,3 +26,15 @@
     },
     quarter = {
       first_mon <- (mon - 1L) %/% 3L * 3L + 1L
+      as.Date(sprintf("%04d-%02d-01", year, first_mon))
+    },
+    half = {
+      first_mon <- (mon - 1L) %/% 6L * 6L + 1L
+      as.Date(sprintf("%04d-%02d-01", year, first_mon))
+    },
+    year = {
+      as.Date(sprintf("%04d-01-01", year))
+    }
+  )
+  return(out)
+}
```

**Target (gold region)**

```r
      as.Date(sprintf("%04d-%02d-01", year, first_mon))
    },
    half = {
      first_mon <- (mon - 1L) %/% 6L * 6L + 1L
      as.Date(sprintf("%04d-%02d-01", year, first_mon))
    },
    year = {
      as.Date(sprintf("%04d-01-01", year))
    }
  )
  return(out)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/boundaries.R
start_of_period <- function(
  x,
  unit = c("week", "month", "quarter", "half", "year"),
  week_start = 1L
) {
  unit <- match.arg(unit)
  if (length(week_start) != 1L || !(week_start %in% 1:7)) {
    stop(
      "`week_start` must be a single weekday number from 1 to 7.",
      call. = FALSE
    )
  }
  x <- as.Date(x)
  lt <- as.POSIXlt(x)
  year <- lt$year + 1900L
  mon <- lt$mon + 1L

  out <- switch(
    unit,
    week = {
      wday <- as.integer(format(x, "%u"))
      x - (wday - week_start) %% 7L
    },
    month = {
      as.Date(sprintf("%04d-%02d-01", year, mon))
    },
    quarter = {
      first_mon <- (mon - 1L) %/% 3L * 3L + 1L
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 33. `a57ab9c282b7004aa76f00e7` finish_block / sibling:medium

- package `synth:queuewright`, path `R/capacity.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["storage_util"], "novelty": 1, "rule": "fb_statement", "seed": "b47-ops/queuewright/R/capacity.R", "signature_region": false, "structural_support": 0.617, "target_lines": 20, "test_file": false, "unit": "check_capacity", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 139, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 726, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -71,3 +71,23 @@
   storage_load <- tapply(plan$storage_m3, plan$period, sum)[as.character(
     periods
   )]
+  storage_util <- unname(storage_load) / rep_len(storage_m3, n)
+  if (any(storage_util > 1)) {
+    warning(
+      sprintf(
+        "`%s` over capacity in %d period(s).",
+        "storage_m3",
+        sum(storage_util > 1)
+      ),
+      call. = FALSE
+    )
+  }
+
+  data.frame(
+    period = periods,
+    machine_util = machine_util,
+    labour_util = labour_util,
+    tooling_util = tooling_util,
+    storage_util = storage_util
+  )
+}
```

**Target (gold region)**

```r
  storage_util <- unname(storage_load) / rep_len(storage_m3, n)
  if (any(storage_util > 1)) {
    warning(
      sprintf(
        "`%s` over capacity in %d period(s).",
        "storage_m3",
        sum(storage_util > 1)
      ),
      call. = FALSE
    )
  }

  data.frame(
    period = periods,
    machine_util = machine_util,
    labour_util = labour_util,
    tooling_util = tooling_util,
    storage_util = storage_util
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/capacity.R
# ---- diagnostics (static) --------------------
# typed statements: 12 of 15
# args not yet referenced: none
# unclosed braces in typed code: 1
# ----------------------------------------------
#' Check a production plan against per-resource capacity
#'
#' @param plan A data frame with columns `period`, `machine_hours`,
#'   `labour_hours`, `tooling_sets` and `storage_m3`.
#' @param machine_hours,labour_hours,tooling_sets,storage_m3 Capacity of each
#'   resource per period, recycled over periods.
#' @return A data frame with one row per period and a utilisation column per
#'   resource; warns for every resource that is over capacity.
#' @export
check_capacity <- function(
  plan,
  machine_hours,
  labour_hours,
  tooling_sets,
  storage_m3
) {
  periods <- sort(unique(plan$period))
  n <- length(periods)

  # Machine hours
  machine_load <- tapply(plan$machine_hours, plan$period, sum)[as.character(
    periods
  )]
  machine_util <- unname(machine_load) / rep_len(machine_hours, n)
  if (any(machine_util > 1)) {
    warning(
      sprintf(
        "`%s` over capacity in %d period(s).",
        "machine_hours",
        sum(machine_util > 1)
      ),
      call. = FALSE
    )
  }
  # Labour hours
  labour_load <- tapply(plan$labour_hours, plan$period, sum)[as.character(
    periods
  )]
  labour_util <- unname(labour_load) / rep_len(labour_hours, n)
  if (any(labour_util > 1)) {
    warning(
      sprintf(
        "`%s` over capacity in %d period(s).",
        "labour_hours",
        sum(labour_util > 1)
      ),
      call. = FALSE
    )
  }
  # Tooling sets
  tooling_load <- tapply(plan$tooling_sets, plan$period, sum)[as.character(
    periods
  )]
  tooling_util <- unname(tooling_load) / rep_len(tooling_sets, n)
  if (any(tooling_util > 1)) {
    warning(
      sprintf(
        "`%s` over capacity in %d period(s).",
        "tooling_sets",
        sum(tooling_util > 1)
      ),
      call. = FALSE
    )
  }
  # Storage volume
  storage_load <- tapply(plan$storage_m3, plan$period, sum)[as.character(
    periods
  )]
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 34. `12714fa70775e9a69173969e` finish_block / sibling:medium

- package `synth:gradeledger`, path `R/late-work.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["pmax"], "novelty": 1, "rule": "fb_statement", "seed": "b44-analytics/gradeledger/R/late-work.R", "signature_region": false, "structural_support": 0.571, "target_lines": 16, "test_file": false, "unit": "classifySubmissions", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 128, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 471, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -31,3 +31,19 @@
   # Status `onTime`
   hit <- is.na(status) & minutesLate <= 0
   status[hit] <- "onTime"
+  counts[["onTime"]] <- sum(hit)
+  # Status `grace`
+  hit <- is.na(status) & minutesLate <= graceMinutes
+  status[hit] <- "grace"
+  counts[["grace"]] <- sum(hit)
+  # Status `late`
+  hit <- is.na(status)
+  status[hit] <- "late"
+  counts[["late"]] <- sum(hit)
+
+  return(list(
+    status = status,
+    minutesLate = pmax(minutesLate, 0),
+    counts = counts
+  ))
+}
```

**Target (gold region)**

```r
  counts[["onTime"]] <- sum(hit)
  # Status `grace`
  hit <- is.na(status) & minutesLate <= graceMinutes
  status[hit] <- "grace"
  counts[["grace"]] <- sum(hit)
  # Status `late`
  hit <- is.na(status)
  status[hit] <- "late"
  counts[["late"]] <- sum(hit)

  return(list(
    status = status,
    minutesLate = pmax(minutesLate, 0),
    counts = counts
  ))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/late-work.R
#' Classify submissions by timeliness
#'
#' Statuses are assigned in the order `excused`, `missing`, `onTime`, `grace`
#' and `late`; the first that applies wins.
#'
#' @param due,submitted Deadlines and submission times as date-times;
#'   `submitted` is `NA` when nothing was handed in.
#' @param excused Logical, whether the work was excused.
#' @param graceMinutes Minutes after the deadline still counted as `grace`.
#' @return A list with elements `status`, `minutesLate` and `counts`.
#' @export
classifySubmissions <- function(
  due,
  submitted,
  excused = FALSE,
  graceMinutes = 15
) {
  excused <- rep_len(excused, length(due))
  minutesLate <- as.numeric(difftime(submitted, due, units = "mins"))
  status <- rep(NA_character_, length(due))
  counts <- c(excused = 0L, missing = 0L, onTime = 0L, grace = 0L, late = 0L)

  # Status `excused`
  hit <- is.na(status) & excused
  status[hit] <- "excused"
  counts[["excused"]] <- sum(hit)
  # Status `missing`
  hit <- is.na(status) & is.na(submitted)
  status[hit] <- "missing"
  counts[["missing"]] <- sum(hit)
  # Status `onTime`
  hit <- is.na(status) & minutesLate <= 0
  status[hit] <- "onTime"
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 35. `1f93cdd70fe1e397a283150b` finish_block / sibling:medium

- package `synth:tidyscrub`, path `R/normalize_text.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b25-validation/tidyscrub/R/normalize_text.R", "signature_region": false, "structural_support": 0.514, "target_lines": 9, "test_file": false, "unit": "strip.affix", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 91, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 213, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -4,3 +4,12 @@
     n <- nchar(prefix)
     hit <- !is.na(x) & fold(substr(x, 1L, n)) == fold(prefix)
     x[hit] <- substr(x[hit], n + 1L, nchar(x[hit]))
+  }
+  if (!is.null(suffix)) {
+    n <- nchar(suffix)
+    hit <- !is.na(x) &
+      fold(substr(x, nchar(x) - n + 1L, nchar(x))) == fold(suffix)
+    x[hit] <- substr(x[hit], 1L, nchar(x[hit]) - n)
+  }
+  x
+}
```

**Target (gold region)**

```r
  }
  if (!is.null(suffix)) {
    n <- nchar(suffix)
    hit <- !is.na(x) &
      fold(substr(x, nchar(x) - n + 1L, nchar(x))) == fold(suffix)
    x[hit] <- substr(x[hit], 1L, nchar(x[hit]) - n)
  }
  x
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/normalize_text.R
strip.affix <- function(x, prefix = NULL, suffix = NULL, ignore.case = FALSE) {
  fold <- if (ignore.case) tolower else identity
  if (!is.null(prefix)) {
    n <- nchar(prefix)
    hit <- !is.na(x) & fold(substr(x, 1L, n)) == fold(prefix)
    x[hit] <- substr(x[hit], n + 1L, nchar(x[hit]))
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 36. `fbb53072bfaecc7a2a0d713a` finish_block / sibling:medium

- package `synth:presentia`, path `R/attendance-summary.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["attended.rate", "excused.rate", "left.rate"], "novelty": 3, "rule": "fb_statement", "seed": "b44-analytics/presentia/R/attendance-summary.R", "signature_region": false, "structural_support": 0.701, "target_lines": 14, "test_file": false, "unit": "student.summary", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 136, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 476, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,3 +28,17 @@
   out$absent <- n
   out$absent.rate <- round(n / sessions, digits)
   # Status `excused`
+  n <- as.vector(tapply(records$status == "excused", s, sum))
+  out$excused <- n
+  out$excused.rate <- round(n / sessions, digits)
+  # Status `left`
+  n <- as.vector(tapply(records$status == "left", s, sum))
+  out$left <- n
+  out$left.rate <- round(n / sessions, digits)
+
+  out$attended.rate <- round(
+    (out$present + out$late + out$left) / sessions,
+    digits
+  )
+  out
+}
```

**Target (gold region)**

```r
  n <- as.vector(tapply(records$status == "excused", s, sum))
  out$excused <- n
  out$excused.rate <- round(n / sessions, digits)
  # Status `left`
  n <- as.vector(tapply(records$status == "left", s, sum))
  out$left <- n
  out$left.rate <- round(n / sessions, digits)

  out$attended.rate <- round(
    (out$present + out$late + out$left) / sessions,
    digits
  )
  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/attendance-summary.R
#' Attendance counts per student
#'
#' One block per status `present`, `late`, `absent`, `excused` and `left`; a
#' session counts as attended when the status is `present`, `late` or `left`.
#'
#' @param records A data frame with columns `student`, `date`, `session` and
#'   `status`, one row per student and session.
#' @param digits Number of decimal places kept in rates.
#' @return A data frame with columns `student` and `sessions`, a count and a
#'   `.rate` column per status, and `attended.rate`.
#' @export
student.summary <- function(records, digits = 3) {
  students <- sort(unique(records$student))
  s <- factor(records$student, levels = students)
  sessions <- as.vector(table(s))
  out <- data.frame(student = students, sessions = sessions)

  # Status `present`
  n <- as.vector(tapply(records$status == "present", s, sum))
  out$present <- n
  out$present.rate <- round(n / sessions, digits)
  # Status `late`
  n <- as.vector(tapply(records$status == "late", s, sum))
  out$late <- n
  out$late.rate <- round(n / sessions, digits)
  # Status `absent`
  n <- as.vector(tapply(records$status == "absent", s, sum))
  out$absent <- n
  out$absent.rate <- round(n / sessions, digits)
  # Status `excused`
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 37. `497e33eff782c55815f037f5` finish_block / sibling:medium

- package `synth:geoprism`, path `R/distance.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["ifelse", "num:1e-12"], "novelty": 2, "rule": "fb_statement", "seed": "b42-geo/geoprism/R/distance.R", "signature_region": false, "structural_support": 0.556, "target_lines": 13, "test_file": false, "unit": "geo_destination", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 180, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 495, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -35,3 +35,16 @@
     rhumb = {
       phi2 <- phi1 + delta * cos(theta)
       stretch <- log(tan(pi / 4 + phi2 / 2) / tan(pi / 4 + phi1 / 2))
+      q <- ifelse(abs(stretch) > 1e-12, (phi2 - phi1) / stretch, cos(phi1))
+      lam2 <- lam1 + delta * sin(theta) / q
+      lon2 <- ((lam2 * 180 / pi + 540) %% 360) - 180
+      data.frame(lon = lon2, lat = phi2 * 180 / pi)
+    },
+    flat = {
+      phi2 <- phi1 + delta * cos(theta)
+      lam2 <- lam1 + delta * sin(theta) / cos(phi1)
+      lon2 <- ((lam2 * 180 / pi + 540) %% 360) - 180
+      data.frame(lon = lon2, lat = phi2 * 180 / pi)
+    }
+  )
+}
```

**Target (gold region)**

```r
      q <- ifelse(abs(stretch) > 1e-12, (phi2 - phi1) / stretch, cos(phi1))
      lam2 <- lam1 + delta * sin(theta) / q
      lon2 <- ((lam2 * 180 / pi + 540) %% 360) - 180
      data.frame(lon = lon2, lat = phi2 * 180 / pi)
    },
    flat = {
      phi2 <- phi1 + delta * cos(theta)
      lam2 <- lam1 + delta * sin(theta) / cos(phi1)
      lon2 <- ((lam2 * 180 / pi + 540) %% 360) - 180
      data.frame(lon = lon2, lat = phi2 * 180 / pi)
    }
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/distance.R
#' Point reached from a start point, bearing and distance
#'
#' @param lon,lat Coordinates of the start points in degrees.
#' @param bearing Bearing in degrees clockwise from north.
#' @param distance Distance in metres.
#' @param method One of `"great_circle"`, `"rhumb"` or `"flat"`.
#' @param radius Sphere radius in metres.
#' @return A data frame with columns `lon` and `lat` in degrees.
#' @export
geo_destination <- function(
  lon,
  lat,
  bearing,
  distance,
  method = c("great_circle", "rhumb", "flat"),
  radius = 6371008.8
) {
  method <- match.arg(method)
  phi1 <- lat * pi / 180
  lam1 <- lon * pi / 180
  theta <- bearing * pi / 180
  delta <- distance / radius
  switch(
    method,
    great_circle = {
      phi2 <- asin(sin(phi1) * cos(delta) + cos(phi1) * sin(delta) * cos(theta))
      lam2 <- lam1 +
        atan2(
          sin(theta) * sin(delta) * cos(phi1),
          cos(delta) - sin(phi1) * sin(phi2)
        )
      lon2 <- ((lam2 * 180 / pi + 540) %% 360) - 180
      data.frame(lon = lon2, lat = phi2 * 180 / pi)
    },
    rhumb = {
      phi2 <- phi1 + delta * cos(theta)
      stretch <- log(tan(pi / 4 + phi2 / 2) / tan(pi / 4 + phi1 / 2))
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 38. `ce3f7d94479a05ac3550b5ca` finish_block / sibling:medium

- package `synth:dosewise`, path `R/regimen.R`
- generator: `{"behaviour": "file_checked_not_entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["n.doses"], "novelty": 1, "rule": "fb_nested_close", "seed": "b29-s4/dosewise/R/regimen.R", "signature_region": false, "structural_support": 0.86, "target_lines": 9, "test_file": false, "unit": "set.regimen.n.doses", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 82, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 533, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -38,3 +38,12 @@
 set.regimen.n.doses <- function(x, value) {
   if (!is.numeric(value) && !is.character(value)) {
     stop(sprintf("`%s` must be numeric.", "n.doses"), call. = FALSE)
+  }
+  value <- suppressWarnings(as.integer(value))
+  if (length(value) != 1L || is.na(value)) {
+    stop(sprintf("`%s` must be a single value.", "n.doses"), call. = FALSE)
+  }
+  x@n.doses <- value
+  validObject(x)
+  x
+}
```

**Target (gold region)**

```r
  }
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value)) {
    stop(sprintf("`%s` must be a single value.", "n.doses"), call. = FALSE)
  }
  x@n.doses <- value
  validObject(x)
  x
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/regimen.R
#' Regimen setters
#'
#' @param x A `Regimen` object.
#' @param value The new value.
#' @details These back the replacement methods `amount<-`, `interval<-` and
#'   `n.doses<-`. Each accepts a number or a string holding one, coerces
#'   `value`, stores it and revalidates `x`.
#' @return The modified `Regimen` object.
#' @keywords internal
set.regimen.amount <- function(x, value) {
  if (!is.numeric(value) && !is.character(value)) {
    stop(sprintf("`%s` must be numeric.", "amount"), call. = FALSE)
  }
  value <- suppressWarnings(as.numeric(value))
  if (length(value) != 1L || is.na(value)) {
    stop(sprintf("`%s` must be a single value.", "amount"), call. = FALSE)
  }
  x@amount <- value
  validObject(x)
  x
}

#' @rdname set.regimen.amount
set.regimen.interval <- function(x, value) {
  if (!is.numeric(value) && !is.character(value)) {
    stop(sprintf("`%s` must be numeric.", "interval.h"), call. = FALSE)
  }
  value <- suppressWarnings(as.numeric(value))
  if (length(value) != 1L || is.na(value)) {
    stop(sprintf("`%s` must be a single value.", "interval.h"), call. = FALSE)
  }
  x@interval.h <- value
  validObject(x)
  x
}

#' @rdname set.regimen.amount
set.regimen.n.doses <- function(x, value) {
  if (!is.numeric(value) && !is.character(value)) {
    stop(sprintf("`%s` must be numeric.", "n.doses"), call. = FALSE)
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 39. `6ef918735c1e3cf7f6361adc` finish_block / sibling:medium

- package `synth:fieldsieve`, path `R/extract.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["dates", "hashtags", "phones"], "novelty": 3, "rule": "fb_statement", "seed": "b31-text/fieldsieve/R/extract.R", "signature_region": false, "structural_support": 0.959, "target_lines": 19, "test_file": false, "unit": "first_entities", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 156, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 258, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -18,3 +18,22 @@
 
   urls <- extract_entities(x, "url")[[1]]
   if (length(urls) > 0L) {
+    out[["url"]] <- urls[[1]]
+  }
+
+  phones <- extract_entities(x, "phone")[[1]]
+  if (length(phones) > 0L) {
+    out[["phone"]] <- phones[[1]]
+  }
+
+  hashtags <- extract_entities(x, "hashtag")[[1]]
+  if (length(hashtags) > 0L) {
+    out[["hashtag"]] <- hashtags[[1]]
+  }
+
+  dates <- extract_entities(x, "date")[[1]]
+  if (length(dates) > 0L) {
+    out[["date"]] <- dates[[1]]
+  }
+  return(out)
+}
```

**Target (gold region)**

```r
    out[["url"]] <- urls[[1]]
  }

  phones <- extract_entities(x, "phone")[[1]]
  if (length(phones) > 0L) {
    out[["phone"]] <- phones[[1]]
  }

  hashtags <- extract_entities(x, "hashtag")[[1]]
  if (length(hashtags) > 0L) {
    out[["hashtag"]] <- hashtags[[1]]
  }

  dates <- extract_entities(x, "date")[[1]]
  if (length(dates) > 0L) {
    out[["date"]] <- dates[[1]]
  }
  return(out)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/extract.R
first_entities <- function(x, default = NA_character_) {
  stopifnot(is.character(x), length(x) == 1L)
  out <- c(
    email = default,
    url = default,
    phone = default,
    hashtag = default,
    date = default
  )
  if (is.na(x)) {
    return(out)
  }

  emails <- extract_entities(x, "email")[[1]]
  if (length(emails) > 0L) {
    out[["email"]] <- emails[[1]]
  }

  urls <- extract_entities(x, "url")[[1]]
  if (length(urls) > 0L) {
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 40. `1c031b0b966405144f66ca5b` finish_block / sibling:medium

- package `synth:leaguetab`, path `R/standings.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["goals_against", "goals_against_pg", "points_pg"], "novelty": 3, "rule": "fb_comment_lead", "seed": "b44-analytics/leaguetab/R/standings.R", "signature_region": false, "structural_support": 0.951, "target_lines": 11, "test_file": false, "unit": "standings_per_game", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 97, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 511, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -34,3 +34,14 @@
   rate <- table$goals_for / played
   rate[idle] <- NA_real_
   out$goals_for_pg <- round(rate, digits)
+  # Per game: `goals_against`
+  rate <- table$goals_against / played
+  rate[idle] <- NA_real_
+  out$goals_against_pg <- round(rate, digits)
+  # Per game: `points`
+  rate <- table$points / played
+  rate[idle] <- NA_real_
+  out$points_pg <- round(rate, digits)
+
+  out
+}
```

**Target (gold region)**

```r
  # Per game: `goals_against`
  rate <- table$goals_against / played
  rate[idle] <- NA_real_
  out$goals_against_pg <- round(rate, digits)
  # Per game: `points`
  rate <- table$points / played
  rate[idle] <- NA_real_
  out$points_pg <- round(rate, digits)

  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/standings.R
# ---- diagnostics (static) --------------------
# typed statements: 15 of 22
# args not yet referenced: none
# unclosed braces in typed code: 1
# ----------------------------------------------
#' Rates per game for each statistic column
#'
#' Divides `won`, `drawn`, `lost`, `goals_for`, `goals_against` and `points`
#' by games played; teams without a game get `NA`.
#'
#' @param table A league table as returned by [standings_table()].
#' @param digits Number of decimal places to keep.
#' @return A data frame with columns `team`, `played`, `won_pg`, `drawn_pg`,
#'   `lost_pg`, `goals_for_pg`, `goals_against_pg` and `points_pg`.
#' @export
standings_per_game <- function(table, digits = 2) {
  played <- pmax(table$played, 1L)
  idle <- table$played == 0L
  out <- data.frame(team = table$team, played = table$played)

  # Per game: `won`
  rate <- table$won / played
  rate[idle] <- NA_real_
  out$won_pg <- round(rate, digits)
  # Per game: `drawn`
  rate <- table$drawn / played
  rate[idle] <- NA_real_
  out$drawn_pg <- round(rate, digits)
  # Per game: `lost`
  rate <- table$lost / played
  rate[idle] <- NA_real_
  out$lost_pg <- round(rate, digits)
  # Per game: `goals_for`
  rate <- table$goals_for / played
  rate[idle] <- NA_real_
  out$goals_for_pg <- round(rate, digits)
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 41. `ca4d7aef7cc18938b4f64f8e` finish_block / sibling:long

- package `synth:leaguetab`, path `R/standings.R`
- generator: `{"behaviour": "entered", "bucket": "long", "determinacy": "sibling", "full_body": false, "novel_tokens": ["goals_against", "goals_for", "unname"], "novelty": 3, "rule": "fb_statement", "seed": "b44-analytics/leaguetab/R/standings.R", "signature_region": false, "structural_support": 0.777, "target_lines": 23, "test_file": false, "unit": "column_leaders", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 234, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 364, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -24,3 +24,26 @@
   value[["points"]] <- best
   # Most `won`
   best <- max(table$won)
+  team[["won"]] <- paste(table$team[table$won == best], collapse = "/")
+  value[["won"]] <- best
+  # Most `goals_for`
+  best <- max(table$goals_for)
+  team[["goals_for"]] <- paste(
+    table$team[table$goals_for == best],
+    collapse = "/"
+  )
+  value[["goals_for"]] <- best
+  # Fewest `lost`
+  best <- min(table$lost)
+  team[["lost"]] <- paste(table$team[table$lost == best], collapse = "/")
+  value[["lost"]] <- best
+  # Fewest `goals_against`
+  best <- min(table$goals_against)
+  team[["goals_against"]] <- paste(
+    table$team[table$goals_against == best],
+    collapse = "/"
+  )
+  value[["goals_against"]] <- best
+
+  data.frame(stat = stats, team = unname(team), value = unname(value))
+}
```

**Target (gold region)**

```r
  team[["won"]] <- paste(table$team[table$won == best], collapse = "/")
  value[["won"]] <- best
  # Most `goals_for`
  best <- max(table$goals_for)
  team[["goals_for"]] <- paste(
    table$team[table$goals_for == best],
    collapse = "/"
  )
  value[["goals_for"]] <- best
  # Fewest `lost`
  best <- min(table$lost)
  team[["lost"]] <- paste(table$team[table$lost == best], collapse = "/")
  value[["lost"]] <- best
  # Fewest `goals_against`
  best <- min(table$goals_against)
  team[["goals_against"]] <- paste(
    table$team[table$goals_against == best],
    collapse = "/"
  )
  value[["goals_against"]] <- best

  data.frame(stat = stats, team = unname(team), value = unname(value))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/standings.R
# ---- diagnostics (static) --------------------
# typed statements: 8 of 20
# args not yet referenced: none
# unclosed braces in typed code: 1
# ----------------------------------------------
#' Leaders for each statistic column
#'
#' Most `points`, `won` and `goals_for`; fewest `lost` and `goals_against`.
#' Ties are joined with a slash.
#'
#' @inheritParams standings_per_game
#' @return A data frame with columns `stat`, `team` and `value`, one row per
#'   statistic.
#' @export
column_leaders <- function(table) {
  stats <- c("points", "won", "goals_for", "lost", "goals_against")
  team <- character(length(stats))
  value <- numeric(length(stats))
  names(team) <- names(value) <- stats

  # Most `points`
  best <- max(table$points)
  team[["points"]] <- paste(table$team[table$points == best], collapse = "/")
  value[["points"]] <- best
  # Most `won`
  best <- max(table$won)
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 42. `131054a7f05dc28454d40e8a` finish_block / sibling:long

- package `synth:fluxunit`, path `R/indices-air.R`
- generator: `{"behaviour": "entered", "bucket": "long", "determinacy": "sibling", "full_body": false, "novel_tokens": ["comfort", "score_humidity", "score_wind"], "novelty": 3, "rule": "fb_comment_lead", "seed": "b40-science/fluxunit/R/indices-air.R", "signature_region": false, "structural_support": 0.604, "target_lines": 23, "test_file": false, "unit": "comfort_index", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 211, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 283, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -14,3 +14,26 @@
   gap <- pmax(lo - temp, temp - hi, 0)
   score_temp <- 1 - pmin(gap / tolerance[["temp"]], 1)
 
+  # Score `humidity`
+  lo <- optimum$humidity[1]
+  hi <- optimum$humidity[2]
+  gap <- pmax(lo - humidity, humidity - hi, 0)
+  score_humidity <- 1 - pmin(gap / tolerance[["humidity"]], 1)
+
+  # Score `wind`
+  lo <- optimum$wind[1]
+  hi <- optimum$wind[2]
+  gap <- pmax(lo - wind, wind - hi, 0)
+  score_wind <- 1 - pmin(gap / tolerance[["wind"]], 1)
+
+  comfort <- weights[["temp"]] *
+    score_temp +
+    weights[["humidity"]] * score_humidity +
+    weights[["wind"]] * score_wind
+  data.frame(
+    score_temp = score_temp,
+    score_humidity = score_humidity,
+    score_wind = score_wind,
+    comfort = comfort
+  )
+}
```

**Target (gold region)**

```r
  # Score `humidity`
  lo <- optimum$humidity[1]
  hi <- optimum$humidity[2]
  gap <- pmax(lo - humidity, humidity - hi, 0)
  score_humidity <- 1 - pmin(gap / tolerance[["humidity"]], 1)

  # Score `wind`
  lo <- optimum$wind[1]
  hi <- optimum$wind[2]
  gap <- pmax(lo - wind, wind - hi, 0)
  score_wind <- 1 - pmin(gap / tolerance[["wind"]], 1)

  comfort <- weights[["temp"]] *
    score_temp +
    weights[["humidity"]] * score_humidity +
    weights[["wind"]] * score_wind
  data.frame(
    score_temp = score_temp,
    score_humidity = score_humidity,
    score_wind = score_wind,
    comfort = comfort
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/indices-air.R
comfort_index <- function(
  temp,
  humidity,
  wind,
  weights = c(temp = 0.5, humidity = 0.3, wind = 0.2)
) {
  optimum <- list(temp = c(18, 24), humidity = c(40, 60), wind = c(0, 3))
  tolerance <- c(temp = 10, humidity = 30, wind = 7)
  weights <- weights[names(optimum)] / sum(weights[names(optimum)])

  # Score `temp`
  lo <- optimum$temp[1]
  hi <- optimum$temp[2]
  gap <- pmax(lo - temp, temp - hi, 0)
  score_temp <- 1 - pmin(gap / tolerance[["temp"]], 1)

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 43. `393ad4a99d9052018b131d93` finish_block / sibling:long

- package `synth:dashkit`, path `R/format_values.R`
- generator: `{"behaviour": "entered", "bucket": "long", "determinacy": "sibling", "full_body": false, "novel_tokens": ["ifelse"], "novelty": 1, "rule": "fb_nested_close", "seed": "b36-shiny/dashkit/R/format_values.R", "signature_region": false, "structural_support": 0.531, "target_lines": 22, "test_file": false, "unit": "format_table_columns", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 144, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 601, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -58,3 +58,25 @@
   # Format `currency` columns
   for (col in currency) {
     data[[col]] <- format_kpi(data[[col]], "currency", digits = 2)
+  }
+
+  # Format `percent` columns
+  for (col in percent) {
+    data[[col]] <- format_kpi(data[[col]], "percent", digits = 1)
+  }
+
+  # Format `count` columns
+  for (col in count) {
+    data[[col]] <- format_kpi(data[[col]], "count")
+  }
+
+  # Format `date` columns
+  for (col in date) {
+    data[[col]] <- ifelse(
+      is.na(data[[col]]),
+      "-",
+      format(as.Date(data[[col]]), "%d %b %Y")
+    )
+  }
+  data
+}
```

**Target (gold region)**

```r
  }

  # Format `percent` columns
  for (col in percent) {
    data[[col]] <- format_kpi(data[[col]], "percent", digits = 1)
  }

  # Format `count` columns
  for (col in count) {
    data[[col]] <- format_kpi(data[[col]], "count")
  }

  # Format `date` columns
  for (col in date) {
    data[[col]] <- ifelse(
      is.na(data[[col]]),
      "-",
      format(as.Date(data[[col]]), "%d %b %Y")
    )
  }
  data
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/format_values.R
#' @family formatters
#' @export
format_period_label <- function(
  start,
  by = c("day", "week", "month", "quarter", "year")
) {
  by <- match.arg(by)
  start <- as.Date(start)
  year <- format(start, "%Y")
  switch(
    by,
    day = {
      format(start, "%a %d %b %Y")
    },
    week = {
      monday <- start - (as.integer(format(start, "%u")) - 1L)
      paste("Week of", format(monday, "%Y-%m-%d"))
    },
    month = {
      paste(month.name[as.integer(format(start, "%m"))], year)
    },
    quarter = {
      quarter <- (as.integer(format(start, "%m")) - 1L) %/% 3L + 1L
      paste0("Q", quarter, " ", year)
    },
    year = {
      year
    }
  )
}

#' Format the columns of a results table
#'
#' Applies [format_kpi()] to groups of columns so that the table can be
#' rendered without further formatting.
#'
#' @param data A data frame of results.
#' @param currency Names of columns holding money amounts.
#' @param percent Names of columns holding shares between 0 and 1.
#' @param count Names of columns holding counts.
#' @param date Names of columns holding dates.
#' @return `data` with the named columns converted to character.
#' @family formatters
#' @export
format_table_columns <- function(
  data,
  currency = character(0),
  percent = character(0),
  count = character(0),
  date = character(0)
) {
  wanted <- c(currency, percent, count, date)
  missing <- setdiff(wanted, names(data))
  if (length(missing) > 0L) {
    stop("Unknown columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }

  # Format `currency` columns
  for (col in currency) {
    data[[col]] <- format_kpi(data[[col]], "currency", digits = 2)
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 44. `643995d72dbb0da2b429de4c` finish_block / sibling:long

- package `synth:dosewise`, path `R/dose.R`
- generator: `{"behaviour": "file_checked_not_entered", "bucket": "long", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b29-s4/dosewise/R/dose.R", "signature_region": false, "structural_support": 0.506, "target_lines": 29, "test_file": false, "unit": "dose.arith", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 182, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 427, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -27,3 +27,32 @@
     )
     validObject(e1)
     return(e1)
+  }
+
+  # Dose op number
+  if (is(e1, "Dose")) {
+    e1@amount <- switch(
+      op,
+      "*" = e1@amount * e2,
+      "/" = e1@amount / e2,
+      stop(
+        sprintf("`%s` is not defined for a dose and a number.", op),
+        call. = FALSE
+      )
+    )
+    validObject(e1)
+    return(e1)
+  }
+
+  # number op Dose
+  e2@amount <- switch(
+    op,
+    "*" = e1 * e2@amount,
+    stop(
+      sprintf("`%s` is not defined for a number and a dose.", op),
+      call. = FALSE
+    )
+  )
+  validObject(e2)
+  e2
+}
```

**Target (gold region)**

```r
  }

  # Dose op number
  if (is(e1, "Dose")) {
    e1@amount <- switch(
      op,
      "*" = e1@amount * e2,
      "/" = e1@amount / e2,
      stop(
        sprintf("`%s` is not defined for a dose and a number.", op),
        call. = FALSE
      )
    )
    validObject(e1)
    return(e1)
  }

  # number op Dose
  e2@amount <- switch(
    op,
    "*" = e1 * e2@amount,
    stop(
      sprintf("`%s` is not defined for a number and a dose.", op),
      call. = FALSE
    )
  )
  validObject(e2)
  e2
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/dose.R
# ---- diagnostics (static) --------------------
# typed statements: 0 of 5
# args not yet referenced: none
# unclosed braces in typed code: 2
# ----------------------------------------------
#' Arithmetic on doses
#'
#' @param e1,e2 The operands. At least one is a `Dose`.
#' @param op The operator name, as found in `.Generic`.
#' @details Two doses may be added or subtracted when they share a route; the
#'   result uses the unit of `e1`. A dose may be multiplied or divided by a
#'   number, and a number may multiply a dose.
#' @return A `Dose` object.
#' @keywords internal
dose.arith <- function(e1, e2, op) {
  # Dose op Dose
  if (is(e1, "Dose") && is(e2, "Dose")) {
    if (!identical(e1@route, e2@route)) {
      stop("Doses with different routes cannot be combined.", call. = FALSE)
    }
    e2 <- dose.convert(e2, e1@unit)
    e1@amount <- switch(
      op,
      "+" = e1@amount + e2@amount,
      "-" = e1@amount - e2@amount,
      stop(sprintf("`%s` is not defined for two doses.", op), call. = FALSE)
    )
    validObject(e1)
    return(e1)
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 45. `a558105c9b88df7fb26dc1e5` finish_block / constrained_short

- package `synth:tallyroll`, path `R/report.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["toupper"], "novelty": 1, "rule": "fb_statement", "seed": "b16-classic/tallyroll/R/report.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "reportHeading", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 26, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 657, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -58,3 +58,5 @@
 #' @export
 reportHeading <- function(title, width = 60, char = "=") {
   rule <- strrep(char, width)
+  c(rule, padText(toupper(title), width, "centre"), rule)
+}
```

**Target (gold region)**

```r
  c(rule, padText(toupper(title), width, "centre"), rule)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/report.R
#' @param digits Number of decimal places.
#' @return A character vector the same length as `x`.
#' @rdname formatters
#' @export
formatMoney <- function(x, symbol = "$", digits = 2) {
  out <- paste0(
    symbol,
    formatC(abs(x), format = "f", digits = digits, big.mark = ",")
  )
  # The sign goes in front of the symbol: -$1,200.00 rather than $-1,200.00
  out[x < 0 & !is.na(x)] <- paste0("-", out[x < 0 & !is.na(x)])
  out[is.na(x)] <- "NA"
  out
}

#' @rdname formatters
#' @export
formatPercent <- function(x, digits = 1) {
  out <- sprintf(paste0("%.", digits, "f%%"), 100 * x)
  out[is.na(x)] <- "NA"
  out
}

#' @rdname formatters
#' @export
formatCount <- function(x, digits = 0) {
  out <- formatC(round(x), format = "d", big.mark = ",")
  out[is.na(x)] <- "NA"
  out
}

#' Pad strings to a fixed width
#'
#' @param x A character vector.
#' @param width Target width in characters.
#' @param align One of "left", "right" or "centre".
#' @return A character vector the same length as `x`.
#' @keywords internal
padText <- function(x, width, align = c("left", "right", "centre")) {
  align <- match.arg(align)
  x <- as.character(x)
  gap <- pmax(0, width - nchar(x))

  switch(
    align,
    left = paste0(x, strrep(" ", gap)),
    right = paste0(strrep(" ", gap), x),
    centre = paste0(strrep(" ", gap %/% 2), x, strrep(" ", gap - gap %/% 2))
  )
}

#' Build a report heading
#'
#' @param title Heading text.
#' @param width Total width of the rule lines.
#' @param char Character used for the rule lines.
#' @return A character vector of three lines.
#' @export
reportHeading <- function(title, width = 60, char = "=") {
  rule <- strrep(char, width)
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 46. `1b0cacdf00fac1d9e25c54e9` finish_block / constrained_short

- package `synth:lintwright`, path `R/rules.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b46-lint/lintwright/R/rules.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "lint_code", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 11, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 454, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -34,3 +34,5 @@
 
   lints <- lints[order(lints$line, lints$column), , drop = FALSE]
   rownames(lints) <- NULL
+  lints
+}
```

**Target (gold region)**

```r
  lints
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/rules.R
lint_code <- function(
  text,
  rules = c(
    "assign_eq",
    "true_false",
    "semicolon",
    "trailing_space",
    "single_quotes",
    "browser_call"
  ),
  nolint = TRUE
) {
  rules <- match.arg(rules, several.ok = TRUE)
  lints <- do.call(rbind, lapply(rules, function(r) lint_rule(text, r)))

  if (nolint) {
    # A bare `# nolint` silences every rule on its line
    all_off <- grepl("# nolint\\s*$", text)
    lints <- lints[!all_off[lints$line], , drop = FALSE]
    # `# nolint: a, b` silences only the listed rules
    listed <- regmatches(text, regexec("# nolint: ?([a-z_, ]+)$", text))
    off <- lapply(listed, function(m) {
      if (length(m) == 0L) character() else trimws(strsplit(m[2], ",")[[1]])
    })
    silenced <- vapply(
      seq_len(nrow(lints)),
      function(i) {
        lints$rule[i] %in% off[[lints$line[i]]]
      },
      logical(1)
    )
    lints <- lints[!silenced, , drop = FALSE]
  }

  lints <- lints[order(lints$line, lints$column), , drop = FALSE]
  rownames(lints) <- NULL
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 47. `a90a5fc66a0706f03d270e30` finish_block / constrained_short

- package `synth:gaugemark`, path `R/arith.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b28-s3/gaugemark/R/arith.R", "signature_region": false, "structural_support": 0.111, "target_lines": 7, "test_file": false, "unit": "commonUnit", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 51, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 424, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -27,3 +27,10 @@
   u1 <- attr(e1, "unit")
   u2 <- attr(e2, "unit")
   if (is.null(u1)) {
+    return(u2)
+  }
+  if (is.null(u2)) {
+    return(u1)
+  }
+  if (unitFactor(u1) <= unitFactor(u2)) u1 else u2
+}
```

**Target (gold region)**

```r
    return(u2)
  }
  if (is.null(u2)) {
    return(u1)
  }
  if (unitFactor(u1) <= unitFactor(u2)) u1 else u2
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/arith.R
#' @title Express an operand in a given unit
#' @param x A `gm_reading`, a `gm_range` or a plain number.
#' @param unit The target unit.
#' @return A double vector for readings and numbers, a list with elements
#'   `lower` and `upper` for ranges. Plain numbers are returned unchanged.
#' @keywords internal
inUnit <- function(x, unit) {
  from <- attr(x, "unit")
  if (is.null(from)) {
    return(as.double(x))
  }
  ratio <- unitFactor(from) / unitFactor(unit)
  if (inherits(x, "gm_range")) {
    fields <- unclass(x)
    return(list(lower = fields$lower * ratio, upper = fields$upper * ratio))
  }
  as.double(unclass(x)) * ratio
}

#' @title Pick the unit two operands should share
#' @description The finer of the two units wins, so that no precision is lost
#'   when one side is converted. A plain number has no unit and never wins.
#' @param e1,e2 Operands of a group generic.
#' @return A single unit string.
#' @keywords internal
commonUnit <- function(e1, e2) {
  u1 <- attr(e1, "unit")
  u2 <- attr(e2, "unit")
  if (is.null(u1)) {
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 48. `b603247c237b6bd3e2648ef8` finish_block / constrained_short

- package `synth:gaugemark`, path `R/classes.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b28-s3/gaugemark/R/classes.R", "signature_region": false, "structural_support": 0.263, "target_lines": 6, "test_file": false, "unit": "validateReading", "variant": "outline"}`
- verification: `{"air_idempotent": true, "completion_tokens": 46, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 404, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -27,3 +27,9 @@
   }
   if (any(is.infinite(value))) {
     stop("`value` must not contain infinite values.", call. = FALSE)
+  }
+  if (any(value < 0, na.rm = TRUE)) {
+    stop("`value` must not be negative.", call. = FALSE)
+  }
+  x
+}
```

**Target (gold region)**

```r
  }
  if (any(value < 0, na.rm = TRUE)) {
    stop("`value` must not be negative.", call. = FALSE)
  }
  x
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/classes.R
# ---- outline --------------------------------
# fn: validateReading(x)
# plan: 8 top-level statements; 5 typed above the cursor
# docs: yes
# ----------------------------------------------
#' @title Validate a reading
#' @description Checks the `unit` attribute, then the values: they must be
#'   doubles, finite where present, and not negative.
#' @param x An object of class `gm_reading`.
#' @return `x`, unchanged, if it is valid.
#' @keywords internal
validateReading <- function(x) {
  value <- unclass(x)
  unit <- attr(x, "unit")

  # Check the unit first: every later step depends on it
  if (!is.character(unit) || length(unit) != 1L) {
    stop("`unit` must be a single string.", call. = FALSE)
  }
  if (!unit %in% c("mm", "cm", "m", "km", "in")) {
    stop(sprintf("Unknown unit '%s'.", unit), call. = FALSE)
  }

  # Check the values
  if (!is.double(value)) {
    stop("`value` must be a double vector.", call. = FALSE)
  }
  if (any(is.infinite(value))) {
    stop("`value` must not contain infinite values.", call. = FALSE)
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 49. `1b891f94e067298e34ae29f0` roxygen_drafting / roxy_full

- package `synth:markovine`, path `R/montecarlo.R`
- generator: `{"behaviour": "entered", "determined_share": 0.667, "rule": "roxy_full", "seed": "b23-simulation/markovine/R/montecarlo.R", "tag_order": "rdname param export", "target_lines": 3, "unit": "mc.coverage"}`
- verification: `{"air_idempotent": true, "completion_tokens": 35, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 294, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -16,6 +16,9 @@
   return(sqrt(mean((estimate - truth)^2, na.rm = TRUE)))
 }
 
+#' @rdname mc.bias
+#' @param lower,upper Numeric vectors of replicate interval bounds.
+#' @export
 mc.coverage <- function(lower, upper, truth) {
   return(mean(lower <= truth & truth <= upper, na.rm = TRUE))
 }
```

**Target (gold region)**

```r
#' @rdname mc.bias
#' @param lower,upper Numeric vectors of replicate interval bounds.
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/montecarlo.R
#' @title Monte Carlo performance measures
#' @description Bias, root mean squared error and interval coverage of
#'   replicate estimates against a known true value. Missing replicates are
#'   dropped.
#' @param estimate Numeric vector of replicate estimates.
#' @param truth True parameter value.
#' @return A single number.
#' @export
mc.bias <- function(estimate, truth) {
  return(mean(estimate - truth, na.rm = TRUE))
}

#' @rdname mc.bias
#' @export
mc.rmse <- function(estimate, truth) {
  return(sqrt(mean((estimate - truth)^2, na.rm = TRUE)))
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
mc.coverage <- function(lower, upper, truth) {
  return(mean(lower <= truth & truth <= upper, na.rm = TRUE))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 50. `36a5d1b7fe134aca9b12b46b` roxygen_drafting / roxy_full

- package `synth:numquill`, path `R/linalg.R`
- generator: `{"behaviour": "entered", "determined_share": 0.833, "rule": "roxy_full", "seed": "b02-base-stats/numquill/R/linalg.R", "tag_order": "param return export", "target_lines": 6, "unit": "ols.normal"}`
- verification: `{"air_idempotent": true, "completion_tokens": 64, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 509, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,6 +28,12 @@
   list(coef = beta, fitted = fitted, resid = resid, sigma = sigma)
 }
 
+#' Least squares via the normal equations
+#'
+#' @param X Numeric design matrix.
+#' @param y Numeric response vector.
+#' @return A list with `coef`, `fitted`, `resid` and `sigma`.
+#' @export
 ols.normal <- function(X, y) {
   # Squares the condition number of X; fine for well-scaled designs only.
   XtX <- crossprod(X)
```

**Target (gold region)**

```r
#' Least squares via the normal equations
#'
#' @param X Numeric design matrix.
#' @param y Numeric response vector.
#' @return A list with `coef`, `fitted`, `resid` and `sigma`.
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/linalg.R
#' @param A A symmetric positive definite matrix.
#' @param b Right-hand side vector or matrix.
#' @return The solution of `A %*% x = b`.
#' @export
cholesky.solve <- function(A, b) {
  R <- check.pos.def(A)

  # A = R'R, so solve R'z = b forwards and then R x = z backwards.
  backsolve(R, forwardsolve(t(R), b))
}

#' Least squares via the QR decomposition
#'
#' @param X Numeric design matrix.
#' @param y Numeric response vector.
#' @return A list with `coef`, `fitted`, `resid` and `sigma`.
#' @export
ols.qr <- function(X, y) {
  qx <- qr(X)
  if (qx$rank < ncol(X)) {
    stop("`X` is rank deficient.", call. = FALSE)
  }

  beta <- qr.coef(qx, y)
  fitted <- qr.fitted(qx, y)
  resid <- y - fitted
  sigma <- sqrt(sum(resid^2) / (nrow(X) - ncol(X)))
  list(coef = beta, fitted = fitted, resid = resid, sigma = sigma)
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
ols.normal <- function(X, y) {
  # Squares the condition number of X; fine for well-scaled designs only.
  XtX <- crossprod(X)
  beta <- drop(cholesky.solve(XtX, crossprod(X, y)))
  names(beta) <- colnames(X)
  fitted <- drop(X %*% beta)
  resid <- y - fitted
  sigma <- sqrt(sum(resid^2) / (nrow(X) - ncol(X)))
  list(coef = beta, fitted = fitted, resid = resid, sigma = sigma)
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 51. `a65ffef41834cb11cb6716e8` roxygen_drafting / roxy_full

- package `synth:chainlet`, path `R/metropolis.R`
- generator: `{"behaviour": "entered", "determined_share": 0.667, "rule": "roxy_full", "seed": "b02-base-stats/chainlet/R/metropolis.R", "tag_order": "inheritParams param return export", "target_lines": 6, "unit": "logPriorBeta"}`
- verification: `{"air_idempotent": true, "completion_tokens": 62, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 390, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -23,6 +23,12 @@
   return(sum(stats::dgamma(theta, shape = shape, rate = rate, log = TRUE)))
 }
 
+#' Log density of a beta prior
+#'
+#' @inheritParams logPriorNormal
+#' @param shape1,shape2 Prior shape parameters.
+#' @return The summed log density, or `-Inf` outside the support.
+#' @export
 logPriorBeta <- function(theta, shape1 = 1, shape2 = 1) {
   if (any(theta <= 0 | theta >= 1)) {
     return(-Inf)
```

**Target (gold region)**

```r
#' Log density of a beta prior
#'
#' @inheritParams logPriorNormal
#' @param shape1,shape2 Prior shape parameters.
#' @return The summed log density, or `-Inf` outside the support.
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/metropolis.R
#' Log density of a normal prior
#'
#' @param theta Numeric parameter vector.
#' @param mean Prior mean.
#' @param sd Prior standard deviation.
#' @return The summed log density.
#' @export
logPriorNormal <- function(theta, mean = 0, sd = 10) {
  return(sum(stats::dnorm(theta, mean = mean, sd = sd, log = TRUE)))
}

#' Log density of a gamma prior
#'
#' @inheritParams logPriorNormal
#' @param shape Prior shape.
#' @param rate Prior rate.
#' @return The summed log density, or `-Inf` outside the support.
#' @export
logPriorGamma <- function(theta, shape = 1, rate = 1) {
  if (any(theta <= 0)) {
    return(-Inf)
  }
  return(sum(stats::dgamma(theta, shape = shape, rate = rate, log = TRUE)))
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
logPriorBeta <- function(theta, shape1 = 1, shape2 = 1) {
  if (any(theta <= 0 | theta >= 1)) {
    return(-Inf)
  }
  return(sum(stats::dbeta(theta, shape1 = shape1, shape2 = shape2, log = TRUE)))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 52. `5ca5a45323ceb31f7eb3cdb1` roxygen_drafting / roxy_full

- package `synth:nestwalk`, path `R/tree_walk.R`
- generator: `{"behaviour": "entered", "determined_share": 0.8, "rule": "roxy_full", "seed": "b16-classic/nestwalk/R/tree_walk.R", "tag_order": "param return export", "target_lines": 5, "unit": "tree_count_leaves"}`
- verification: `{"air_idempotent": true, "completion_tokens": 40, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 294, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -16,6 +16,11 @@
   return(1L + max(vapply(x, tree_depth, integer(1))))
 }
 
+#' Number of leaves in a nested list
+#'
+#' @param x A nested list.
+#' @return A single integer.
+#' @export
 tree_count_leaves <- function(x) {
   if (!is.list(x)) {
     return(1L)
```

**Target (gold region)**

```r
#' Number of leaves in a nested list
#'
#' @param x A nested list.
#' @return A single integer.
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/tree_walk.R
#' Depth of a nested list
#'
#' A leaf (anything that is not a list) has depth zero; an empty list has
#' depth one.
#'
#' @param x A nested list.
#' @return A single integer.
#' @export
tree_depth <- function(x) {
  if (!is.list(x)) {
    return(0L)
  }
  if (length(x) == 0L) {
    return(1L)
  }
  return(1L + max(vapply(x, tree_depth, integer(1))))
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
tree_count_leaves <- function(x) {
  if (!is.list(x)) {
    return(1L)
  }
  if (length(x) == 0L) {
    return(0L)
  }
  return(sum(vapply(x, tree_count_leaves, integer(1))))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 53. `226518e642e2815f1d5560b8` roxygen_drafting / roxy_full

- package `synth:cellfmt`, path `R/numbers.R`
- generator: `{"behaviour": "entered", "determined_share": 0.7, "rule": "roxy_full", "seed": "b26-reporting/cellfmt/R/numbers.R", "tag_order": "param return family export", "target_lines": 10, "unit": "fmt_integer"}`
- verification: `{"air_idempotent": true, "completion_tokens": 98, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 482, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,6 +28,16 @@
   return(out)
 }
 
+#' Format whole numbers
+#'
+#' Rounds `x` to the nearest integer and inserts a thousands separator.
+#'
+#' @param x A numeric vector to format.
+#' @param big_mark Character used to separate thousands.
+#' @param na_string String used in place of missing values.
+#' @return A character vector the same length as `x`.
+#' @family number formatters
+#' @export
 fmt_integer <- function(x, big_mark = ",", na_string = "") {
   check_number_args(x, 0, big_mark, na_string)
   out <- formatC(round(x), format = "d", big.mark = big_mark)
```

**Target (gold region)**

```r
#' Format whole numbers
#'
#' Rounds `x` to the nearest integer and inserts a thousands separator.
#'
#' @param x A numeric vector to format.
#' @param big_mark Character used to separate thousands.
#' @param na_string String used in place of missing values.
#' @return A character vector the same length as `x`.
#' @family number formatters
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/numbers.R
#' @family number formatters
#' @export
fmt_fixed <- function(x, digits = 2, big_mark = ",", na_string = "") {
  check_number_args(x, digits, big_mark, na_string)
  out <- formatC(x, format = "f", digits = digits, big.mark = big_mark)
  out[is.na(x)] <- na_string
  return(out)
}

#' Format proportions as percentages
#'
#' Multiplies `x` by `scale`, rounds to `digits` decimals and appends a
#' percent sign.
#'
#' @param x A numeric vector to format.
#' @param digits Number of decimal places to keep.
#' @param scale Multiplier applied before formatting; use 1 when `x` already
#'   holds percentages.
#' @param na_string String used in place of missing values.
#' @return A character vector the same length as `x`.
#' @family number formatters
#' @export
fmt_percent <- function(x, digits = 1, scale = 100, na_string = "") {
  check_number_args(x, digits, ",", na_string)
  out <- formatC(x * scale, format = "f", digits = digits)
  out <- paste0(out, "%")
  out[is.na(x)] <- na_string
  return(out)
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
fmt_integer <- function(x, big_mark = ",", na_string = "") {
  check_number_args(x, 0, big_mark, na_string)
  out <- formatC(round(x), format = "d", big.mark = big_mark)
  out[is.na(x)] <- na_string
  return(out)
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 54. `252a6be02f4c70afefb1954a` roxygen_drafting / roxy_full

- package `synth:pathwren`, path `R/strings.R`
- generator: `{"behaviour": "entered", "determined_share": 0.75, "rule": "roxy_full", "seed": "b06-infra-tests/pathwren/R/strings.R", "tag_order": "title param noRd", "target_lines": 4, "unit": "checkPadChar"}`
- verification: `{"air_idempotent": true, "completion_tokens": 54, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 504, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -27,6 +27,10 @@
   return(invisible(x))
 }
 
+#' @title Check that an argument is a single padding character
+#' @param x The object to check.
+#' @param arg The name of the argument being checked, used in error messages.
+#' @noRd
 checkPadChar <- function(x, arg = deparse(substitute(x))) {
   if (!is.character(x) || length(x) != 1L || is.na(x) || nchar(x) != 1L) {
     rlang::abort(
```

**Target (gold region)**

```r
#' @title Check that an argument is a single padding character
#' @param x The object to check.
#' @param arg The name of the argument being checked, used in error messages.
#' @noRd
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/strings.R
#' @title Count pattern matches in strings
#' @param x A character vector.
#' @param pattern A regular expression, or a fixed string when `fixed = TRUE`.
#' @param fixed Whether `pattern` is matched literally.
#' @return An integer vector the same length as `x`.
#' @export
strCountMatches <- function(x, pattern, fixed = FALSE) {
  checkCharacter(x)
  matches <- gregexpr(pattern, x, fixed = fixed)
  # gregexpr() reports "no match" as a single -1
  counts <- vapply(matches, function(m) sum(m > 0L), integer(1))
  counts[is.na(x)] <- NA_integer_
  return(counts)
}

#' @title Check that an argument is a character vector
#' @param x The object to check.
#' @param arg The name of the argument being checked, used in error messages.
#' @noRd
checkCharacter <- function(x, arg = deparse(substitute(x))) {
  if (!is.character(x)) {
    rlang::abort(
      sprintf("`%s` must be a character vector, not %s.", arg, class(x)[1L]),
      class = c("pathwren_error_type", "pathwren_error")
    )
  }
  return(invisible(x))
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
checkPadChar <- function(x, arg = deparse(substitute(x))) {
  if (!is.character(x) || length(x) != 1L || is.na(x) || nchar(x) != 1L) {
    rlang::abort(
      sprintf("`%s` must be a single character.", arg),
      class = c("pathwren_error_pad", "pathwren_error")
    )
  }
  return(invisible(x))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 55. `997a3cf870224dca7e232c07` roxygen_drafting / roxy_full

- package `synth:summarine`, path `R/column_profile.R`
- generator: `{"behaviour": "entered", "determined_share": 0.7, "rule": "roxy_full", "seed": "b27-statistics/summarine/R/column_profile.R", "tag_order": "inheritParams return family export", "target_lines": 10, "unit": "profile_factor"}`
- verification: `{"air_idempotent": true, "completion_tokens": 110, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 558, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -27,6 +27,16 @@
   )
 }
 
+#' Profile a factor column
+#'
+#' The center is the proportion of the modal level and the spread is the
+#' Shannon entropy of the level frequencies divided by its maximum.
+#'
+#' @inheritParams profile_numeric
+#' @return A one-row data frame with columns `column`, `type`, `n`,
+#'   `n_missing`, `n_distinct`, `center` and `spread`.
+#' @family column profilers
+#' @export
 profile_factor <- function(x, name = "x") {
   ok <- x[!is.na(x)]
 
```

**Target (gold region)**

```r
#' Profile a factor column
#'
#' The center is the proportion of the modal level and the spread is the
#' Shannon entropy of the level frequencies divided by its maximum.
#'
#' @inheritParams profile_numeric
#' @return A one-row data frame with columns `column`, `type`, `n`,
#'   `n_missing`, `n_distinct`, `center` and `spread`.
#' @family column profilers
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/column_profile.R
#' Profile a logical column
#'
#' The center is the proportion of `TRUE` values and the spread is the
#' Bernoulli standard deviation.
#'
#' @inheritParams profile_numeric
#' @return A one-row data frame with columns `column`, `type`, `n`,
#'   `n_missing`, `n_distinct`, `center` and `spread`.
#' @family column profilers
#' @export
profile_logical <- function(x, name = "x") {
  ok <- x[!is.na(x)]

  # Center and spread of the non-missing values
  center <- if (length(ok) > 0L) mean(ok) else NA_real_
  spread <- if (length(ok) > 0L) sqrt(center * (1 - center)) else NA_real_

  data.frame(
    column = name,
    type = "logical",
    n = length(x),
    n_missing = sum(is.na(x)),
    n_distinct = length(unique(ok)),
    center = center,
    spread = spread,
    stringsAsFactors = FALSE
  )
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
profile_factor <- function(x, name = "x") {
  ok <- x[!is.na(x)]

  # Center and spread of the non-missing values
  freq <- tabulate(as.integer(ok), nbins = nlevels(x)) / max(length(ok), 1L)
  freq <- freq[freq > 0]
  center <- if (length(ok) > 0L) max(freq) else NA_real_
  spread <- if (length(freq) > 1L) {
    -sum(freq * log(freq)) / log(length(freq))
  } else {
    0
  }

  data.frame(
    column = name,
    type = "factor",
    n = length(x),
    n_missing = sum(is.na(x)),
    n_distinct = length(unique(ok)),
    center = center,
    spread = spread,
    stringsAsFactors = FALSE
  )
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 56. `2ef03d9c077e791e3b6bd6b3` roxygen_drafting / roxy_full

- package `synth:pursekeep`, path `R/projection.R`
- generator: `{"behaviour": "entered", "determined_share": 0.667, "rule": "roxy_full", "seed": "b04-shiny/pursekeep/R/projection.R", "tag_order": "noRd param", "target_lines": 6, "unit": "checkMonths"}`
- verification: `{"air_idempotent": true, "completion_tokens": 52, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 413, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,6 +28,12 @@
   NULL
 }
 
+#' Check a number of months input
+#'
+#' Months must be present and a whole number of at least one.
+#'
+#' @noRd
+#' @param value The value entered by the user.
 checkMonths <- function(value) {
   if (is.null(value) || is.na(value)) {
     return("Number of months is required.")
```

**Target (gold region)**

```r
#' Check a number of months input
#'
#' Months must be present and a whole number of at least one.
#'
#' @noRd
#' @param value The value entered by the user.
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/projection.R
#' @keywords internal
#' @param value The value entered by the user.
#' @param label A human-readable field name used in the message.
#' @return `NULL` when valid, otherwise a message for [shiny::validate()].
checkAmount <- function(value, label) {
  if (is.null(value) || is.na(value)) {
    return(paste(label, "is required."))
  }
  if (value < 0) {
    return(paste(label, "cannot be negative."))
  }
  NULL
}

#' Check an interest rate input
#'
#' Rates must be present and between 0 and 25 percent.
#'
#' @noRd
#' @param value The value entered by the user, in percent.
checkRate <- function(value) {
  if (is.null(value) || is.na(value)) {
    return("Interest rate is required.")
  }
  if (value < 0 || value > 25) {
    return("Interest rate must be between 0% and 25%.")
  }
  NULL
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
checkMonths <- function(value) {
  if (is.null(value) || is.na(value)) {
    return("Number of months is required.")
  }
  if (value < 1 || value != round(value)) {
    return("Number of months must be a whole number of at least 1.")
  }
  NULL
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 57. `84067fe155894f2d22a66805` roxygen_drafting / roxy_full

- package `synth:trialglue`, path `R/contrasts.R`
- generator: `{"behaviour": "no_check", "determined_share": 0.833, "rule": "roxy_full", "seed": "b15-psych-trials/trialglue/R/contrasts.R", "tag_order": "inheritParams param return export", "target_lines": 6, "unit": "arm_pairwise"}`
- verification: `{"air_idempotent": true, "completion_tokens": 58, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 491, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,6 +28,12 @@
   contrast(emm, method = "trt.vs.ctrl", ref = ref, adjust = adjust)
 }
 
+#' All pairwise arm contrasts at each visit
+#'
+#' @inheritParams arm_means
+#' @param adjust Multiplicity adjustment passed to emmeans.
+#' @return An `emmGrid` object of contrasts.
+#' @export
 arm_pairwise <- function(fit, treatment, time, adjust = "tukey") {
   emm <- arm_means(fit, treatment, time)
   contrast(emm, method = "pairwise", adjust = adjust)
```

**Target (gold region)**

```r
#' All pairwise arm contrasts at each visit
#'
#' @inheritParams arm_means
#' @param adjust Multiplicity adjustment passed to emmeans.
#' @return An `emmGrid` object of contrasts.
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/contrasts.R
#' repeated-measures trial.
#'
#' @param fit A fitted model supported by emmeans, e.g. from [fit_trial_lmm()].
#' @param treatment Name of the treatment arm column.
#' @param time Name of the visit or time column.
#' @param type Scale of the estimates, `"link"` or `"response"`.
#' @return An `emmGrid` object.
#' @importFrom emmeans emmeans
#' @export
arm_means <- function(fit, treatment, time, type = c("link", "response")) {
  type <- match.arg(type)
  specs <- stats::as.formula(paste("~", treatment, "|", time))
  emmeans(fit, specs = specs, type = type)
}

#' Treatment-versus-control contrasts at each visit
#'
#' Compares every active arm with the reference arm, within each visit.
#'
#' @inheritParams arm_means
#' @param ref Index of the reference (control) arm among the factor levels.
#' @param adjust Multiplicity adjustment passed to emmeans.
#' @return An `emmGrid` object of contrasts.
#' @importFrom emmeans contrast
#' @export
arm_vs_control <- function(fit, treatment, time, ref = 1, adjust = "dunnett") {
  emm <- arm_means(fit, treatment, time)
  contrast(emm, method = "trt.vs.ctrl", ref = ref, adjust = adjust)
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
arm_pairwise <- function(fit, treatment, time, adjust = "tukey") {
  emm <- arm_means(fit, treatment, time)
  contrast(emm, method = "pairwise", adjust = adjust)
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 58. `fd870bdf69b0973c44b929cb` roxygen_drafting / roxy_full

- package `synth:glyphgauge`, path `R/distance.R`
- generator: `{"behaviour": "entered", "determined_share": 0.667, "rule": "roxy_full", "seed": "b10-text/glyphgauge/R/distance.R", "tag_order": "title description param return export", "target_lines": 9, "unit": "lcsDist"}`
- verification: `{"air_idempotent": true, "completion_tokens": 95, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 527, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -22,6 +22,15 @@
   sum(toChars(a) != toChars(b))
 }
 
+#' @title Longest common subsequence distance
+#'
+#' @description The number of characters that must be deleted from both
+#'   strings so that what remains is identical.
+#'
+#' @param a,b Single strings to compare.
+#' @param ignoreCase Logical; compare case-insensitively?
+#' @return A single integer distance, or `NA` if either input is missing.
+#' @export
 lcsDist <- function(a, b, ignoreCase = FALSE) {
   checkPair(a, b)
   if (is.na(a) || is.na(b)) {
```

**Target (gold region)**

```r
#' @title Longest common subsequence distance
#'
#' @description The number of characters that must be deleted from both
#'   strings so that what remains is identical.
#'
#' @param a,b Single strings to compare.
#' @param ignoreCase Logical; compare case-insensitively?
#' @return A single integer distance, or `NA` if either input is missing.
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/distance.R
#' @title Hamming distance
#'
#' @description Counts positions at which two equal-length strings differ.
#'
#' @param a,b Single strings to compare.
#' @param ignoreCase Logical; compare case-insensitively?
#' @return A single integer distance, or `NA` if either input is missing.
#' @export
hammingDist <- function(a, b, ignoreCase = FALSE) {
  checkPair(a, b)
  if (is.na(a) || is.na(b)) {
    return(NA_integer_)
  }
  if (ignoreCase) {
    a <- tolower(a)
    b <- tolower(b)
  }

  if (nchar(a) != nchar(b)) {
    stop("Hamming distance needs strings of equal length.", call. = FALSE)
  }
  sum(toChars(a) != toChars(b))
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
lcsDist <- function(a, b, ignoreCase = FALSE) {
  checkPair(a, b)
  if (is.na(a) || is.na(b)) {
    return(NA_integer_)
  }
  if (ignoreCase) {
    a <- tolower(a)
    b <- tolower(b)
  }

  ca <- toChars(a)
  cb <- toChars(b)
  prev <- integer(length(cb) + 1L)
  for (i in seq_along(ca)) {
    curr <- integer(length(cb) + 1L)
    for (j in seq_along(cb)) {
      curr[j + 1L] <- if (ca[i] == cb[j]) {
        prev[j] + 1L
      } else {
        max(prev[j + 1L], curr[j])
      }
    }
    prev <- curr
  }

  # everything outside the common subsequence has to go, from both sides
  length(ca) + length(cb) - 2L * prev[length(prev)]
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 59. `531d6b0f2ddf7b9e32cc2ee1` roxygen_drafting / roxy_full

- package `synth:reportloom`, path `R/summaries.R`
- generator: `{"behaviour": "entered", "determined_share": 0.667, "rule": "roxy_full", "seed": "b26-reporting/reportloom/R/summaries.R", "tag_order": "export param rdname", "target_lines": 3, "unit": "describe.factor"}`
- verification: `{"air_idempotent": true, "completion_tokens": 31, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 595, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,6 +28,9 @@
   paste0(text, ".")
 }
 
+#' @export
+#' @param top Number of most frequent levels to list.
+#' @rdname describe
 describe.factor <- function(x, name = "x", top = 3L, ...) {
   n.missing <- sum(is.na(x))
   x <- x[!is.na(x)]
```

**Target (gold region)**

```r
#' @export
#' @param top Number of most frequent levels to list.
#' @rdname describe
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/summaries.R
#' @param x The vector or data frame to describe.
#' @param name Name used as the subject of the sentence.
#' @param ... Passed on to methods.
#' @return A single string ending in a full stop; a named character vector
#'   for data frames.
describe <- function(x, name = "x", ...) {
  UseMethod("describe")
}

#' @export
#' @param digits Number of decimal places to keep.
#' @rdname describe
describe.numeric <- function(x, name = "x", digits = 1L, ...) {
  n.missing <- sum(is.na(x))
  x <- x[!is.na(x)]
  if (length(x) == 0L) {
    return(sprintf("%s has no observed values.", name))
  }

  low <- formatC(min(x), format = "f", digits = digits)
  high <- formatC(max(x), format = "f", digits = digits)
  centre <- formatC(mean(x), format = "f", digits = digits)
  text <- sprintf("%s ranges from %s to %s (mean %s)", name, low, high, centre)
  if (n.missing > 0L) {
    text <- paste0(text, "; ", pluralise(n.missing, "value"), " missing")
  }

  paste0(text, ".")
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
describe.factor <- function(x, name = "x", top = 3L, ...) {
  n.missing <- sum(is.na(x))
  x <- x[!is.na(x)]
  if (length(x) == 0L) {
    return(sprintf("%s has no observed values.", name))
  }

  counts <- sort(table(x), decreasing = TRUE)
  counts <- counts[seq_len(min(top, length(counts)))]
  levels <- sprintf("%s (%d)", names(counts), as.integer(counts))
  text <- sprintf("%s is most often %s", name, list.phrase(levels, "or"))
  if (n.missing > 0L) {
    text <- paste0(text, "; ", pluralise(n.missing, "value"), " missing")
  }

  paste0(text, ".")
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 60. `0dd245533c505ee239b84e50` roxygen_drafting / roxy_full

- package `synth:spanwise`, path `R/span-summary.R`
- generator: `{"behaviour": "entered", "determined_share": 0.714, "rule": "roxy_full", "seed": "b28-s3/spanwise/R/span-summary.R", "tag_order": "param return export", "target_lines": 7, "unit": "print.summary_spanwise_gap"}`
- verification: `{"air_idempotent": true, "completion_tokens": 69, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 544, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -23,6 +23,13 @@
   return(invisible(x))
 }
 
+#' Print a gap summary
+#'
+#' @param x An object of class `summary_spanwise_gap`.
+#' @param digits Number of significant digits used for widths.
+#' @param ... Unused.
+#' @return `x`, invisibly.
+#' @export
 print.summary_spanwise_gap <- function(x, digits = 4, ...) {
   cat(sprintf("%-12s%d\n", "Gaps", x$n))
   if (x$n == 0L) {
```

**Target (gold region)**

```r
#' Print a gap summary
#'
#' @param x An object of class `summary_spanwise_gap`.
#' @param digits Number of significant digits used for widths.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/span-summary.R
#' Print a span summary
#'
#' @param x An object of class `summary_spanwise_span`.
#' @param digits Number of significant digits used for widths.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @export
print.summary_spanwise_span <- function(x, digits = 4, ...) {
  cat(sprintf("%-12s%d\n", "Spans", x$n))
  if (x$n == 0L) {
    return(invisible(x))
  }
  rows <- c(
    "Total width" = x$total_width,
    "Shortest" = x$min_width,
    "Longest" = x$max_width,
    "Extent from" = x$extent[1L],
    "Extent to" = x$extent[2L]
  )
  labels <- formatC(names(rows), width = -12L)
  cat(paste0(labels, format(rows, digits = digits)), sep = "\n")
  cat(sprintf("%-12s%d\n", "Overlaps", x$n_overlaps))
  return(invisible(x))
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
print.summary_spanwise_gap <- function(x, digits = 4, ...) {
  cat(sprintf("%-12s%d\n", "Gaps", x$n))
  if (x$n == 0L) {
    return(invisible(x))
  }
  rows <- c(
    "Total width" = x$total_width,
    "Shortest" = x$min_width,
    "Longest" = x$max_width,
    "Extent from" = x$extent[1L],
    "Extent to" = x$extent[2L]
  )
  labels <- formatC(names(rows), width = -12L)
  cat(paste0(labels, format(rows, digits = digits)), sep = "\n")
  cat(sprintf("%-12s%d\n", "Empty", x$n_empty))
  return(invisible(x))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>
