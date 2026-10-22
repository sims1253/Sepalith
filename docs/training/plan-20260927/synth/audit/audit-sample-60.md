# SYNTH-v2 audit sample

Bound to `/mnt/e/sepalith/campaign-20260915/data-work/SYNTH-v2/manifest.json` sha256 `c3c7d743ab4f90fd6789823b30aec3f8c562156488cd31106b963d6c1649f280` (build-summary.json `3987382c6353a1ee51f736392175a34959d3760cc6adb51efc3e511053f0da2d`).

Stratified random sample (seed 20261022) of 60 of 643 cases. Strata: {"finish_block": {"constrained_short": 22, "sibling:short": 6, "sibling:medium": 16, "sibling:long": 4}, "roxygen_drafting": {"roxy_full": 12}}.

Each case shows the edited window as a diff (prefix + region + suffix, before and after the gold edit),
the gold target, the verification record and the exact rendered prompt of the drop-in TRAIN row.

## 1. `9ae71fa0af815b9db3a167c3` finish_block / constrained_short

- package `synth:herdwalk`, path `R/movement.R`
- generator: `{"behaviour": "checked", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["apply_boundary"], "novelty": 1, "rule": "fb_statement", "seed": "b23-simulation/herdwalk/R/movement.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "move_random", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 16, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 334, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -17,3 +17,5 @@
 
   pop$x[mobile] <- pop$x[mobile] + stats::rnorm(n_mobile, sd = step_sd)
   pop$y[mobile] <- pop$y[mobile] + stats::rnorm(n_mobile, sd = step_sd)
+  apply_boundary(pop, boundary)
+}
```

**Target (gold region)**

```r
  apply_boundary(pop, boundary)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/movement.R
#' Move agents by a Gaussian random walk
#'
#' Each agent takes an independent step with normally distributed x and y
#' displacements. Removed agents do not move.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param step_sd Standard deviation of the step along each axis.
#' @param boundary Boundary condition: `"wrap"`, `"reflect"` or `"clamp"`.
#' @return The population after one movement step.
#' @export
move_random <- function(pop, step_sd = 1, boundary = "wrap") {
  mobile <- pop$state != "R"
  n_mobile <- sum(mobile)
  if (n_mobile == 0) {
    return(pop)
  }

  pop$x[mobile] <- pop$x[mobile] + stats::rnorm(n_mobile, sd = step_sd)
  pop$y[mobile] <- pop$y[mobile] + stats::rnorm(n_mobile, sd = step_sd)
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

## 2. `38e3c80569c493b5c96896ed` finish_block / constrained_short

- package `synth:glyphgauge`, path `R/extract.R`
- generator: `{"behaviour": "checked", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["base", "sub"], "novelty": 2, "rule": "fb_statement", "seed": "b10-text/glyphgauge/R/extract.R", "signature_region": false, "structural_support": 0.0, "target_lines": 6, "test_file": false, "unit": "extractUrls", "variant": "outline"}`
- verification: `{"air_idempotent": true, "completion_tokens": 55, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 266, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -14,3 +14,9 @@
 #' @export
 extractUrls <- function(text, unique = TRUE) {
   hits <- extractPattern(text, extractPatterns()$url, unique = FALSE)
+  hits <- lapply(hits, function(h) sub("[.,;:!?]+$", "", h))
+  if (unique) {
+    hits <- lapply(hits, base::unique)
+  }
+  hits
+}
```

**Target (gold region)**

```r
  hits <- lapply(hits, function(h) sub("[.,;:!?]+$", "", h))
  if (unique) {
    hits <- lapply(hits, base::unique)
  }
  hits
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/extract.R
# ---- outline --------------------------------
# fn: extractUrls(text, unique)
# plan: 4 top-level statements; 1 typed above the cursor
# docs: yes
# ----------------------------------------------
#' @title Extract web addresses
#'
#' @description Trailing punctuation such as a full stop at the end of a
#'   sentence is trimmed from each match.
#'
#' @param text A character vector to search.
#' @param unique Logical; drop repeated matches within each string?
#' @return A list of character vectors, one element per string in `text`.
#' @export
extractUrls <- function(text, unique = TRUE) {
  hits <- extractPattern(text, extractPatterns()$url, unique = FALSE)
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

## 3. `6ef973882deccee9262b502d` finish_block / constrained_short

- package `synth:qcwatch`, path `R/chart-plot.R`
- generator: `{"behaviour": "file_checked_unit_unexercised", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b24-modelling/qcwatch/R/chart-plot.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "plot.qc_ewma", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 11, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1596, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -162,3 +162,5 @@
   graphics::points(index, x$x, pch = 1, col = "grey70", cex = 0.6)
   draw.limits(index, x$target, x$lcl, x$ucl)
   mark.signals(index, x$stat, x$signals)
+  invisible(x)
+}
```

**Target (gold region)**

```r
  invisible(x)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/chart-plot.R
#' @return A numeric vector of length two.
#' @noRd
chart.ylim <- function(values, lcl, ucl, pad = 0.05) {
  rng <- range(c(values, lcl, ucl), na.rm = TRUE, finite = TRUE)
  width <- diff(rng)
  # A flat chart still needs some vertical room.
  if (width == 0) {
    width <- max(abs(rng[1L]), 1)
  }
  c(rng[1L] - pad * width, rng[2L] + pad * width)
}

#' Draw center line and control limits
#'
#' Adds the center line and step-shaped control limits to an open plot.
#'
#' @param index Integer vector of time points.
#' @param center Center line.
#' @param lcl,ucl Lower and upper control limits.
#' @param col Colour of the limit lines.
#' @noRd
draw.limits <- function(index, center, lcl, ucl, col = "firebrick") {
  graphics::abline(h = center, col = "grey40", lty = 1)

  # Constant limits are drawn across the whole plot, varying ones as steps.
  if (length(unique(ucl)) == 1L) {
    graphics::abline(h = ucl[1L], col = col, lty = 2)
  } else {
    graphics::lines(index, ucl, type = "s", col = col, lty = 2)
  }

  if (length(unique(lcl)) == 1L) {
    graphics::abline(h = lcl[1L], col = col, lty = 2)
  } else {
    graphics::lines(index, lcl, type = "s", col = col, lty = 2)
  }
  invisible(NULL)
}

#' Highlight signalling points
#'
#' Overplots the points that triggered an alarm in a contrasting colour.
#'
#' @param index Integer vector of time points.
#' @param values Numeric vector of plotted statistics.
#' @param signals Integer vector of signalling positions.
#' @param col Colour of the highlighted points.
#' @noRd
mark.signals <- function(index, values, signals, col = "firebrick") {
  if (length(signals) == 0L) {
    return(invisible(NULL))
  }
  graphics::points(
    index[signals],
    values[signals],
    pch = 19,
    col = col,
    cex = 1.2
  )
  invisible(NULL)
}

#' Plot a Shewhart chart
#'
#' Draws the statistic over time with its center line and control limits and
#' highlights out-of-control points.
#'
#' @param x An object of class `qc_shewhart`.
#' @param main Plot title; defaults to the chart type.
#' @param ... Passed to [graphics::plot()].
#' @return `x`, invisibly.
#' @export
plot.qc_shewhart <- function(x, main = NULL, ...) {
  index <- seq_along(x$stat)
  if (is.null(main)) {
    main <- sprintf("%s chart", toupper(x$type))
  }

  ylim <- chart.ylim(x$stat, x$lcl, x$ucl)
  graphics::plot(
    index,
    x$stat,
    type = "b",
    pch = 20,
    ylim = ylim,
    xlab = "Subgroup",
    ylab = "Statistic",
    main = main,
    ...
  )
  draw.limits(index, x$center, x$lcl, x$ucl)
  mark.signals(index, x$stat, x$signals)
  invisible(x)
}

#' Plot a CUSUM chart
#'
#' Draws the upper sum above zero and the lower sum mirrored below zero, with
#' the decision interval on both sides.
#'
#' @param x An object of class `qc_cusum`.
#' @param main Plot title; defaults to the chart type.
#' @param ... Passed to [graphics::plot()].
#' @return `x`, invisibly.
#' @export
plot.qc_cusum <- function(x, main = NULL, ...) {
  index <- seq_along(x$upper)
  if (is.null(main)) {
    main <- "Tabular CUSUM"
  }

  # Mirror the lower sum below zero so both sides share one panel.
  lower <- -x$lower
  ylim <- chart.ylim(c(x$upper, lower), -x$h, x$h)
  graphics::plot(
    index,
    x$upper,
    type = "b",
    pch = 20,
    ylim = ylim,
    xlab = "Observation",
    ylab = "Cumulative sum",
    main = main,
    ...
  )
  graphics::lines(index, lower, type = "b", pch = 20, col = "grey30")
  draw.limits(index, 0, -x$h, x$h)
  mark.signals(index, x$upper, x$signals.upper)
  mark.signals(index, lower, x$signals.lower)
  invisible(x)
}

#' Plot an EWMA chart
#'
#' Draws the raw observations faintly and the EWMA statistic on top, with
#' time-varying control limits.
#'
#' @param x An object of class `qc_ewma`.
#' @param main Plot title; defaults to the chart type.
#' @param ... Passed to [graphics::plot()].
#' @return `x`, invisibly.
#' @export
plot.qc_ewma <- function(x, main = NULL, ...) {
  index <- seq_along(x$stat)
  if (is.null(main)) {
    main <- sprintf("EWMA chart (lambda = %g)", x$lambda)
  }

  ylim <- chart.ylim(x$stat, x$lcl, x$ucl)
  graphics::plot(
    index,
    x$stat,
    type = "b",
    pch = 20,
    ylim = ylim,
    xlab = "Observation",
    ylab = "EWMA",
    main = main,
    ...
  )
  # Raw data in the background for context.
  graphics::points(index, x$x, pch = 1, col = "grey70", cex = 0.6)
  draw.limits(index, x$target, x$lcl, x$ucl)
  mark.signals(index, x$stat, x$signals)
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

## 4. `2a89991cd9e9ea8908aac0ac` finish_block / constrained_short

- package `synth:trophicstan`, path `R/standata.R`
- generator: `{"behaviour": "checked", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b08-rcpp-science/trophicstan/R/standata.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "check_counts", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 16, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 271, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -18,3 +18,5 @@
     )
   }
 
+  return(as.integer(x))
+}
```

**Target (gold region)**

```r
  return(as.integer(x))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/standata.R
#' Check a vector of non-negative counts
#'
#' @param x Vector to check.
#' @param arg Name of the argument, used in error messages.
#' @return `x` as an integer vector, if all checks pass.
#' @noRd
check_counts <- function(x, arg) {
  if (!is.numeric(x) || anyNA(x)) {
    stop(
      sprintf("`%s` must be a numeric vector without missing values.", arg),
      call. = FALSE
    )
  }
  if (any(x < 0) || any(x != round(x))) {
    stop(
      sprintf("`%s` must contain non-negative whole numbers.", arg),
      call. = FALSE
    )
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

## 5. `e5966b5b95baf0ce4de7d33d` finish_block / constrained_short

- package `synth:strataq`, path `R/collect.R`
- generator: `{"behaviour": "no_check", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["n_distinct"], "novelty": 1, "rule": "fb_statement", "seed": "b12-db/strataq/R/collect.R", "signature_region": false, "structural_support": 0.455, "target_lines": 5, "test_file": false, "unit": "countDistinctRemote", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 38, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1105, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -117,3 +117,8 @@
 #' @export
 countDistinctRemote <- function(lazy, column) {
   col <- sym(column)
+  res <- lazy %>%
+    summarise(n = n_distinct(!!col)) %>%
+    collect()
+  as.numeric(res$n[1])
+}
```

**Target (gold region)**

```r
  res <- lazy %>%
    summarise(n = n_distinct(!!col)) %>%
    collect()
  as.numeric(res$n[1])
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/collect.R
#' Collect a lazy table with a row limit
#'
#' Collects at most `maxRows` rows and warns when the result was cut off,
#' so that an accidental full-table pull is noticed.
#'
#' @param lazy A `tbl_lazy` object.
#' @param maxRows Maximum number of rows to fetch.
#' @return A tibble.
#' @importFrom dplyr collect
#' @importFrom utils head
#' @importFrom magrittr %>%
#' @export
collectCapped <- function(lazy, maxRows = 1e5) {
  # fetch one extra row so we can tell a full result from a cut-off one
  out <- lazy %>%
    head(maxRows + 1) %>%
    collect()

  if (nrow(out) > maxRows) {
    warning(
      "Result truncated to ",
      format(maxRows, big.mark = ","),
      " rows.",
      call. = FALSE
    )
    out <- out[seq_len(maxRows), ]
  }
  out
}

#' Collect a lazy table and time the query
#'
#' Collects `lazy` and attaches the elapsed wall time, in seconds, as the
#' `"elapsed"` attribute of the result.
#'
#' @param lazy A `tbl_lazy` object.
#' @importFrom dplyr collect
#' @export
collectTimed <- function(lazy) {
  started <- proc.time()[["elapsed"]]
  out <- collect(lazy)
  attr(out, "elapsed") <- proc.time()[["elapsed"]] - started
  out
}

#' Collect a lazy table in chunks
#'
#' Pages through `lazy` ordered by `orderCol`, fetching `chunkSize` rows at
#' a time, and calls `callback` on each chunk.
#'
#' @param lazy A `tbl_lazy` object.
#' @param orderCol Name of the ordering column, as a string.
#' @param chunkSize Number of rows per chunk.
#' @param callback A function called with each chunk and its index.
#' @return The total number of rows processed, invisibly.
#' @importFrom dplyr collect arrange
#' @importFrom rlang sym
#' @export
collectChunked <- function(lazy, orderCol, chunkSize = 10000L, callback) {
  callback <- match.fun(callback)
  # paste() would render 1e5 as "1e+05", which no database accepts
  chunkSize <- as.integer(chunkSize)
  if (is.na(chunkSize) || chunkSize < 1L) {
    stop("`chunkSize` must be a positive whole number.", call. = FALSE)
  }

  # a stable order is what makes LIMIT/OFFSET paging correct
  ordered <- lazy %>% arrange(!!sym(orderCol))
  total <- 0L
  chunk <- 0L

  repeat {
    sql <- paste(
      dbplyr::sql_render(ordered),
      "LIMIT",
      chunkSize,
      "OFFSET",
      total
    )
    page <- DBI::dbGetQuery(dbplyr::remote_con(lazy), sql)
    if (nrow(page) == 0) {
      break
    }

    chunk <- chunk + 1L
    callback(page, chunk)
    total <- total + nrow(page)
    if (nrow(page) < chunkSize) {
      break
    }
  }

  invisible(total)
}

#' Count rows or distinct values of a lazy table on the server
#'
#' `countRemote()` runs `COUNT(*)` and `countDistinctRemote()` runs
#' `COUNT(DISTINCT col)` remotely; both return a plain number instead of a
#' one-row tibble.
#'
#' @param lazy A `tbl_lazy` object.
#' @return A single number.
#' @importFrom dplyr summarise collect n
#' @export
countRemote <- function(lazy) {
  res <- lazy %>%
    summarise(n = n()) %>%
    collect()
  as.numeric(res$n[1])
}

#' @rdname countRemote
#' @param column Name of the column, as a string.
#' @importFrom dplyr n_distinct
#' @importFrom rlang sym
#' @export
countDistinctRemote <- function(lazy, column) {
  col <- sym(column)
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

## 6. `19428fdb9c4ca9350e882753` finish_block / constrained_short

- package `synth:quantia`, path `R/accessors.R`
- generator: `{"behaviour": "file_checked_unit_unexercised", "bucket": "short", "determinacy": "constrained_short", "full_body": true, "novel_tokens": [".Data"], "novelty": 1, "rule": "fb_statement", "seed": "b05-oop/quantia/R/accessors.R", "signature_region": true, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "setMethod", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 28, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 473, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -32,3 +32,5 @@
 
 #' @rdname magnitude
 setMethod("magnitude", "Quantity", function(object, ...) {
+  object@.Data
+})
```

**Target (gold region)**

```r
setMethod("magnitude", "Quantity", function(object, ...) {
  object@.Data
})
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/accessors.R
#' @title Get or set the magnitudes of a quantity
#' @description `magnitude()` returns the bare numeric values of a quantity,
#'   dropping its unit; `magnitude<-` replaces them and keeps the unit.
#' @param object A [Quantity-class] object.
#' @param ... Additional arguments for methods.
#' @return A numeric vector, or the modified object for the replacement form.
#' @export
setGeneric("magnitude", function(object, ...) standardGeneric("magnitude"))

#' @rdname magnitude
#' @param value A numeric vector of the same length as `object`.
#' @export
setGeneric("magnitude<-", function(object, value) {
  standardGeneric("magnitude<-")
})

#' @title Get or set the unit of a quantity
#' @description `unitOf<-` replaces the unit but keeps the magnitudes
#'   unchanged. Use [convertTo()] to change units while preserving the
#'   physical value.
#' @param object A [Quantity-class] object.
#' @param ... Additional arguments for methods.
#' @return A [Unit-class] object, or the modified object for the replacement
#'   form.
#' @export
setGeneric("unitOf", function(object, ...) standardGeneric("unitOf"))

#' @rdname unitOf
#' @param value A [Unit-class] object or a registered unit symbol.
#' @export
setGeneric("unitOf<-", function(object, value) standardGeneric("unitOf<-"))

#' @rdname magnitude
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
setMethod("magnitude", "Quantity", function(object, ...) {<|user_cursor|>
=======
<[fim-middle]>
````

</details>

## 7. `047c906974748836a470cbd8` finish_block / constrained_short

- package `synth:cellarlink`, path `R/sql_build.R`
- generator: `{"behaviour": "no_check", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b12-db/cellarlink/R/sql_build.R", "signature_region": false, "structural_support": 0.0, "target_lines": 3, "test_file": false, "unit": "build_select", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 14, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 400, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -19,3 +19,6 @@
   sql <- glue::glue_sql("SELECT {col_sql} FROM {tbl}", .con = con)
   if (length(where) > 0) {
     sql <- glue::glue_sql("{sql} WHERE {where_clause(con, where)}", .con = con)
+  }
+  return(sql)
+}
```

**Target (gold region)**

```r
  }
  return(sql)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/sql_build.R
# ---- diagnostics (static) --------------------
# typed statements: 3 of 5
# args not yet referenced: none
# unclosed braces in typed code: 2
# ----------------------------------------------
#' @title Build a SELECT statement
#' @description Builds a `SELECT` with optional equality filters. Filter
#'   values are interpolated safely with [glue::glue_sql()].
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @param cols A character vector of column names, or `NULL` for all columns.
#' @param where A named list of equality filters; names are columns.
#' @return An `SQL` string.
#' @export
build_select <- function(con, table, cols = NULL, where = list()) {
  tbl <- quote_table(con, table)
  col_sql <- if (is.null(cols)) DBI::SQL("*") else quote_cols(con, cols)

  sql <- glue::glue_sql("SELECT {col_sql} FROM {tbl}", .con = con)
  if (length(where) > 0) {
    sql <- glue::glue_sql("{sql} WHERE {where_clause(con, where)}", .con = con)
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

## 8. `d41f4900ecef1c552e850a81` finish_block / constrained_short

- package `synth:seqlattice`, path `R/sequence.R`
- generator: `{"behaviour": "file_checked_unit_unexercised", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["end", "start"], "novelty": 2, "rule": "fb_statement", "seed": "b08-rcpp-science/seqlattice/R/sequence.R", "signature_region": false, "structural_support": 0.0, "target_lines": 6, "test_file": false, "unit": "gc_profile", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 38, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 415, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -27,3 +27,9 @@
 
   gc <- gc_window_cpp(seq, width, step)
   starts <- seq.int(1L, by = step, length.out = length(gc))
+  return(data.frame(
+    start = starts,
+    end = starts + width - 1L,
+    gc = gc
+  ))
+}
```

**Target (gold region)**

```r
  return(data.frame(
    start = starts,
    end = starts + width - 1L,
    gc = gc
  ))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/sequence.R
# ---- diagnostics (static) --------------------
# typed statements: 7 of 8
# args not yet referenced: none
# unclosed braces in typed code: 1
# ----------------------------------------------
#' Sliding-window GC profile
#'
#' Splits a single sequence into windows and returns the GC fraction of each,
#' together with window coordinates.
#'
#' @param seq A single DNA sequence as a character string.
#' @param width Integer window width in bases.
#' @param step Integer step between window starts. Defaults to `width`, giving
#'   non-overlapping windows.
#' @return A data frame with columns `start`, `end` and `gc`.
#' @export
gc_profile <- function(seq, width = 1000L, step = width) {
  if (length(seq) != 1L) {
    stop("`seq` must be a single sequence.", call. = FALSE)
  }

  seq <- check_dna(seq, arg = "seq")
  len <- nchar(seq)
  width <- check_count(width, "width", max = len)
  # A step larger than the window would leave gaps between windows
  step <- check_count(step, "step", max = width)

  gc <- gc_window_cpp(seq, width, step)
  starts <- seq.int(1L, by = step, length.out = length(gc))
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

## 9. `24f001d8a9493ec81c887e73` finish_block / constrained_short

- package `synth:diffledger`, path `R/annotate.R`
- generator: `{"behaviour": "file_checked_unit_unexercised", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [".tidy_annotation"], "novelty": 1, "rule": "fb_statement", "seed": "b17-bioc/diffledger/R/annotate.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "read_annotation_tsv", "variant": "outline"}`
- verification: `{"air_idempotent": true, "completion_tokens": 19, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 287, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -18,3 +18,5 @@
     col_types = readr::cols(.default = "c"),
     progress = FALSE
   )
+  .tidy_annotation(annot, id_col)
+}
```

**Target (gold region)**

```r
  .tidy_annotation(annot, id_col)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/annotate.R
# ---- outline --------------------------------
# fn: read_annotation_tsv(path, id_col)
# plan: 2 top-level statements; 1 typed above the cursor
# docs: yes
# ----------------------------------------------
#' Read an annotation table
#'
#' Readers for tab- and comma-separated annotation files. Every column is
#' read as character so identifiers keep their leading zeros.
#'
#' @param path Path to the annotation file.
#' @param id_col Name of the identifier column.
#' @return A tibble with an `id` column.
#' @export
read_annotation_tsv <- function(path, id_col = "gene_id") {
  annot <- readr::read_tsv(
    path,
    col_types = readr::cols(.default = "c"),
    progress = FALSE
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

## 10. `2354b6ffe79a3f826e5ca94f` finish_block / constrained_short

- package `synth:orbitnet`, path `R/centrality.R`
- generator: `{"behaviour": "no_check", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b19-graphs-or/orbitnet/R/centrality.R", "signature_region": false, "structural_support": 0.0, "target_lines": 3, "test_file": false, "unit": "vertex_names", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 12, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 221, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -11,3 +11,6 @@
   nm <- igraph::V(g)$name
   if (is.null(nm)) {
     nm <- as.character(seq_len(igraph::vcount(g)))
+  }
+  nm
+}
```

**Target (gold region)**

```r
  }
  nm
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/centrality.R
# ---- diagnostics (static) --------------------
# typed statements: 1 of 3
# args not yet referenced: none
# unclosed braces in typed code: 2
# ----------------------------------------------
#' Vertex names with an index fallback
#'
#' @param g An `igraph` object.
#' @noRd
vertex_names <- function(g) {
  nm <- igraph::V(g)$name
  if (is.null(nm)) {
    nm <- as.character(seq_len(igraph::vcount(g)))
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

## 11. `238d4b2a16f39d644376c691` finish_block / constrained_short

- package `synth:shiftplan`, path `R/scheduling.R`
- generator: `{"behaviour": "checked", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["rownames"], "novelty": 1, "rule": "fb_statement", "seed": "b19-graphs-or/shiftplan/R/scheduling.R", "signature_region": false, "structural_support": 0.0, "target_lines": 3, "test_file": false, "unit": "evaluateSequence", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 19, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 298, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -19,3 +19,6 @@
   out$completion <- completion
   out$lateness <- completion - out$due
   out$tardiness <- pmax(out$lateness, 0)
+  rownames(out) <- NULL
+  return(out)
+}
```

**Target (gold region)**

```r
  rownames(out) <- NULL
  return(out)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/scheduling.R
evaluateSequence <- function(jobs, sequence) {
  jobs <- checkJobs(jobs)
  n <- length(sequence)
  start <- numeric(n)
  completion <- numeric(n)

  # The machine waits for a job's release but never idles otherwise
  clock <- 0
  for (k in seq_len(n)) {
    j <- sequence[k]
    start[k] <- max(clock, jobs$release[j])
    completion[k] <- start[k] + jobs$processing[j]
    clock <- completion[k]
  }

  out <- jobs[sequence, c("job", "processing", "due", "weight"), drop = FALSE]
  out$position <- seq_len(n)
  out$start <- start
  out$completion <- completion
  out$lateness <- completion - out$due
  out$tardiness <- pmax(out$lateness, 0)
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

## 12. `00177c51512525376f76c1d3` finish_block / constrained_short

- package `synth:cellarlink`, path `R/upsert.R`
- generator: `{"behaviour": "checked", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b12-db/cellarlink/R/upsert.R", "signature_region": false, "structural_support": 0.0, "target_lines": 3, "test_file": false, "unit": "upsert_batched", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 31, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1293, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -126,3 +126,6 @@
     }
   }
 
+  names(affected) <- paste0("batch_", seq_along(starts))
+  return(invisible(affected))
+}
```

**Target (gold region)**

```r
  names(affected) <- paste0("batch_", seq_along(starts))
  return(invisible(affected))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/upsert.R
#' @title Upsert rows using delete-then-insert
#' @description Deletes rows of `table` whose keys appear in `data`, then
#'   appends `data`, inside one transaction. Useful on backends without
#'   `ON CONFLICT`.
#' @param con A `DBIConnection` object.
#' @param table Name of the target table.
#' @param data A data frame of rows to write.
#' @param keys A character vector of key column names.
#' @return The number of rows affected, invisibly.
#' @export
upsert_replace <- function(con, table, data, keys) {
  check_keys(data, keys)
  stage <- write_staging(con, data)
  on.exit(DBI::dbRemoveTable(con, stage, temporary = TRUE), add = TRUE)

  join_on <- vapply(
    keys,
    function(k) {
      glue::glue_sql("t.{`k`} = s.{`k`}", .con = con)
    },
    character(1)
  ) |>
    paste(collapse = " AND ") |>
    DBI::SQL()

  n <- DBI::dbWithTransaction(con, {
    DBI::dbExecute(
      con,
      glue::glue_sql(
        "DELETE FROM {`table`} AS t WHERE EXISTS (SELECT 1 FROM {`stage`} AS s WHERE {join_on})",
        .con = con
      )
    )
    DBI::dbAppendTable(con, table, data)
  })
  return(invisible(n))
}

#' @title Insert only rows whose keys are new
#' @description Appends rows of `data` whose keys do not already exist in
#'   `table`; existing rows are left untouched.
#' @param con A `DBIConnection` object.
#' @param table Name of the target table.
#' @param data A data frame of rows to write.
#' @param keys A character vector of key column names.
#' @return The number of rows affected, invisibly.
#' @export
insert_new <- function(con, table, data, keys) {
  check_keys(data, keys)
  stage <- write_staging(con, data)
  on.exit(DBI::dbRemoveTable(con, stage, temporary = TRUE), add = TRUE)

  join_on <- vapply(
    keys,
    function(k) {
      glue::glue_sql("t.{`k`} = s.{`k`}", .con = con)
    },
    character(1)
  ) |>
    paste(collapse = " AND ") |>
    DBI::SQL()

  sql <- glue::glue_sql(
    "INSERT INTO {`table`} ({`names(data)`*})
     SELECT {`names(data)`*} FROM {`stage`} AS s
     WHERE NOT EXISTS (SELECT 1 FROM {`table`} AS t WHERE {join_on})",
    .con = con
  )
  n <- DBI::dbExecute(con, sql)
  return(invisible(n))
}

#' @title Upsert rows in batches
#' @description Splits `data` into chunks of `batch_size` rows and upserts
#'   each chunk with the chosen strategy, reporting progress as it goes.
#' @param con A `DBIConnection` object.
#' @param table Name of the target table.
#' @param data A data frame of rows to write.
#' @param keys A character vector of key column names.
#' @param batch_size Number of rows per batch.
#' @param method Upsert strategy, one of `"on_conflict"`, `"replace"` or
#'   `"insert_new"`.
#' @param verbose Whether to print a message after each batch.
#' @return A named integer vector of rows affected per batch, invisibly.
#' @export
upsert_batched <- function(
  con,
  table,
  data,
  keys,
  batch_size = 5000L,
  method = c("on_conflict", "replace", "insert_new"),
  verbose = TRUE
) {
  method <- match.arg(method)
  writer <- switch(
    method,
    on_conflict = upsert_on_conflict,
    replace = upsert_replace,
    insert_new = insert_new
  )
  if (batch_size < 1) {
    stop("`batch_size` must be at least 1.", call. = FALSE)
  }

  n_rows <- nrow(data)
  if (n_rows == 0) {
    return(invisible(integer()))
  }

  # check keys across the whole input; per-batch checks miss cross-batch duplicates
  check_keys(data, keys)
  starts <- seq(1L, n_rows, by = batch_size)
  affected <- integer(length(starts))
  for (i in seq_along(starts)) {
    idx <- starts[i]:min(starts[i] + batch_size - 1L, n_rows)
    affected[i] <- writer(con, table, data[idx, , drop = FALSE], keys)
    if (verbose) {
      message(sprintf(
        "[%s] batch %d/%d: %d rows",
        table,
        i,
        length(starts),
        affected[i]
      ))
    }
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

## 13. `a631be9f6f34728b3b909752` finish_block / constrained_short

- package `synth:parcelwise`, path `R/read.R`
- generator: `{"behaviour": "file_checked_unit_unexercised", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["st_transform"], "novelty": 1, "rule": "fb_statement", "seed": "b13-geo/parcelwise/R/read.R", "signature_region": false, "structural_support": 0.0, "target_lines": 5, "test_file": false, "unit": "finalize_layer", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 34, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 315, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -17,3 +17,8 @@
   }
 
   x <- sf::st_zm(x, drop = TRUE, what = "ZM")
+  if (!is.null(crs)) {
+    x <- sf::st_transform(x, crs)
+  }
+  x
+}
```

**Target (gold region)**

```r
  if (!is.null(crs)) {
    x <- sf::st_transform(x, crs)
  }
  x
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/read.R
#' @title Normalise a freshly read layer
#'
#' @description
#' Renames the active geometry column to `geometry`, drops Z/M dimensions and
#' optionally transforms to a target CRS.
#'
#' @param x An `sf` object.
#' @param crs Target coordinate reference system, or `NULL` to keep the source CRS.
#' @return An `sf` object with a geometry column named `geometry`.
#' @keywords internal
finalize_layer <- function(x, crs) {
  # Drivers disagree on the name (geom, wkb_geometry, ...); settle on one
  geom_col <- attr(x, "sf_column")
  if (geom_col != "geometry") {
    names(x)[names(x) == geom_col] <- "geometry"
    sf::st_geometry(x) <- "geometry"
  }

  x <- sf::st_zm(x, drop = TRUE, what = "ZM")
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

## 14. `d50ba25ad364dd9e5f0a30cf` finish_block / constrained_short

- package `synth:termsmith`, path `R/compare-models.R`
- generator: `{"behaviour": "checked", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b24-modelling/termsmith/R/compare-models.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "lr_sequence", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 10, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 443, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -35,3 +35,5 @@
   out <- do.call(rbind, rows)
   # Holm controls the family-wise error across the sequence of tests.
   out$p.adjusted <- stats::p.adjust(out$p.value, method = "holm")
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
<filename>R/compare-models.R
#' Sequential likelihood-ratio tests
#'
#' Tests each model in a list against the next one, as in a forward build-up
#' of nested models, and adjusts the p-values with Holm's method.
#'
#' @param models A list of nested fitted models, smallest first.
#' @return A data frame with one row per consecutive pair.
#' @export
lr_sequence <- function(models) {
  if (!is.list(models) || length(models) < 2L) {
    stop(
      "`models` must be a list of at least two fitted models.",
      call. = FALSE
    )
  }
  labels <- names(models)
  if (is.null(labels) || any(labels == "")) {
    labels <- paste0("model_", seq_along(models))
  }

  n_steps <- length(models) - 1L
  rows <- vector("list", n_steps)
  for (i in seq_len(n_steps)) {
    test <- lr_test(models[[i]], models[[i + 1L]])
    rows[[i]] <- data.frame(
      from = labels[i],
      to = labels[i + 1L],
      statistic = test$statistic,
      df = test$df,
      p.value = test$p.value,
      stringsAsFactors = FALSE
    )
  }

  out <- do.call(rbind, rows)
  # Holm controls the family-wise error across the sequence of tests.
  out$p.adjusted <- stats::p.adjust(out$p.value, method = "holm")
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

## 15. `665307d771b02eaf505bc03e` finish_block / constrained_short

- package `synth:newsreel`, path `R/gitlog.R`
- generator: `{"behaviour": "checked", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_comment_lead", "seed": "b22-devtools/newsreel/R/gitlog.R", "signature_region": false, "structural_support": 0.0, "target_lines": 3, "test_file": false, "unit": "summarise_by_author", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 38, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 345, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -21,3 +21,6 @@
     out$last[[i]] <- max(dates)
   }
 
+  # Most active authors first, ties broken alphabetically.
+  return(out[order(-out$commits, out$author), , drop = FALSE])
+}
```

**Target (gold region)**

```r
  # Most active authors first, ties broken alphabetically.
  return(out[order(-out$commits, out$author), , drop = FALSE])
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/gitlog.R
#' Summarise a commit log by author
#'
#' @param log A data frame of commits as returned by [git_log()].
#' @return A data frame with one row per author: `author`, `commits`,
#'   `first` and `last`, ordered by commit count.
#' @export
summarise_by_author <- function(log) {
  keys <- sort(unique(log$author))
  out <- data.frame(
    author = keys,
    commits = integer(length(keys)),
    first = as.POSIXct(rep(NA, length(keys)), tz = "UTC"),
    last = as.POSIXct(rep(NA, length(keys)), tz = "UTC"),
    stringsAsFactors = FALSE
  )

  for (i in seq_along(keys)) {
    dates <- log$date[log$author == keys[[i]]]
    out$commits[[i]] <- length(dates)
    out$first[[i]] <- min(dates)
    out$last[[i]] <- max(dates)
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

## 16. `1269e37beec3f12d839525a8` finish_block / constrained_short

- package `synth:clinic-access-survey`, path `analysis/02_access_and_population.R`
- generator: `{"behaviour": "no_check", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["population", "tibble"], "novelty": 2, "rule": "fb_statement", "seed": "b13-geo/clinic-access-survey/analysis/02_access_and_population.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "pop_within", "variant": "outline"}`
- verification: `{"air_idempotent": true, "completion_tokens": 26, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 230, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -14,3 +14,5 @@
     exact = TRUE,
     ID = FALSE
   )
+  tibble(!!id_col := zones[[id_col]], population = totals[[1]])
+}
```

**Target (gold region)**

```r
  tibble(!!id_col := zones[[id_col]], population = totals[[1]])
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>analysis/02_access_and_population.R
# ---- outline --------------------------------
# fn: pop_within(pop, zones, id_col)
# plan: 3 top-level statements; 2 typed above the cursor
# docs: no
# ----------------------------------------------
pop_within <- function(pop, zones, id_col) {
  # exact = TRUE weights border cells by the share inside each zone
  zones_v <- vect(zones)
  totals <- extract(
    pop,
    zones_v,
    fun = sum,
    na.rm = TRUE,
    exact = TRUE,
    ID = FALSE
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

## 17. `05fc9bbd67739e1703e60a89` finish_block / constrained_short

- package `synth:nestwalk`, path `R/registry.R`
- generator: `{"behaviour": "file_checked_unit_unexercised", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b16-classic/nestwalk/R/registry.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "print.nw_registry", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 13, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 316, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -21,3 +21,5 @@
   for (k in keys) {
     cat(sprintf("  %s: %s\n", k, class(get(k, envir = x))[1]))
   }
+  return(invisible(x))
+}
```

**Target (gold region)**

```r
  return(invisible(x))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/registry.R
# ---- diagnostics (static) --------------------
# typed statements: 3 of 4
# args not yet referenced: none
# unclosed braces in typed code: 1
# ----------------------------------------------
#' Print a registry
#'
#' @param x A registry created by [new_registry()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.nw_registry <- function(x, ...) {
  keys <- sort(ls(x, all.names = TRUE))
  cat(sprintf(
    "<nw_registry> %d entr%s\n",
    length(keys),
    if (length(keys) == 1L) "y" else "ies"
  ))

  # One line per entry showing only the class, never the value itself
  for (k in keys) {
    cat(sprintf("  %s: %s\n", k, class(get(k, envir = x))[1]))
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

## 18. `1e8321cf97c93e1a6dc14642` finish_block / constrained_short

- package `synth:taskrelay`, path `R/parallel.R`
- generator: `{"behaviour": "checked", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["workers"], "novelty": 1, "rule": "fb_statement", "seed": "b14-pipelines/taskrelay/R/parallel.R", "signature_region": false, "structural_support": 0.0, "target_lines": 6, "test_file": false, "unit": "detect_workers", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 52, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 410, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -30,3 +30,9 @@
 
   # Physical cores: hyperthreads rarely help the I/O-heavy jobs we run
   cores <- parallel::detectCores(logical = FALSE)
+  if (is.na(cores)) {
+    cores <- 1L
+  }
+  workers <- min(cores - reserve, max_workers)
+  return(as.integer(max(1L, workers)))
+}
```

**Target (gold region)**

```r
  if (is.na(cores)) {
    cores <- 1L
  }
  workers <- min(cores - reserve, max_workers)
  return(as.integer(max(1L, workers)))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/parallel.R
# ---- diagnostics (static) --------------------
# typed statements: 3 of 6
# args not yet referenced: max_workers, reserve
# unclosed braces in typed code: 1
# ----------------------------------------------
#' Decide how many workers to use
#'
#' Starts from the number of available cores, keeps `reserve` cores free
#' for the rest of the system and caps the result at `max_workers`. The
#' `TASKRELAY_WORKERS` environment variable overrides all of this.
#'
#' @export
#' @param max_workers Upper limit on the number of workers.
#' @param reserve Number of cores to leave idle.
#' @return A single positive integer.
detect_workers <- function(max_workers = Inf, reserve = 1L) {
  env_value <- Sys.getenv("TASKRELAY_WORKERS", "")
  if (nzchar(env_value)) {
    n <- suppressWarnings(as.integer(env_value))
    if (!is.na(n)) {
      return(max(1L, n))
    }
    warning(
      "Ignoring non-numeric TASKRELAY_WORKERS='",
      env_value,
      "'.",
      call. = FALSE
    )
  }

  # Physical cores: hyperthreads rarely help the I/O-heavy jobs we run
  cores <- parallel::detectCores(logical = FALSE)
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

## 19. `1de547ec367b3ad910b89726` finish_block / constrained_short

- package `synth:millrace_catchment`, path `analysis/01_clean_stations.R`
- generator: `{"behaviour": "no_check", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["bind_rows"], "novelty": 1, "rule": "fb_nested_close", "seed": "b20-environment/millrace_catchment/analysis/01_clean_stations.R", "signature_region": false, "structural_support": 0.0, "target_lines": 4, "test_file": false, "unit": "station_year_summary", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 16, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 517, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -42,3 +42,7 @@
       filled_share = filled_share,
       status = status
     )
+  }
+
+  bind_rows(rows)
+}
```

**Target (gold region)**

```r
  }

  bind_rows(rows)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>analysis/01_clean_stations.R
station_year_summary <- function(df, min_coverage = 0.9) {
  df <- df %>% mutate(year = year(date))
  keys <- df %>% distinct(station, year) %>% arrange(station, year)
  rows <- vector("list", nrow(keys))

  for (i in seq_len(nrow(keys))) {
    sub <- df %>% filter(station == keys$station[i], year == keys$year[i])
    n_days <- nrow(sub)

    coverage <- c(
      tmin = mean(!is.na(sub$tmin)),
      tmax = mean(!is.na(sub$tmax)),
      precip = mean(!is.na(sub$precip))
    )
    # A year is only "complete" when every key variable is well covered
    if (all(coverage >= min_coverage)) {
      status <- "complete"
    } else if (any(coverage >= min_coverage)) {
      status <- "partial"
    } else {
      status <- "sparse"
    }

    filled_share <- if ("tmax_source" %in% names(sub)) {
      mean(!is.na(sub$tmax_source) & sub$tmax_source != "observed")
    } else {
      NA_real_
    }

    # Precipitation totals with gaps would be biased low, so drop them
    rows[[i]] <- tibble(
      station = keys$station[i],
      year = keys$year[i],
      n_days = n_days,
      tmin_mean = mean(sub$tmin, na.rm = TRUE),
      tmax_mean = mean(sub$tmax, na.rm = TRUE),
      precip_total = if (coverage[["precip"]] >= min_coverage) {
        sum(sub$precip, na.rm = TRUE)
      } else {
        NA_real_
      },
      filled_share = filled_share,
      status = status
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

## 20. `2e684f0b72e6fd733a5bf146` finish_block / constrained_short

- package `synth:pathwren`, path `R/strings.R`
- generator: `{"behaviour": "checked", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b06-infra-tests/pathwren/R/strings.R", "signature_region": false, "structural_support": 0.643, "target_lines": 8, "test_file": false, "unit": "strToCamel", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 53, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1187, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -106,3 +106,11 @@
       rest <- w[-1L]
       rest <- paste0(toupper(substr(rest, 1L, 1L)), substring(rest, 2L))
       return(paste0(c(w[1L], rest), collapse = ""))
+    },
+    character(1)
+  )
+
+  # strsplit() turns NA into the literal word "NA"; put the NAs back
+  out[is.na(x)] <- NA_character_
+  return(out)
+}
```

**Target (gold region)**

```r
    },
    character(1)
  )

  # strsplit() turns NA into the literal word "NA"; put the NAs back
  out[is.na(x)] <- NA_character_
  return(out)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/strings.R
#' @title Squish whitespace in strings
#' @description Trims leading and trailing whitespace and collapses internal
#'   runs of whitespace to a single space.
#' @param x A character vector.
#' @return A character vector the same length as `x`.
#' @export
#' @examples
#' strSquish("  too   many\tspaces ")
strSquish <- function(x) {
  checkCharacter(x)
  out <- gsub("[[:space:]]+", " ", x)
  out <- trimws(out)
  return(out)
}

#' @title Pad strings to a fixed width
#' @description `strPadLeft()` adds padding on the left so text is
#'   right-aligned; `strPadRight()` adds padding on the right so text is
#'   left-aligned. Strings already wider than `width` are returned unchanged.
#' @param x A character vector.
#' @param width The target display width.
#' @param pad A single character used for padding.
#' @return A character vector the same length as `x`.
#' @export
#' @examples
#' strPadLeft(c("7", "42", "512"), 4, pad = "0")
strPadLeft <- function(x, width, pad = " ") {
  checkCharacter(x)
  checkPadChar(pad)

  # Display width, so wide characters count double
  missingWidth <- pmax(width - nchar(x, type = "width"), 0L)
  out <- paste0(strrep(pad, missingWidth), x)
  out[is.na(x)] <- NA_character_
  return(out)
}

#' @rdname strPadLeft
#' @export
strPadRight <- function(x, width, pad = " ") {
  checkCharacter(x)
  checkPadChar(pad)

  missingWidth <- pmax(width - nchar(x, type = "width"), 0L)
  out <- paste0(x, strrep(pad, missingWidth))
  out[is.na(x)] <- NA_character_
  return(out)
}

#' @title Truncate strings to a maximum length
#' @description Strings longer than `width` characters are cut and finished
#'   with `ellipsis` so that the result is exactly `width` characters long.
#' @param x A character vector.
#' @param width The maximum number of characters to keep.
#' @param ellipsis The string appended to truncated values.
#' @return A character vector the same length as `x`.
#' @export
strTruncate <- function(x, width, ellipsis = "...") {
  checkCharacter(x)
  if (width < nchar(ellipsis)) {
    rlang::abort(
      "`width` must be at least as wide as `ellipsis`.",
      class = c("pathwren_error_width", "pathwren_error")
    )
  }

  tooLong <- !is.na(x) & nchar(x) > width
  keep <- width - nchar(ellipsis)
  x[tooLong] <- paste0(substr(x[tooLong], 1L, keep), ellipsis)
  return(x)
}

#' @title Convert between snake_case and camelCase
#' @description `strToSnake()` lower-cases words and joins them with
#'   underscores. `strToCamel()` capitalises every word but the first and
#'   removes separators. Both treat spaces, dots, hyphens and underscores as
#'   word separators.
#' @param x A character vector.
#' @return A character vector the same length as `x`.
#' @export
#' @examples
#' strToSnake("monthlyNetTotal")
#' strToCamel("monthly_net_total")
strToSnake <- function(x) {
  checkCharacter(x)
  # Split camel humps first, then normalise every other separator
  out <- gsub("([a-z0-9])([A-Z])", "\\1_\\2", x)
  out <- gsub("[[:space:].-]+", "_", out)
  out <- gsub("_+", "_", out)
  out <- tolower(gsub("^_|_$", "", out))
  return(out)
}

#' @rdname strToSnake
#' @export
strToCamel <- function(x) {
  checkCharacter(x)
  words <- strsplit(strToSnake(x), "_", fixed = TRUE)

  out <- vapply(
    words,
    function(w) {
      if (length(w) == 0L) {
        return("")
      }
      rest <- w[-1L]
      rest <- paste0(toupper(substr(rest, 1L, 1L)), substring(rest, 2L))
      return(paste0(c(w[1L], rest), collapse = ""))
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

## 21. `8e61ed50682172e8065a238b` finish_block / constrained_short

- package `synth:tallyhook`, path `R/zzz.R`
- generator: `{"behaviour": "checked", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b06-infra-tests/tallyhook/R/zzz.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "tallyhook_max_retries", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 11, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 194, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -7,3 +7,5 @@
 tallyhook_max_retries <- function() {
   value <- tallyhook_opt("max_retries")
   value <- check_count(value, arg = "tallyhook.max_retries")
+  return(value)
+}
```

**Target (gold region)**

```r
  return(value)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/zzz.R
# ---- diagnostics (static) --------------------
# typed statements: 2 of 3
# args not yet referenced: none
# unclosed braces in typed code: 1
# ----------------------------------------------
#' @rdname option-getters
tallyhook_max_retries <- function() {
  value <- tallyhook_opt("max_retries")
  value <- check_count(value, arg = "tallyhook.max_retries")
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

## 22. `a00a7a0242dca722cb3c16b2` finish_block / constrained_short

- package `synth:tabulora`, path `R/base_plots.R`
- generator: `{"behaviour": "checked", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b07-plotting/tabulora/R/base_plots.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "legend_outside", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 11, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 315, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -23,3 +23,5 @@
     bty = "n",
     ...
   )
+  invisible(res)
+}
```

**Target (gold region)**

```r
  invisible(res)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/base_plots.R
#' Draw a legend outside the plot region
#'
#' Places a legend to the right of the plotting region, temporarily
#' allowing drawing in the margin.
#'
#' @param legend Character vector of legend labels.
#' @param col Colours matching `legend`.
#' @param ... Passed to [graphics::legend()].
#' @keywords internal
legend_outside <- function(legend, col, ...) {
  old <- graphics::par(xpd = NA)
  on.exit(graphics::par(old), add = TRUE)

  # Anchor the legend's top-left corner at the plot's top-right corner
  usr <- graphics::par("usr")
  res <- graphics::legend(
    x = usr[2],
    y = usr[4],
    legend = legend,
    col = col,
    xjust = 0,
    yjust = 1,
    bty = "n",
    ...
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

## 23. `0a673d53b49b273dc39c8b8d` finish_block / sibling:short

- package `synth:numquill`, path `R/optimise.R`
- generator: `{"behaviour": "checked", "bucket": "short", "determinacy": "sibling", "full_body": false, "novel_tokens": ["diag", "rate", "shape"], "novelty": 3, "rule": "fb_statement", "seed": "b02-base-stats/numquill/R/optimise.R", "signature_region": false, "structural_support": 0.526, "target_lines": 8, "test_file": false, "unit": "mle.gamma", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 112, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1547, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -153,3 +153,11 @@
   m <- mean(x)
   v <- stats::var(x)
   # Method-of-moments starting values.
+  start <- c(log(m^2 / v), log(m / v))
+
+  fit <- optim.safe(start, nll, method = "BFGS")
+  est <- c(shape = exp(fit$par[1]), rate = exp(fit$par[2]))
+  vc <- solve(num.hessian(nll, fit$par))
+  se <- est * sqrt(diag(vc))
+  list(estimate = est, se = se, loglik = -fit$value)
+}
```

**Target (gold region)**

```r
  start <- c(log(m^2 / v), log(m / v))

  fit <- optim.safe(start, nll, method = "BFGS")
  est <- c(shape = exp(fit$par[1]), rate = exp(fit$par[2]))
  vc <- solve(num.hessian(nll, fit$par))
  se <- est * sqrt(diag(vc))
  list(estimate = est, se = se, loglik = -fit$value)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/optimise.R
#' Central-difference derivatives
#'
#' `num.gradient()` returns the gradient and `num.hessian()` the Hessian of
#' `fn` at `x`, both by symmetric finite differences.
#'
#' @param fn Objective function of a numeric vector.
#' @param x Point at which to evaluate the derivative.
#' @param h Step size.
#' @return A numeric vector of partial derivatives.
#' @export
num.gradient <- function(fn, x, h = 1e-5) {
  p <- length(x)
  grad <- numeric(p)

  for (j in seq_len(p)) {
    e <- replace(numeric(p), j, h)
    grad[j] <- (fn(x + e) - fn(x - e)) / (2 * h)
  }
  grad
}

#' @rdname num.gradient
#' @export
num.hessian <- function(fn, x, h = 1e-4) {
  p <- length(x)
  hess <- matrix(0, p, p)

  # Only the upper triangle is evaluated; the lower one is mirrored.
  for (i in seq_len(p)) {
    for (j in i:p) {
      ei <- replace(numeric(p), i, h)
      ej <- replace(numeric(p), j, h)
      hess[i, j] <- (fn(x + ei + ej) -
        fn(x + ei - ej) -
        fn(x - ei + ej) +
        fn(x - ei - ej)) /
        (4 * h^2)
      hess[j, i] <- hess[i, j]
    }
  }
  hess
}

#' Run optim() with uniform error handling
#'
#' Calls [stats::optim()] and converts errors and non-convergence into a
#' consistent result list.
#'
#' @param par Numeric vector of starting values.
#' @param fn Objective function to minimise.
#' @param method Optimisation method passed to [stats::optim()].
#' @param lower,upper Bounds used by `"L-BFGS-B"`.
#' @param ... Passed to `fn`.
#' @return A list with `par`, `value`, `converged` and `message`.
#' @export
optim.safe <- function(
  par,
  fn,
  method = c("BFGS", "Nelder-Mead", "L-BFGS-B"),
  lower = -Inf,
  upper = Inf,
  ...
) {
  method <- match.arg(method)

  # Only L-BFGS-B accepts bounds; optim() warns if others are given them.
  fit <- tryCatch(
    if (method == "L-BFGS-B") {
      stats::optim(par, fn, method = method, lower = lower, upper = upper, ...)
    } else {
      stats::optim(par, fn, method = method, ...)
    },
    error = function(e) e
  )

  if (inherits(fit, "error")) {
    return(list(
      par = rep(NA_real_, length(par)),
      value = NA_real_,
      converged = FALSE,
      message = conditionMessage(fit)
    ))
  }
  list(
    par = fit$par,
    value = fit$value,
    converged = fit$convergence == 0L,
    message = if (is.null(fit$message)) "" else fit$message
  )
}

#' Multi-start optimisation
#'
#' Runs [optim.safe()] from every row of `starts` and keeps the best
#' converged solution.
#'
#' @param starts Numeric matrix of starting values, one row per start.
#' @param fn Objective function to minimise.
#' @param ... Passed to [optim.safe()].
#' @return The best result list, with an extra `n.converged` element.
#' @export
multi.start <- function(starts, fn, ...) {
  starts <- as.matrix(starts)
  runs <- lapply(seq_len(nrow(starts)), function(i) {
    optim.safe(starts[i, ], fn, ...)
  })

  ok <- vapply(runs, function(r) isTRUE(r$converged), logical(1))
  if (!any(ok)) {
    stop("No start converged.", call. = FALSE)
  }

  values <- vapply(runs[ok], function(r) r$value, numeric(1))
  best <- runs[ok][[which.min(values)]]
  best$n.converged <- sum(ok)
  best
}

#' Maximum likelihood fits of common distributions
#'
#' Each fit optimises over log-transformed positive parameters, so the
#' optimiser is unconstrained, and returns delta-method standard errors on
#' the natural scale. `mle.normal()` estimates the mean and standard
#' deviation, `mle.gamma()` the shape and rate, and `mle.weibull()` the
#' shape and scale. Non-finite (and, for the positive distributions,
#' non-positive) observations are dropped.
#'
#' @param x Numeric vector of observations.
#' @return A list with `estimate`, `se` and `loglik`.
#' @export
mle.normal <- function(x) {
  x <- x[is.finite(x)]
  nll <- function(theta) {
    -sum(stats::dnorm(x, theta[1], exp(theta[2]), log = TRUE))
  }
  start <- c(mean(x), log(stats::sd(x)))

  fit <- optim.safe(start, nll, method = "BFGS")
  est <- c(mean = fit$par[1], sd = exp(fit$par[2]))
  vc <- solve(num.hessian(nll, fit$par))
  # Delta method: d sd / d log(sd) = sd.
  se <- c(mean = sqrt(vc[1, 1]), sd = est[["sd"]] * sqrt(vc[2, 2]))
  list(estimate = est, se = se, loglik = -fit$value)
}

#' @rdname mle.normal
#' @export
mle.gamma <- function(x) {
  x <- x[is.finite(x) & x > 0]
  nll <- function(theta) {
    -sum(stats::dgamma(x, exp(theta[1]), exp(theta[2]), log = TRUE))
  }
  m <- mean(x)
  v <- stats::var(x)
  # Method-of-moments starting values.
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

## 24. `1a4f2c883b3412d86239613d` finish_block / sibling:short

- package `synth:quantia`, path `R/convert.R`
- generator: `{"behaviour": "checked", "bucket": "short", "determinacy": "sibling", "full_body": false, "novel_tokens": ["str:See", "str:Unknown", "str:listUnits"], "novelty": 3, "rule": "fb_statement", "seed": "b05-oop/quantia/R/convert.R", "signature_region": false, "structural_support": 0.75, "target_lines": 5, "test_file": false, "unit": "lookupUnit", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 44, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 441, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -33,3 +33,8 @@
     stop("`symbol` must be a single string.", call. = FALSE)
   }
   unit <- get0(symbol, envir = .unitRegistry, inherits = FALSE)
+  if (is.null(unit)) {
+    stop("Unknown unit '", symbol, "'. See `listUnits()`.", call. = FALSE)
+  }
+  unit
+}
```

**Target (gold region)**

```r
  if (is.null(unit)) {
    stop("Unknown unit '", symbol, "'. See `listUnits()`.", call. = FALSE)
  }
  unit
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/convert.R
#' @title Register a unit
#' @description Adds a unit to the package registry so it can be referred to
#'   by symbol in [quantity()] and [convertTo()].
#' @param unit A [Unit-class] object.
#' @param overwrite Whether to replace an existing unit with the same symbol.
#' @return The unit, invisibly.
#' @export
registerUnit <- function(unit, overwrite = FALSE) {
  if (!is(unit, "Unit")) {
    stop("`unit` must be a Unit object.", call. = FALSE)
  }
  validObject(unit)

  sym <- unit@symbol
  if (!overwrite && exists(sym, envir = .unitRegistry, inherits = FALSE)) {
    stop(
      "Unit '",
      sym,
      "' is already registered; use `overwrite = TRUE`.",
      call. = FALSE
    )
  }
  assign(sym, unit, envir = .unitRegistry)
  invisible(unit)
}

#' @title Look up a registered unit
#' @param symbol The symbol of a registered unit.
#' @return A [Unit-class] object.
#' @export
lookupUnit <- function(symbol) {
  if (!is.character(symbol) || length(symbol) != 1L || is.na(symbol)) {
    stop("`symbol` must be a single string.", call. = FALSE)
  }
  unit <- get0(symbol, envir = .unitRegistry, inherits = FALSE)
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

## 25. `254e41ad1466287a27bf867c` finish_block / sibling:short

- package `synth:cellarlink`, path `R/sql_build.R`
- generator: `{"behaviour": "no_check", "bucket": "short", "determinacy": "sibling", "full_body": false, "novel_tokens": ["set_sql", "str:SET", "str:set"], "novelty": 3, "rule": "fb_statement", "seed": "b12-db/cellarlink/R/sql_build.R", "signature_region": false, "structural_support": 0.68, "target_lines": 8, "test_file": false, "unit": "build_update", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 74, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1226, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -98,3 +98,11 @@
     },
     character(1)
   )
+  set_sql <- DBI::SQL(paste(sets, collapse = ", "))
+
+  sql <- glue::glue_sql(
+    "UPDATE {tbl} SET {set_sql} WHERE {where_clause(con, where)}",
+    .con = con
+  )
+  return(sql)
+}
```

**Target (gold region)**

```r
  set_sql <- DBI::SQL(paste(sets, collapse = ", "))

  sql <- glue::glue_sql(
    "UPDATE {tbl} SET {set_sql} WHERE {where_clause(con, where)}",
    .con = con
  )
  return(sql)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/sql_build.R
#' @title Quote a vector of column names
#' @description Quotes each name with [DBI::dbQuoteIdentifier()] and joins
#'   them into a comma-separated list for use in a `SELECT` or `INSERT`
#'   clause.
#' @param con A `DBIConnection` object.
#' @param cols A character vector of column names.
#' @keywords internal
quote_cols <- function(con, cols) {
  if (length(cols) == 0) {
    stop("`cols` must contain at least one column name.", call. = FALSE)
  }
  quoted <- DBI::dbQuoteIdentifier(con, cols)
  return(DBI::SQL(paste(quoted, collapse = ", ")))
}

#' @title Quote a possibly schema-qualified table name
#' @description Splits `"schema.table"` on the first dot and builds a
#'   [DBI::Id()] so both parts are quoted separately.
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @keywords internal
quote_table <- function(con, table) {
  # anything with more than one dot is passed through as a plain name
  parts <- strsplit(table, ".", fixed = TRUE)[[1]]
  id <- if (length(parts) == 2) {
    DBI::Id(schema = parts[1], table = parts[2])
  } else {
    DBI::Id(table = table)
  }
  return(DBI::dbQuoteIdentifier(con, id))
}

#' @title Build a SELECT statement
#' @description Builds a `SELECT` with optional equality filters. Filter
#'   values are interpolated safely with [glue::glue_sql()].
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @param cols A character vector of column names, or `NULL` for all columns.
#' @param where A named list of equality filters; names are columns.
#' @return An `SQL` string.
#' @export
build_select <- function(con, table, cols = NULL, where = list()) {
  tbl <- quote_table(con, table)
  col_sql <- if (is.null(cols)) DBI::SQL("*") else quote_cols(con, cols)

  sql <- glue::glue_sql("SELECT {col_sql} FROM {tbl}", .con = con)
  if (length(where) > 0) {
    sql <- glue::glue_sql("{sql} WHERE {where_clause(con, where)}", .con = con)
  }
  return(sql)
}

#' @title Build a DELETE statement
#' @description Builds a `DELETE` restricted by equality filters. An empty
#'   `where` is refused so that a typo never wipes a whole table.
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @param where A named list of equality filters; names are columns.
#' @return An `SQL` string.
#' @export
build_delete <- function(con, table, where) {
  if (length(where) == 0) {
    stop("Refusing to build a DELETE without a WHERE clause.", call. = FALSE)
  }

  tbl <- quote_table(con, table)
  sql <- glue::glue_sql(
    "DELETE FROM {tbl} WHERE {where_clause(con, where)}",
    .con = con
  )
  return(sql)
}

#' @title Build an UPDATE statement
#' @description Builds an `UPDATE` setting the columns in `values`,
#'   restricted by equality filters.
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @param values A named list of new values; names are columns.
#' @param where A named list of equality filters; names are columns.
#' @return An `SQL` string.
#' @export
build_update <- function(con, table, values, where) {
  if (length(where) == 0) {
    stop("Refusing to build an UPDATE without a WHERE clause.", call. = FALSE)
  }
  if (
    length(values) == 0 || is.null(names(values)) || any(!nzchar(names(values)))
  ) {
    stop("`values` must be a non-empty, fully named list.", call. = FALSE)
  }

  tbl <- quote_table(con, table)
  sets <- vapply(
    names(values),
    function(col) {
      glue::glue_sql("{`col`} = {values[[col]]}", .con = con)
    },
    character(1)
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

## 26. `1665f9e5e6524142b0820bc7` finish_block / sibling:short

- package `synth:lintfern`, path `R/parseTools.R`
- generator: `{"behaviour": "checked", "bucket": "short", "determinacy": "sibling", "full_body": false, "novel_tokens": ["inner", "str:CALL", "syms"], "novelty": 3, "rule": "fb_statement", "seed": "b22-devtools/lintfern/R/parseTools.R", "signature_region": false, "structural_support": 0.522, "target_lines": 4, "test_file": false, "unit": "functionBodySymbols", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 52, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1318, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -115,3 +115,7 @@
 
   # Default values are `expr` children too; the body is always the last one.
   bodyId <- body$id[[nrow(body)]]
+  inner <- tokenDescendants(pd, bodyId)
+  syms <- inner$text[inner$token %in% c("SYMBOL", "SYMBOL_FUNCTION_CALL")]
+  unique(syms)
+}
```

**Target (gold region)**

```r
  inner <- tokenDescendants(pd, bodyId)
  syms <- inner$text[inner$token %in% c("SYMBOL", "SYMBOL_FUNCTION_CALL")]
  unique(syms)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/parseTools.R
#' Navigate the parse tree
#'
#' @description
#' Small helpers for walking the token table: `tokenChildren()` returns the
#' direct children of a node, `tokenDescendants()` everything below it and
#' `tokenParent()` the node one level up.
#'
#' @param pd A data frame of parse data as returned by [readSourceTokens()].
#' @param id The id of the node to start from.
#' @return Rows of `pd` in source order; `tokenParent()` returns a single row,
#'   or `NULL` at the top level.
#' @keywords internal
tokenChildren <- function(pd, id) {
  kids <- pd[pd$parent == id, , drop = FALSE]
  kids[order(kids$line1, kids$col1), , drop = FALSE]
}

#' @rdname tokenChildren
tokenDescendants <- function(pd, id) {
  found <- integer()
  frontier <- id

  # Breadth-first: each pass collects the next generation of ids.
  while (length(frontier) > 0L) {
    kids <- pd$id[pd$parent %in% frontier]
    found <- c(found, kids)
    frontier <- kids
  }

  out <- pd[pd$id %in% found, , drop = FALSE]
  out[order(out$line1, out$col1), , drop = FALSE]
}

#' @rdname tokenChildren
tokenParent <- function(pd, id) {
  parentId <- pd$parent[pd$id == id]
  # Top-level expressions have parent 0; comments may carry negative ids.
  if (length(parentId) == 0L || parentId <= 0L) {
    return(NULL)
  }
  pd[pd$id == parentId, , drop = FALSE]
}

#' Find function definitions in parse data
#'
#' Locates every `function` keyword and resolves the name it is assigned to,
#' if any. Anonymous functions get the name `NA`.
#'
#' @param pd A data frame of parse data as returned by [readSourceTokens()].
#' @return A data frame with columns `name`, `id`, `line1` and `line2`, where
#'   `id` is the expression node of the function definition.
#' @export
findFunctionDefs <- function(pd) {
  keywords <- pd[pd$token == "FUNCTION", , drop = FALSE]
  n <- nrow(keywords)
  out <- data.frame(
    name = rep(NA_character_, n),
    id = integer(n),
    line1 = integer(n),
    line2 = integer(n),
    stringsAsFactors = FALSE
  )

  for (i in seq_len(n)) {
    # The FUNCTION token sits inside the expression for the whole definition.
    fnExpr <- tokenParent(pd, keywords$id[[i]])
    out$id[[i]] <- fnExpr$id
    out$line1[[i]] <- fnExpr$line1
    out$line2[[i]] <- fnExpr$line2

    assign <- tokenParent(pd, fnExpr$id)
    if (is.null(assign)) {
      next
    }
    # Only `name <- function` and `name = function` give a name; the function
    # must be the right-hand side, not the target.
    siblings <- tokenChildren(pd, assign$id)
    ops <- c("LEFT_ASSIGN", "EQ_ASSIGN")
    isAssign <- nrow(siblings) == 3L && siblings$token[[2L]] %in% ops
    if (isAssign && siblings$id[[3L]] == fnExpr$id) {
      target <- tokenDescendants(pd, siblings$id[[1L]])
      sym <- target$text[target$token %in% c("SYMBOL", "STR_CONST")]
      if (length(sym) == 1L) {
        out$name[[i]] <- gsub("^[\"'`]|[\"'`]$", "", sym)
      }
    }
  }

  out
}

#' List the formal arguments of a function definition
#'
#' @param pd A data frame of parse data as returned by [readSourceTokens()].
#' @param fnId The id of a function definition node from [findFunctionDefs()].
#' @return A data frame of the `SYMBOL_FORMALS` tokens.
#' @export
functionFormals <- function(pd, fnId) {
  kids <- tokenChildren(pd, fnId)
  kids[kids$token == "SYMBOL_FORMALS", c("text", "line1", "col1"), drop = FALSE]
}

#' List the symbols used in a function body
#'
#' @param pd A data frame of parse data as returned by [readSourceTokens()].
#' @param fnId The id of a function definition node from [findFunctionDefs()].
#' @return A character vector of distinct symbol names.
#' @export
functionBodySymbols <- function(pd, fnId) {
  kids <- tokenChildren(pd, fnId)
  body <- kids[kids$token == "expr", , drop = FALSE]
  if (nrow(body) == 0L) {
    return(character())
  }

  # Default values are `expr` children too; the body is always the last one.
  bodyId <- body$id[[nrow(body)]]
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

## 27. `63939cb5fb58d2472f60d96a` finish_block / sibling:medium

- package `synth:rakewell`, path `R/design.R`
- generator: `{"behaviour": "file_checked_unit_unexercised", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["vartype"], "novelty": 1, "rule": "fb_nested_close", "seed": "b21-applied-analytics/rakewell/R/design.R", "signature_region": false, "structural_support": 0.568, "target_lines": 14, "test_file": false, "unit": "weighted_prop_by", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 79, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1324, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -133,3 +133,17 @@
   outcome <- stats::as.formula(paste0("~I(as.numeric(", variable, "))"))
   if (is.null(by)) {
     stat <- survey::svyciprop(outcome, design, method = "logit", level = level)
+  } else {
+    stat <- survey::svyby(
+      outcome,
+      as_rhs_formula(by),
+      design,
+      survey::svyciprop,
+      method = "logit",
+      level = level,
+      vartype = "se"
+    )
+  }
+
+  tidy_svystat(stat, level = level)
+}
```

**Target (gold region)**

```r
  } else {
    stat <- survey::svyby(
      outcome,
      as_rhs_formula(by),
      design,
      survey::svyciprop,
      method = "logit",
      level = level,
      vartype = "se"
    )
  }

  tidy_svystat(stat, level = level)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/design.R
#' Build a replicate-weight design from raked weights
#'
#' Converts the linearisation design from [design_from_weights()] into a
#' replicate design so that variance estimates reflect the raking step.
#'
#' @inheritParams design_from_weights
#' @param type Replication method passed to [survey::as.svrepdesign()].
#' @param replicates Number of bootstrap replicates when `type = "bootstrap"`.
#' @return A `svyrep.design` object.
#' @export
replicate_design <- function(
  data,
  weights,
  ids = NULL,
  strata = NULL,
  fpc = NULL,
  type = c("JK1", "JKn", "bootstrap"),
  replicates = 200L
) {
  type <- match.arg(type)
  base <- design_from_weights(
    data,
    weights,
    ids = ids,
    strata = strata,
    fpc = fpc
  )

  switch(
    type,
    JK1 = survey::as.svrepdesign(base, type = "JK1"),
    JKn = {
      if (is.null(strata)) {
        stop("`type = \"JKn\"` requires `strata`.", call. = FALSE)
      }
      survey::as.svrepdesign(base, type = "JKn")
    },
    bootstrap = survey::as.svrepdesign(
      base,
      type = "bootstrap",
      replicates = replicates
    )
  )
}

#' Tidy a survey statistic into a data frame
#'
#' Pulls estimates and standard errors out of an `svystat` or `svyby` object
#' and adds a normal-approximation confidence interval.
#'
#' @param stat An object returned by [survey::svymean()], [survey::svytotal()]
#'   or [survey::svyby()].
#' @param level Confidence level of the interval.
#' @return A data frame with columns `term`, `estimate`, `std_error`,
#'   `conf_low` and `conf_high`.
#' @keywords internal
tidy_svystat <- function(stat, level = 0.95) {
  estimate <- stats::coef(stat)
  se <- survey::SE(stat)
  z <- stats::qnorm(1 - (1 - level) / 2)

  out <- data.frame(
    term = names(estimate),
    estimate = unname(estimate),
    std_error = unname(as.numeric(se)),
    stringsAsFactors = FALSE
  )

  # Normal approximation; svyciprop() intervals are not reused here so all
  # three wrappers report the same kind of interval.
  out$conf_low <- out$estimate - z * out$std_error
  out$conf_high <- out$estimate + z * out$std_error
  rownames(out) <- NULL
  out
}

#' Weighted means, totals and proportions by group
#'
#' Thin wrappers around [survey::svyby()] with [survey::svymean()],
#' [survey::svytotal()] or [survey::svyciprop()] that take column names and
#' return a tidy data frame.
#'
#' @param design A survey design object.
#' @param variable Name of the outcome column.
#' @param by Character vector of grouping columns, or `NULL` for no grouping.
#' @param level Confidence level of the interval.
#' @return A data frame as returned by `tidy_svystat()`.
#' @export
weighted_mean_by <- function(design, variable, by = NULL, level = 0.95) {
  outcome <- as_rhs_formula(variable)
  if (is.null(by)) {
    stat <- survey::svymean(outcome, design, na.rm = TRUE)
  } else {
    stat <- survey::svyby(
      outcome,
      as_rhs_formula(by),
      design,
      survey::svymean,
      na.rm = TRUE,
      keep.var = TRUE
    )
  }

  tidy_svystat(stat, level = level)
}

#' @rdname weighted_mean_by
#' @export
weighted_total_by <- function(design, variable, by = NULL, level = 0.95) {
  outcome <- as_rhs_formula(variable)
  if (is.null(by)) {
    stat <- survey::svytotal(outcome, design, na.rm = TRUE)
  } else {
    stat <- survey::svyby(
      outcome,
      as_rhs_formula(by),
      design,
      survey::svytotal,
      na.rm = TRUE,
      keep.var = TRUE
    )
  }

  tidy_svystat(stat, level = level)
}

#' @rdname weighted_mean_by
#' @details `weighted_prop_by()` expects a logical or 0/1 outcome column and
#'   uses [survey::svyciprop()] with the logit method.
#' @export
weighted_prop_by <- function(design, variable, by = NULL, level = 0.95) {
  # svyciprop() needs a 0/1 numeric, so coerce logicals inside the formula.
  outcome <- stats::as.formula(paste0("~I(as.numeric(", variable, "))"))
  if (is.null(by)) {
    stat <- survey::svyciprop(outcome, design, method = "logit", level = level)
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

## 28. `0ef82d81cf15dbb950d9d16c` finish_block / sibling:medium

- package `synth:burrowlog`, path `R/messages.R`
- generator: `{"behaviour": "checked", "bucket": "medium", "determinacy": "sibling", "full_body": true, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b06-infra-tests/burrowlog/R/messages.R", "signature_region": true, "structural_support": 1.0, "target_lines": 9, "test_file": false, "unit": "burrow_warn", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 74, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 538, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -44,3 +44,12 @@
 
 #' @rdname burrow_inform
 burrow_warn <- function(message, .envir = parent.frame()) {
+  if (is_quiet()) {
+    return(invisible())
+  }
+  cli::cli_inform(
+    c("!" = message),
+    class = "burrowlog_message",
+    .envir = .envir
+  )
+}
```

**Target (gold region)**

```r
burrow_warn <- function(message, .envir = parent.frame()) {
  if (is_quiet()) {
    return(invisible())
  }
  cli::cli_inform(
    c("!" = message),
    class = "burrowlog_message",
    .envir = .envir
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/messages.R
#' Is burrowlog running quietly?
#'
#' Messages are suppressed when the `burrowlog.quiet` option is `TRUE` or
#' when the `BURROWLOG_QUIET` environment variable is set to `"true"`.
#'
#' @return `TRUE` or `FALSE`.
#' @keywords internal
is_quiet <- function() {
  # An explicit option always wins over the environment
  opt <- getOption("burrowlog.quiet")
  if (!is.null(opt)) {
    return(isTRUE(opt))
  }
  tolower(Sys.getenv("BURROWLOG_QUIET", "false")) %in% c("true", "1", "yes")
}

#' Emit burrowlog status messages
#'
#' Thin wrappers around cli that respect [is_quiet()] and tag every message
#' with a `burrowlog_message` class so callers can muffle them selectively.
#'
#' @param message A cli-formatted message, interpolated in the caller's frame.
#' @param .envir The environment used to interpolate `message`.
#' @return `NULL`, invisibly.
#' @keywords internal
burrow_inform <- function(message, .envir = parent.frame()) {
  if (is_quiet()) {
    return(invisible())
  }
  cli::cli_inform(message, class = "burrowlog_message", .envir = .envir)
}

#' @rdname burrow_inform
burrow_success <- function(message, .envir = parent.frame()) {
  if (is_quiet()) {
    return(invisible())
  }
  cli::cli_inform(
    c("v" = message),
    class = "burrowlog_message",
    .envir = .envir
  )
}

#' @rdname burrow_inform
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
burrow_warn <- function(message, .envir = parent.frame()) {<|user_cursor|>
=======
<[fim-middle]>
````

</details>

## 29. `2331867bc13a324554993d79` finish_block / sibling:medium

- package `synth:seqlattice`, path `R/sequence.R`
- generator: `{"behaviour": "file_checked_unit_unexercised", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["skew", "total"], "novelty": 2, "rule": "fb_statement", "seed": "b08-rcpp-science/seqlattice/R/sequence.R", "signature_region": false, "structural_support": 0.686, "target_lines": 13, "test_file": false, "unit": "gc_skew", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 109, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1038, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -97,3 +97,16 @@
 #' @export
 gc_skew <- function(seqs, na_if_empty = TRUE) {
   seqs <- check_dna(seqs)
+  counts <- base_counts_cpp(seqs, TRUE)
+  total <- counts[, "G"] + counts[, "C"]
+  skew <- (counts[, "G"] - counts[, "C"]) / total
+
+  if (na_if_empty) {
+    skew[total == 0] <- NA_real_
+  } else {
+    skew[total == 0] <- 0
+  }
+
+  names(skew) <- names(seqs)
+  return(skew)
+}
```

**Target (gold region)**

```r
  counts <- base_counts_cpp(seqs, TRUE)
  total <- counts[, "G"] + counts[, "C"]
  skew <- (counts[, "G"] - counts[, "C"]) / total

  if (na_if_empty) {
    skew[total == 0] <- NA_real_
  } else {
    skew[total == 0] <- 0
  }

  names(skew) <- names(seqs)
  return(skew)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/sequence.R
#' Check that input is a vector of DNA sequences
#'
#' Validates type and alphabet before sequences are handed to compiled code,
#' which assumes clean input and does no checking of its own.
#'
#' @param seqs A character vector of DNA sequences.
#' @param arg Name of the argument, used in error messages.
#' @return `seqs`, invisibly and upper-cased, if all checks pass.
#' @noRd
check_dna <- function(seqs, arg = "seqs") {
  if (!is.character(seqs)) {
    stop(
      sprintf("`%s` must be a character vector, not %s.", arg, class(seqs)[1]),
      call. = FALSE
    )
  }
  if (anyNA(seqs)) {
    stop(sprintf("`%s` must not contain missing values.", arg), call. = FALSE)
  }

  # The kernels only know upper-case bases; IUPAC codes other than N are
  # rejected rather than silently miscounted
  seqs <- toupper(seqs)
  bad <- grepl("[^ACGTN]", seqs)
  if (any(bad)) {
    stop(
      sprintf(
        "`%s` contains non-ACGTN characters at position(s) %s.",
        arg,
        paste(which(bad), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  return(invisible(seqs))
}

#' Check a positive integer scalar
#'
#' Used for window widths, step sizes and k-mer lengths before `.Call()`.
#'
#' @param x Value to check.
#' @param arg Name of the argument, used in error messages.
#' @param max Largest allowed value.
#' @noRd
check_count <- function(x, arg, max = .Machine$integer.max) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("`%s` must be a single number.", arg), call. = FALSE)
  }
  if (x < 1 || x != round(x)) {
    stop(sprintf("`%s` must be a positive whole number.", arg), call. = FALSE)
  }
  if (x > max) {
    stop(
      sprintf("`%s` must be at most %d.", arg, as.integer(max)),
      call. = FALSE
    )
  }

  return(as.integer(x))
}

#' GC content and GC skew of DNA sequences
#'
#' `gc_content()` computes the fraction of G and C bases in each sequence;
#' ambiguous `N` bases are excluded from the denominator. `gc_skew()`
#' computes `(G - C) / (G + C)`, a common signal for locating replication
#' origins in bacterial genomes.
#'
#' @param seqs A character vector of DNA sequences.
#' @param na_if_empty Logical; return `NA` for sequences with no called bases
#'   (no G or C bases for `gc_skew()`)?
#' @return A numeric vector, named like `seqs`.
#' @export
#' @examples
#' gc_content(c(a = "ACGT", b = "GGCCNN"))
gc_content <- function(seqs, na_if_empty = TRUE) {
  seqs <- check_dna(seqs)
  counts <- base_counts_cpp(seqs, TRUE)

  # N is counted by the kernel but is not a called base
  called <- rowSums(counts[, c("A", "C", "G", "T"), drop = FALSE])
  gc <- (counts[, "G"] + counts[, "C"]) / called

  if (na_if_empty) {
    gc[called == 0] <- NA_real_
  } else {
    gc[called == 0] <- 0
  }

  names(gc) <- names(seqs)
  return(gc)
}

#' @rdname gc_content
#' @export
gc_skew <- function(seqs, na_if_empty = TRUE) {
  seqs <- check_dna(seqs)
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

## 30. `7c1ec202ca3dee71a35b8b18` finish_block / sibling:medium

- package `synth:assayshelf`, path `R/accessors.R`
- generator: `{"behaviour": "checked", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b17-bioc/assayshelf/R/accessors.R", "signature_region": false, "structural_support": 1.0, "target_lines": 10, "test_file": false, "unit": "colInfo", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 76, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 960, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -103,3 +103,13 @@
     return(info)
   }
 
+  missing_cols <- setdiff(columns, names(info))
+  if (length(missing_cols) > 0L) {
+    stop(
+      "unknown colInfo columns: ",
+      paste(missing_cols, collapse = ", "),
+      call. = FALSE
+    )
+  }
+  info[, columns, drop = FALSE]
+}
```

**Target (gold region)**

```r
  missing_cols <- setdiff(columns, names(info))
  if (length(missing_cols) > 0L) {
    stop(
      "unknown colInfo columns: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  info[, columns, drop = FALSE]
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/accessors.R
#' @importFrom methods setGeneric setMethod validObject
NULL

#' Access assay matrices
#'
#' Generic for extracting a single assay from a container.
#'
#' @param x An `AssayShelf` object.
#' @param i Assay name or index; defaults to the first assay.
#' @param ... Passed to methods.
#' @return A numeric matrix.
#' @export
setGeneric("assay", function(x, i, ...) standardGeneric("assay"))

#' Replace an assay matrix
#'
#' Generic for storing a single assay in a container.
#'
#' @param x An `AssayShelf` object.
#' @param i Assay name or index; defaults to the first assay.
#' @param ... Passed to methods.
#' @param value A numeric matrix with the same dimensions as `x`.
#' @return The modified object.
#' @export
setGeneric("assay<-", function(x, i, ..., value) standardGeneric("assay<-"))

#' @rdname assay
#' @param withDimnames Should the returned matrix carry feature and sample
#'   names?
#' @export
setMethod("assay", "AssayShelf", function(x, i, withDimnames = TRUE, ...) {
  if (length(x@assays) == 0L) {
    stop("object contains no assays", call. = FALSE)
  }
  if (missing(i)) {
    i <- 1L
  }
  if (is.character(i) && !i %in% names(x@assays)) {
    stop(sprintf("no assay named '%s'", i), call. = FALSE)
  }

  out <- x@assays[[i]]
  # Dimnames live in the annotation tables, not in the stored matrices.
  if (withDimnames) {
    dimnames(out) <- list(rownames(x@rowInfo), rownames(x@colInfo))
  }
  out
})

#' @rdname assay-set
#' @export
setMethod("assay<-", "AssayShelf", function(x, i, ..., value) {
  if (missing(i)) {
    i <- if (length(x@assays) > 0L) names(x@assays)[1L] else "counts"
  }
  # A numeric index can only replace; new assays need a name.
  if (is.numeric(i) && i > length(x@assays)) {
    stop(
      "numeric index is out of range; use a name to add an assay",
      call. = FALSE
    )
  }

  value <- as.matrix(value)
  dimnames(value) <- NULL
  x@assays[[i]] <- value
  validObject(x)
  x
})

#' Get annotations
#'
#' Return the feature or sample annotation table, optionally restricted to
#' a subset of columns.
#'
#' @param x An `AssayShelf` object.
#' @param columns Optional character vector of columns to keep.
#' @return A data frame with one row per feature (`rowInfo()`) or per sample
#'   (`colInfo()`).
#' @export
rowInfo <- function(x, columns = NULL) {
  info <- x@rowInfo
  if (is.null(columns)) {
    return(info)
  }

  missing_cols <- setdiff(columns, names(info))
  if (length(missing_cols) > 0L) {
    stop(
      "unknown rowInfo columns: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  info[, columns, drop = FALSE]
}

#' @rdname rowInfo
#' @export
colInfo <- function(x, columns = NULL) {
  info <- x@colInfo
  if (is.null(columns)) {
    return(info)
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

## 31. `4ce66886e6b44c046e04abdc` finish_block / sibling:medium

- package `synth:assayshelf`, path `R/accessors.R`
- generator: `{"behaviour": "checked", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b17-bioc/assayshelf/R/accessors.R", "signature_region": false, "structural_support": 1.0, "target_lines": 14, "test_file": false, "unit": "colInfo", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 93, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 943, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -99,3 +99,17 @@
 #' @export
 colInfo <- function(x, columns = NULL) {
   info <- x@colInfo
+  if (is.null(columns)) {
+    return(info)
+  }
+
+  missing_cols <- setdiff(columns, names(info))
+  if (length(missing_cols) > 0L) {
+    stop(
+      "unknown colInfo columns: ",
+      paste(missing_cols, collapse = ", "),
+      call. = FALSE
+    )
+  }
+  info[, columns, drop = FALSE]
+}
```

**Target (gold region)**

```r
  if (is.null(columns)) {
    return(info)
  }

  missing_cols <- setdiff(columns, names(info))
  if (length(missing_cols) > 0L) {
    stop(
      "unknown colInfo columns: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  info[, columns, drop = FALSE]
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/accessors.R
#' @importFrom methods setGeneric setMethod validObject
NULL

#' Access assay matrices
#'
#' Generic for extracting a single assay from a container.
#'
#' @param x An `AssayShelf` object.
#' @param i Assay name or index; defaults to the first assay.
#' @param ... Passed to methods.
#' @return A numeric matrix.
#' @export
setGeneric("assay", function(x, i, ...) standardGeneric("assay"))

#' Replace an assay matrix
#'
#' Generic for storing a single assay in a container.
#'
#' @param x An `AssayShelf` object.
#' @param i Assay name or index; defaults to the first assay.
#' @param ... Passed to methods.
#' @param value A numeric matrix with the same dimensions as `x`.
#' @return The modified object.
#' @export
setGeneric("assay<-", function(x, i, ..., value) standardGeneric("assay<-"))

#' @rdname assay
#' @param withDimnames Should the returned matrix carry feature and sample
#'   names?
#' @export
setMethod("assay", "AssayShelf", function(x, i, withDimnames = TRUE, ...) {
  if (length(x@assays) == 0L) {
    stop("object contains no assays", call. = FALSE)
  }
  if (missing(i)) {
    i <- 1L
  }
  if (is.character(i) && !i %in% names(x@assays)) {
    stop(sprintf("no assay named '%s'", i), call. = FALSE)
  }

  out <- x@assays[[i]]
  # Dimnames live in the annotation tables, not in the stored matrices.
  if (withDimnames) {
    dimnames(out) <- list(rownames(x@rowInfo), rownames(x@colInfo))
  }
  out
})

#' @rdname assay-set
#' @export
setMethod("assay<-", "AssayShelf", function(x, i, ..., value) {
  if (missing(i)) {
    i <- if (length(x@assays) > 0L) names(x@assays)[1L] else "counts"
  }
  # A numeric index can only replace; new assays need a name.
  if (is.numeric(i) && i > length(x@assays)) {
    stop(
      "numeric index is out of range; use a name to add an assay",
      call. = FALSE
    )
  }

  value <- as.matrix(value)
  dimnames(value) <- NULL
  x@assays[[i]] <- value
  validObject(x)
  x
})

#' Get annotations
#'
#' Return the feature or sample annotation table, optionally restricted to
#' a subset of columns.
#'
#' @param x An `AssayShelf` object.
#' @param columns Optional character vector of columns to keep.
#' @return A data frame with one row per feature (`rowInfo()`) or per sample
#'   (`colInfo()`).
#' @export
rowInfo <- function(x, columns = NULL) {
  info <- x@rowInfo
  if (is.null(columns)) {
    return(info)
  }

  missing_cols <- setdiff(columns, names(info))
  if (length(missing_cols) > 0L) {
    stop(
      "unknown rowInfo columns: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  info[, columns, drop = FALSE]
}

#' @rdname rowInfo
#' @export
colInfo <- function(x, columns = NULL) {
  info <- x@colInfo
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

## 32. `00548bf23e3ff8c13a23ebf0` finish_block / sibling:medium

- package `synth:cellwright`, path `R/puzzles.R`
- generator: `{"behaviour": "checked", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b16-classic/cellwright/R/puzzles.R", "signature_region": false, "structural_support": 0.667, "target_lines": 14, "test_file": false, "unit": "spiral.order", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 90, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 451, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -41,3 +41,17 @@
     if (top <= bottom && left <= right) {
       for (j in right:left) {
         k <- k + 1L
+        out[k] <- m[bottom, j]
+      }
+      bottom <- bottom - 1L
+    }
+    if (left <= right && top <= bottom) {
+      for (i in bottom:top) {
+        k <- k + 1L
+        out[k] <- m[i, left]
+      }
+      left <- left + 1L
+    }
+  }
+  out
+}
```

**Target (gold region)**

```r
        out[k] <- m[bottom, j]
      }
      bottom <- bottom - 1L
    }
    if (left <= right && top <= bottom) {
      for (i in bottom:top) {
        k <- k + 1L
        out[k] <- m[i, left]
      }
      left <- left + 1L
    }
  }
  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/puzzles.R
# ---- diagnostics (static) --------------------
# typed statements: 6 of 8
# args not yet referenced: none
# unclosed braces in typed code: 4
# ----------------------------------------------
#' Read a matrix in spiral order
#'
#' Walks the outer ring clockwise from the top left, then the next ring in,
#' and so on.
#'
#' @param m A matrix.
#' @return A vector of the elements of `m` in spiral order.
#' @export
spiral.order <- function(m) {
  # typeof() rather than mode(): mode() reports integers as "numeric"
  out <- vector(typeof(m), length(m))
  k <- 0L
  top <- 1L
  bottom <- nrow(m)
  left <- 1L
  right <- ncol(m)

  while (top <= bottom && left <= right) {
    # top row, left to right
    for (j in left:right) {
      k <- k + 1L
      out[k] <- m[top, j]
    }
    top <- top + 1L

    # right column, downwards
    if (top <= bottom) {
      for (i in top:bottom) {
        k <- k + 1L
        out[k] <- m[i, right]
      }
    }
    right <- right - 1L

    # bottom row and left column only exist if the ring is not a single line
    if (top <= bottom && left <= right) {
      for (j in right:left) {
        k <- k + 1L
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

## 33. `3f59c8245fae1ab5150e0883` finish_block / sibling:medium

- package `synth:tillreport`, path `R/margins.R`
- generator: `{"behaviour": "no_check", "bucket": "medium", "determinacy": "sibling", "full_body": true, "novel_tokens": ["netUnits", "returnRate", "returnedUnits"], "novelty": 3, "rule": "fb_statement", "seed": "b01-tidyverse/tillreport/R/margins.R", "signature_region": true, "structural_support": 1.0, "target_lines": 10, "test_file": false, "unit": "addReturnRate", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 86, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 502, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -41,3 +41,13 @@
 #'   `returnedUnits` columns.
 #' @return `sales` with `netUnits` and `returnRate` columns.
 addReturnRate <- function(sales) {
+  sales |>
+    dplyr::mutate(
+      netUnits = .data$units - .data$returnedUnits,
+      returnRate = dplyr::if_else(
+        .data$units > 0,
+        .data$returnedUnits / .data$units,
+        NA_real_
+      )
+    )
+}
```

**Target (gold region)**

```r
addReturnRate <- function(sales) {
  sales |>
    dplyr::mutate(
      netUnits = .data$units - .data$returnedUnits,
      returnRate = dplyr::if_else(
        .data$units > 0,
        .data$returnedUnits / .data$units,
        NA_real_
      )
    )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/margins.R
#' Add gross margin columns
#'
#' @export
#' @param sales A data frame of till transactions with `revenue`,
#'   `unitCost` and `units` columns.
#' @return `sales` with `grossProfit` and `grossMargin` columns.
addGrossMargin <- function(sales) {
  sales |>
    dplyr::mutate(
      grossProfit = .data$revenue - .data$unitCost * .data$units,
      grossMargin = dplyr::if_else(
        .data$revenue > 0,
        .data$grossProfit / .data$revenue,
        NA_real_
      )
    )
}

#' Add markdown rate columns
#'
#' @export
#' @param sales A data frame of till transactions with `revenue`,
#'   `listPrice` and `units` columns.
#' @return `sales` with `markdownAmount` and `markdownRate` columns.
addMarkdownRate <- function(sales) {
  sales |>
    dplyr::mutate(
      markdownAmount = .data$listPrice * .data$units - .data$revenue,
      markdownRate = dplyr::if_else(
        .data$listPrice * .data$units > 0,
        .data$markdownAmount / (.data$listPrice * .data$units),
        NA_real_
      )
    )
}

#' Add return rate columns
#'
#' @export
#' @param sales A data frame of till transactions with `units` and
#'   `returnedUnits` columns.
#' @return `sales` with `netUnits` and `returnRate` columns.
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
addReturnRate <- function(sales) {<|user_cursor|>
=======
<[fim-middle]>
````

</details>

## 34. `13938de3caf095efba8caa84` finish_block / sibling:medium

- package `synth:driftbook`, path `R/reconcile.R`
- generator: `{"behaviour": "file_checked_unit_unexercised", "bucket": "medium", "determinacy": "sibling", "full_body": true, "novel_tokens": ["payment_id", "received", "reference"], "novelty": 3, "rule": "fb_statement", "seed": "b09-finance/driftbook/R/reconcile.R", "signature_region": true, "structural_support": 0.772, "target_lines": 9, "test_file": false, "unit": "clean_payments", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 98, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 489, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -31,3 +31,12 @@
 #' @return A tibble with the same columns, cleaned.
 #' @export
 clean_payments <- function(payments) {
+  payments %>%
+    dplyr::mutate(
+      payment_id = trimws(.data$payment_id),
+      reference = toupper(trimws(.data$reference)),
+      received = as.Date(.data$received),
+      amount = round(as.numeric(.data$amount), 2)
+    ) %>%
+    dplyr::filter(!is.na(.data$amount))
+}
```

**Target (gold region)**

```r
clean_payments <- function(payments) {
  payments %>%
    dplyr::mutate(
      payment_id = trimws(.data$payment_id),
      reference = toupper(trimws(.data$reference)),
      received = as.Date(.data$received),
      amount = round(as.numeric(.data$amount), 2)
    ) %>%
    dplyr::filter(!is.na(.data$amount))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/reconcile.R
#' @title Standardise an invoice table
#'
#' @description Trims and upper-cases references, normalises the `INV-`
#'   prefix, coerces dates and rounds amounts to cents so that matching is
#'   not defeated by formatting noise.
#'
#' @param invoices A data frame of invoices with `invoice_id`, `customer`,
#'   `issued` and `amount` columns.
#' @return A tibble with the same columns, cleaned.
#' @export
clean_invoices <- function(invoices) {
  invoices %>%
    dplyr::mutate(
      # same normalisation as extract_invoice_ref(), so ids written as
      # "INV1002" still meet payments that quote "INV-1002"
      invoice_id = sub("^INV-?", "INV-", toupper(trimws(.data$invoice_id))),
      customer = toupper(trimws(.data$customer)),
      issued = as.Date(.data$issued),
      amount = round(as.numeric(.data$amount), 2)
    ) %>%
    dplyr::filter(!is.na(.data$invoice_id), .data$invoice_id != "")
}

#' @title Standardise a payments table
#'
#' @description Trims and upper-cases references, coerces dates and rounds
#'   amounts to cents so that matching is not defeated by formatting noise.
#'
#' @param payments A data frame of bank payments with `payment_id`,
#'   `reference`, `received` and `amount` columns.
#' @return A tibble with the same columns, cleaned.
#' @export
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
clean_payments <- function(payments) {<|user_cursor|>
=======
<[fim-middle]>
````

</details>

## 35. `7ba6a3a2533ef9ef496a41bd` finish_block / sibling:medium

- package `synth:graphwalk`, path `R/toposort.R`
- generator: `{"behaviour": "checked", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["latest_finish", "latest_start", "succ"], "novelty": 3, "rule": "fb_statement", "seed": "b19-graphs-or/graphwalk/R/toposort.R", "signature_region": false, "structural_support": 0.512, "target_lines": 16, "test_file": false, "unit": "cpm_backward", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 111, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1429, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -157,3 +157,19 @@
 
   # Reverse topological order guarantees successors are already settled
   for (v in rev(order)) {
+    succ <- adj[[v]]
+    if (length(succ) > 0) {
+      lf[v] <- min(ls[succ])
+    }
+    ls[v] <- lf[v] - duration[[v]]
+  }
+
+  out <- data.frame(
+    task = order,
+    duration = unname(duration[order]),
+    latest_start = unname(ls),
+    latest_finish = unname(lf),
+    stringsAsFactors = FALSE
+  )
+  return(out)
+}
```

**Target (gold region)**

```r
    succ <- adj[[v]]
    if (length(succ) > 0) {
      lf[v] <- min(ls[succ])
    }
    ls[v] <- lf[v] - duration[[v]]
  }

  out <- data.frame(
    task = order,
    duration = unname(duration[order]),
    latest_start = unname(ls),
    latest_finish = unname(lf),
    stringsAsFactors = FALSE
  )
  return(out)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/toposort.R
#' Topological order with Kahn's algorithm
#'
#' Repeatedly removes vertices with no remaining incoming edges. Ties are
#' broken alphabetically so the result is reproducible.
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()],
#'   built with `directed = TRUE`.
#' @return A character vector of vertex names in topological order.
#' @export
topo_sort_kahn <- function(adj) {
  indeg <- in_degree(adj)
  ready <- sort(names(indeg)[indeg == 0])
  order <- character(0)

  while (length(ready) > 0) {
    v <- ready[1]
    ready <- ready[-1]
    order <- c(order, v)
    for (u in adj[[v]]) {
      indeg[u] <- indeg[u] - 1L
      if (indeg[u] == 0L) {
        # Re-sort so ties stay alphabetical regardless of discovery order
        ready <- sort(c(ready, u))
      }
    }
  }

  # Vertices never released are exactly those on or behind a cycle
  if (length(order) < length(adj)) {
    stuck <- setdiff(names(adj), order)
    stop(
      "Graph has a cycle involving: ",
      paste(stuck, collapse = ", "),
      call. = FALSE
    )
  }
  return(order)
}

#' Topological order with depth-first search
#'
#' Emits vertices in reverse post-order. Uses three-state marking to detect
#' back edges, i.e. cycles.
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()],
#'   built with `directed = TRUE`.
#' @return A character vector of vertex names in topological order.
#' @export
topo_sort_dfs <- function(adj) {
  state <- setNames(rep(0L, length(adj)), names(adj))
  order <- character(0)

  # 0 = unvisited, 1 = on the current DFS path, 2 = finished
  visit <- function(v) {
    if (state[v] == 2L) {
      return(invisible(NULL))
    }
    if (state[v] == 1L) {
      stop("Graph has a cycle through vertex ", v, call. = FALSE)
    }
    state[v] <<- 1L
    for (u in adj[[v]]) {
      visit(u)
    }
    state[v] <<- 2L
    order <<- c(v, order)
    invisible(NULL)
  }

  for (v in sort(names(adj))) {
    if (state[v] == 0L) {
      visit(v)
    }
  }
  return(order)
}

#' Check whether a directed graph is acyclic
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()],
#'   built with `directed = TRUE`.
#' @return `TRUE` if the graph has no directed cycle, otherwise `FALSE`.
#' @export
is_dag <- function(adj) {
  ok <- tryCatch(
    {
      topo_sort_kahn(adj)
      TRUE
    },
    error = function(e) FALSE
  )
  return(ok)
}

#' Earliest start times for a task network
#'
#' Forward pass of the critical path method. Each task can start once all of
#' its predecessors have finished.
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()],
#'   built with `directed = TRUE`.
#' @param duration A named numeric vector of task durations.
#' @return A data frame with columns `task`, `duration`, `earliest_start` and
#'   `earliest_finish`, in topological order.
#' @export
cpm_forward <- function(adj, duration) {
  missing_tasks <- setdiff(names(adj), names(duration))
  if (length(missing_tasks) > 0) {
    stop(
      "No duration for: ",
      paste(missing_tasks, collapse = ", "),
      call. = FALSE
    )
  }

  order <- topo_sort_kahn(adj)
  es <- setNames(numeric(length(order)), order)
  ef <- setNames(numeric(length(order)), order)

  # Push each finish time forward to the successors' start times
  for (v in order) {
    ef[v] <- es[v] + duration[[v]]
    for (u in adj[[v]]) {
      if (ef[v] > es[u]) {
        es[u] <- ef[v]
      }
    }
  }

  out <- data.frame(
    task = order,
    duration = unname(duration[order]),
    earliest_start = unname(es),
    earliest_finish = unname(ef),
    stringsAsFactors = FALSE
  )
  return(out)
}

#' Latest start times for a task network
#'
#' Backward pass of the critical path method, anchored at the project
#' finish time from the forward pass.
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()],
#'   built with `directed = TRUE`.
#' @param duration A named numeric vector of task durations.
#' @return A data frame with columns `task`, `duration`, `latest_start` and
#'   `latest_finish`, in topological order.
#' @export
cpm_backward <- function(adj, duration) {
  fwd <- cpm_forward(adj, duration)
  order <- fwd$task
  horizon <- max(fwd$earliest_finish)
  lf <- setNames(rep(horizon, length(order)), order)
  ls <- setNames(numeric(length(order)), order)

  # Reverse topological order guarantees successors are already settled
  for (v in rev(order)) {
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

## 36. `37cb80bc073fbc50f1db50ab` finish_block / sibling:medium

- package `synth:ledgerkeel`, path `R/quality_checks.R`
- generator: `{"behaviour": "file_checked_unit_unexercised", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["AND", "IN"], "novelty": 2, "rule": "fb_nested_close", "seed": "b12-db/ledgerkeel/R/quality_checks.R", "signature_region": false, "structural_support": 0.576, "target_lines": 10, "test_file": false, "unit": "check.accepted.values", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 95, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 757, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -65,3 +65,13 @@
 check.accepted.values <- function(con, table, column, allowed) {
   if (length(allowed) == 0) {
     stop("`allowed` must contain at least one value.", call. = FALSE)
+  }
+
+  sql <- glue::glue_sql(
+    "SELECT COUNT(*) AS n FROM {`table`}
+     WHERE {`column`} IS NOT NULL AND {`column`} NOT IN ({allowed*})",
+    .con = con
+  )
+  n <- DBI::dbGetQuery(con, sql)$n
+  check.result("accepted_values", table, column, n)
+}
```

**Target (gold region)**

```r
  }

  sql <- glue::glue_sql(
    "SELECT COUNT(*) AS n FROM {`table`}
     WHERE {`column`} IS NOT NULL AND {`column`} NOT IN ({allowed*})",
    .con = con
  )
  n <- DBI::dbGetQuery(con, sql)$n
  check.result("accepted_values", table, column, n)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/quality_checks.R
#' Build a check result
#'
#' Wraps the outcome of one SQL data quality check in a one-row data frame
#' so that results from different checks can be bound together.
#'
#' @param check Short name of the check.
#' @param table Name of the table checked.
#' @param column Name of the column checked, or `NA`.
#' @param failures Number of offending rows.
#' @keywords internal
check.result <- function(check, table, column, failures) {
  failures <- as.numeric(failures)
  data.frame(
    check = check,
    table = table,
    column = column,
    failures = failures,
    passed = failures == 0,
    stringsAsFactors = FALSE
  )
}

#' SQL data quality checks
#'
#' Each check counts offending rows entirely on the server and returns a
#' one-row check result. `check.not.null()` counts rows where `column` is
#' `NULL`; `check.unique()` counts values of `column` that occur more than
#' once; `check.accepted.values()` counts rows whose `column` is not one of
#' `allowed`; `check.in.range()` counts rows whose `column` lies outside
#' `[lower, upper]`. Missing values are only counted by `check.not.null()`.
#'
#' @param con A `DBIConnection` object.
#' @param table Name of the table.
#' @param column Name of the column.
#' @export
#' @return A one-row check result.
check.not.null <- function(con, table, column) {
  sql <- glue::glue_sql(
    "SELECT COUNT(*) AS n FROM {`table`} WHERE {`column`} IS NULL",
    .con = con
  )
  n <- DBI::dbGetQuery(con, sql)$n
  check.result("not_null", table, column, n)
}

#' @rdname check.not.null
#' @export
check.unique <- function(con, table, column) {
  # count duplicated values, not duplicated rows
  sql <- glue::glue_sql(
    "SELECT COUNT(*) AS n FROM (
       SELECT {`column`} FROM {`table`}
       WHERE {`column`} IS NOT NULL
       GROUP BY {`column`} HAVING COUNT(*) > 1
     ) AS d",
    .con = con
  )
  n <- DBI::dbGetQuery(con, sql)$n
  check.result("unique", table, column, n)
}

#' @rdname check.not.null
#' @param allowed A vector of allowed values.
#' @export
check.accepted.values <- function(con, table, column, allowed) {
  if (length(allowed) == 0) {
    stop("`allowed` must contain at least one value.", call. = FALSE)
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

## 37. `5213b709a755f22381de40cc` finish_block / sibling:medium

- package `synth:herdwalk`, path `R/movement.R`
- generator: `{"behaviour": "checked", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["dx", "dy"], "novelty": 2, "rule": "fb_statement", "seed": "b23-simulation/herdwalk/R/movement.R", "signature_region": false, "structural_support": 0.519, "target_lines": 15, "test_file": false, "unit": "move_toward", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 136, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1312, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -121,3 +121,18 @@
 ) {
   stopifnot(length(target) == 2, pull >= 0, pull <= 1)
   mobile <- pop$state != "R"
+  n_mobile <- sum(mobile)
+  if (n_mobile == 0) {
+    return(pop)
+  }
+
+  dx <- target[[1]] - pop$x[mobile]
+  dy <- target[[2]] - pop$y[mobile]
+  pop$x[mobile] <- pop$x[mobile] +
+    pull * dx +
+    stats::rnorm(n_mobile, sd = step_sd)
+  pop$y[mobile] <- pop$y[mobile] +
+    pull * dy +
+    stats::rnorm(n_mobile, sd = step_sd)
+  apply_boundary(pop, boundary)
+}
```

**Target (gold region)**

```r
  n_mobile <- sum(mobile)
  if (n_mobile == 0) {
    return(pop)
  }

  dx <- target[[1]] - pop$x[mobile]
  dy <- target[[2]] - pop$y[mobile]
  pop$x[mobile] <- pop$x[mobile] +
    pull * dx +
    stats::rnorm(n_mobile, sd = step_sd)
  pop$y[mobile] <- pop$y[mobile] +
    pull * dy +
    stats::rnorm(n_mobile, sd = step_sd)
  apply_boundary(pop, boundary)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/movement.R
#' Boundary helpers
#'
#' Put coordinates that left the arena back inside. `wrap_coord()` treats
#' the arena as a torus, `reflect_coord()` mirrors at the walls as if the
#' agent bounced off them, and `clamp_coord()` stops agents at the wall they
#' tried to cross.
#'
#' @param coord Numeric vector of coordinates along one axis.
#' @param limit Size of the arena along that axis.
#' @return A numeric vector of coordinates inside `[0, limit]`.
#' @keywords internal
wrap_coord <- function(coord, limit) {
  coord %% limit
}

#' @rdname wrap_coord
reflect_coord <- function(coord, limit) {
  # Fold onto one period of the mirrored arena, then flip the far half.
  period <- 2 * limit
  folded <- coord %% period
  ifelse(folded > limit, period - folded, folded)
}

#' @rdname wrap_coord
clamp_coord <- function(coord, limit) {
  pmin(pmax(coord, 0), limit)
}

#' Apply a boundary rule to a population
#'
#' Puts every agent back inside the arena using one of the supported
#' boundary conditions.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param boundary Boundary condition: `"wrap"`, `"reflect"` or `"clamp"`.
#' @return The population with corrected `x` and `y` columns.
#' @keywords internal
apply_boundary <- function(pop, boundary = c("wrap", "reflect", "clamp")) {
  boundary <- match.arg(boundary)
  arena <- attr(pop, "arena")
  fix <- switch(
    boundary,
    wrap = wrap_coord,
    reflect = reflect_coord,
    clamp = clamp_coord
  )

  pop$x <- fix(pop$x, arena[["width"]])
  pop$y <- fix(pop$y, arena[["height"]])
  pop
}

#' Move agents by a Gaussian random walk
#'
#' Each agent takes an independent step with normally distributed x and y
#' displacements. Removed agents do not move.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param step_sd Standard deviation of the step along each axis.
#' @param boundary Boundary condition: `"wrap"`, `"reflect"` or `"clamp"`.
#' @return The population after one movement step.
#' @export
move_random <- function(pop, step_sd = 1, boundary = "wrap") {
  mobile <- pop$state != "R"
  n_mobile <- sum(mobile)
  if (n_mobile == 0) {
    return(pop)
  }

  pop$x[mobile] <- pop$x[mobile] + stats::rnorm(n_mobile, sd = step_sd)
  pop$y[mobile] <- pop$y[mobile] + stats::rnorm(n_mobile, sd = step_sd)
  apply_boundary(pop, boundary)
}

#' Move agents by a Levy flight
#'
#' Step lengths follow a Pareto distribution and headings are uniform, so
#' most moves are short with occasional long jumps. Removed agents do not
#' move.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param alpha Tail exponent of the step-length distribution.
#' @param min_step Smallest possible step length.
#' @param boundary Boundary condition: `"wrap"`, `"reflect"` or `"clamp"`.
#' @return The population after one movement step.
#' @export
move_levy <- function(pop, alpha = 1.5, min_step = 0.5, boundary = "wrap") {
  mobile <- pop$state != "R"
  n_mobile <- sum(mobile)
  if (n_mobile == 0) {
    return(pop)
  }

  # Inverse-CDF draw from a Pareto(min_step, alpha) distribution.
  lengths <- min_step * stats::runif(n_mobile)^(-1 / alpha)
  heading <- stats::runif(n_mobile, 0, 2 * pi)

  pop$x[mobile] <- pop$x[mobile] + lengths * cos(heading)
  pop$y[mobile] <- pop$y[mobile] + lengths * sin(heading)
  apply_boundary(pop, boundary)
}

#' Move agents toward an attractor
#'
#' Agents drift a fixed fraction of the way toward a point of interest,
#' such as a market or a water hole, plus a small amount of noise.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param target Numeric vector of length two giving the attractor position.
#' @param pull Fraction of the remaining distance covered per step.
#' @param step_sd Standard deviation of the step along each axis.
#' @param boundary Boundary condition: `"wrap"`, `"reflect"` or `"clamp"`.
#' @return The population after one movement step.
#' @export
move_toward <- function(
  pop,
  target,
  pull = 0.1,
  step_sd = 0.5,
  boundary = "clamp"
) {
  stopifnot(length(target) == 2, pull >= 0, pull <= 1)
  mobile <- pop$state != "R"
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

## 38. `0513f41134419e3489c3211a` finish_block / sibling:medium

- package `synth:querywright`, path `R/xml.R`
- generator: `{"behaviour": "file_checked_unit_unexercised", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b18-web/querywright/R/xml.R", "signature_region": false, "structural_support": 0.918, "target_lines": 16, "test_file": false, "unit": "read_sitemap_index", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 118, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1172, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -104,3 +104,19 @@
   doc <- xml2::read_xml(x) %>% xml2::xml_ns_strip()
   nodes <- xml2::xml_find_all(doc, "/sitemapindex/sitemap")
 
+  out <- data.frame(
+    loc = character(length(nodes)),
+    lastmod = as.Date(rep(NA_character_, length(nodes))),
+    stringsAsFactors = FALSE
+  )
+  for (i in seq_along(nodes)) {
+    out$loc[i] <- child_text(nodes[[i]], "./loc")
+    out$lastmod[i] <- as.Date(substr(
+      child_text(nodes[[i]], "./lastmod"),
+      1,
+      10
+    ))
+  }
+
+  out
+}
```

**Target (gold region)**

```r
  out <- data.frame(
    loc = character(length(nodes)),
    lastmod = as.Date(rep(NA_character_, length(nodes))),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(nodes)) {
    out$loc[i] <- child_text(nodes[[i]], "./loc")
    out$lastmod[i] <- as.Date(substr(
      child_text(nodes[[i]], "./lastmod"),
      1,
      10
    ))
  }

  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/xml.R
#' Read the text of a child node
#'
#' Finds the first node matching `xpath` below `node` and returns its
#' trimmed text, or `NA` when there is no such node.
#'
#' @param node An `xml_node` to search from.
#' @param xpath An XPath expression relative to `node`.
#' @param ns A namespace map as returned by [xml2::xml_ns()].
#' @return A single string.
#' @keywords internal
child_text <- function(node, xpath, ns = character()) {
  child <- xml2::xml_find_first(node, xpath, ns)
  if (inherits(child, "xml_missing")) {
    return(NA_character_)
  }
  child %>% xml2::xml_text() %>% trimws()
}

#' Read an attribute of a child node
#'
#' Finds the first node matching `xpath` below `node` and returns the
#' value of one attribute, or `NA` when there is no such node.
#'
#' @inheritParams child_text
#' @param attr Name of the attribute to read.
#' @return A single string.
#' @keywords internal
child_attr <- function(node, xpath, attr, ns = character()) {
  child <- xml2::xml_find_first(node, xpath, ns)
  if (inherits(child, "xml_missing")) {
    return(NA_character_)
  }
  child %>% xml2::xml_attr(attr) %>% trimws()
}

#' Convert a trailing UTC offset to seconds
#'
#' Reads offsets such as `+0200`, `-05:00` or `+00:00` at the end of a
#' timestamp. `Z`, `GMT` and missing offsets count as zero.
#'
#' @param x Character vector of timestamps.
#' @return A numeric vector of offsets in seconds east of UTC.
#' @noRd
tz_offset_seconds <- function(x) {
  m <- regmatches(x, regexec("([+-])([0-9]{2}):?([0-9]{2})$", x))
  vapply(
    m,
    function(p) {
      if (length(p) == 0) {
        return(0)
      }
      sign <- if (p[2] == "-") -1 else 1
      sign * (as.numeric(p[3]) * 3600 + as.numeric(p[4]) * 60)
    },
    numeric(1)
  )
}

#' Read a sitemap
#'
#' Parses a `urlset` sitemap into one row per URL. Namespaces are stripped
#' so that sitemaps with and without the standard namespace both work.
#'
#' @param x A path, URL or literal XML string.
#' @return A data frame with columns `loc`, `lastmod`, `changefreq` and
#'   `priority`.
#' @export
read_sitemap <- function(x) {
  doc <- xml2::read_xml(x) %>% xml2::xml_ns_strip()
  nodes <- xml2::xml_find_all(doc, "/urlset/url")

  out <- data.frame(
    loc = character(length(nodes)),
    lastmod = as.Date(rep(NA_character_, length(nodes))),
    changefreq = character(length(nodes)),
    priority = numeric(length(nodes)),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(nodes)) {
    out$loc[i] <- child_text(nodes[[i]], "./loc")
    # lastmod may be a full W3C datetime; the date part is all we keep
    out$lastmod[i] <- as.Date(substr(
      child_text(nodes[[i]], "./lastmod"),
      1,
      10
    ))
    out$changefreq[i] <- child_text(nodes[[i]], "./changefreq")
    out$priority[i] <- as.numeric(child_text(nodes[[i]], "./priority"))
  }

  # 0.5 is the default priority defined by the sitemap protocol
  out$priority[is.na(out$priority)] <- 0.5
  out
}

#' Read a sitemap index
#'
#' Parses a `sitemapindex` document into one row per child sitemap.
#'
#' @param x A path, URL or literal XML string.
#' @return A data frame with columns `loc` and `lastmod`.
#' @export
read_sitemap_index <- function(x) {
  doc <- xml2::read_xml(x) %>% xml2::xml_ns_strip()
  nodes <- xml2::xml_find_all(doc, "/sitemapindex/sitemap")

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

## 39. `4dc16add5549947c1cc16e63` finish_block / sibling:medium

- package `synth:inkplate`, path `R/template.R`
- generator: `{"behaviour": "file_checked_unit_unexercised", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["escape.for"], "novelty": 1, "rule": "fb_statement", "seed": "b10-text/inkplate/R/template.R", "signature_region": false, "structural_support": 0.718, "target_lines": 12, "test_file": false, "unit": "render.braces", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 105, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 447, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -27,3 +27,15 @@
   })
 
   esc.pat <- "\\{\\{\\s*([A-Za-z_][A-Za-z0-9_.]*)\\s*\\}\\}"
+  m <- gregexpr(esc.pat, template, perl = TRUE)
+  regmatches(template, m) <- lapply(regmatches(template, m), function(tags) {
+    keys <- sub(esc.pat, "\\1", tags, perl = TRUE)
+    vapply(
+      keys,
+      function(k) escape.for(values[[k]], format = format),
+      character(1)
+    )
+  })
+
+  template
+}
```

**Target (gold region)**

```r
  m <- gregexpr(esc.pat, template, perl = TRUE)
  regmatches(template, m) <- lapply(regmatches(template, m), function(tags) {
    keys <- sub(esc.pat, "\\1", tags, perl = TRUE)
    vapply(
      keys,
      function(k) escape.for(values[[k]], format = format),
      character(1)
    )
  })

  template
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/template.R
#' Render a double-brace template
#'
#' Substitutes each `{{ name }}` placeholder with the matching entry of
#' `values`. Values are escaped for the chosen format unless the
#' placeholder is written with triple braces.
#'
#' @param template A single template string.
#' @param values A named list or character vector of substitution values.
#' @param format Output format passed to [escape.for()].
#' @return A single rendered string.
#' @family renderers
#' @export
render.braces <- function(
  template,
  values,
  format = c("html", "markdown", "latex")
) {
  format <- match.arg(format)
  check.template.vars(template, values)

  # raw placeholders first so the double-brace pass does not see them
  raw.pat <- "\\{\\{\\{\\s*([A-Za-z_][A-Za-z0-9_.]*)\\s*\\}\\}\\}"
  m <- gregexpr(raw.pat, template, perl = TRUE)
  regmatches(template, m) <- lapply(regmatches(template, m), function(tags) {
    keys <- sub(raw.pat, "\\1", tags, perl = TRUE)
    vapply(keys, function(k) as.character(values[[k]]), character(1))
  })

  esc.pat <- "\\{\\{\\s*([A-Za-z_][A-Za-z0-9_.]*)\\s*\\}\\}"
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

## 40. `0b39e5a26402c26353689926` finish_block / sibling:medium

- package `synth:assayshelf`, path `R/show-methods.R`
- generator: `{"behaviour": "file_checked_unit_unexercised", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["meta", "str:meta"], "novelty": 2, "rule": "fb_statement", "seed": "b17-bioc/assayshelf/R/show-methods.R", "signature_region": false, "structural_support": 0.509, "target_lines": 11, "test_file": false, "unit": "setMethod", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 86, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 341, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -21,3 +21,14 @@
   cat(.format_row_line(object), "\n", sep = "")
   cat(.format_info_line(object@rowInfo, "rowInfo"), "\n", sep = "")
   cat(.format_col_line(object), "\n", sep = "")
+  cat(.format_info_line(object@colInfo, "colInfo"), "\n", sep = "")
+
+  if (length(object@meta) > 0L) {
+    cat(sprintf(
+      "meta(%d): %s\n",
+      length(object@meta),
+      .abbrev_names(names(object@meta))
+    ))
+  }
+  invisible(object)
+})
```

**Target (gold region)**

```r
  cat(.format_info_line(object@colInfo, "colInfo"), "\n", sep = "")

  if (length(object@meta) > 0L) {
    cat(sprintf(
      "meta(%d): %s\n",
      length(object@meta),
      .abbrev_names(names(object@meta))
    ))
  }
  invisible(object)
})
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/show-methods.R
#' Display an AssayShelf
#'
#' Prints dimensions, assay names, dimnames and annotation columns.
#'
#' @param object An `AssayShelf` object.
#' @return `object`, invisibly.
#' @export
setMethod("show", "AssayShelf", function(object) {
  n_row <- nrow(object@rowInfo)
  n_col <- nrow(object@colInfo)

  cat("class: AssayShelf\n")
  cat(sprintf("dim: %d %d\n", n_row, n_col))
  cat(sprintf(
    "assays(%d): %s\n",
    length(object@assays),
    .abbrev_names(names(object@assays))
  ))

  # Feature lines first, then sample lines, mirroring the matrix layout.
  cat(.format_row_line(object), "\n", sep = "")
  cat(.format_info_line(object@rowInfo, "rowInfo"), "\n", sep = "")
  cat(.format_col_line(object), "\n", sep = "")
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

## 41. `0ed4ddc29d673588da5e5c57` finish_block / sibling:medium

- package `synth:querywright`, path `R/xml.R`
- generator: `{"behaviour": "file_checked_unit_unexercised", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["num:19", "str:dT", "str:updated"], "novelty": 3, "rule": "fb_statement", "seed": "b18-web/querywright/R/xml.R", "signature_region": false, "structural_support": 0.821, "target_lines": 13, "test_file": false, "unit": "parse_atom_entries", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 127, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1543, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -141,3 +141,16 @@
       "./link[not(@rel) or @rel='alternate']",
       "href"
     )
+    stamp <- child_text(entries[[i]], "./updated")
+    local <- as.POSIXct(
+      substr(stamp, 1, 19),
+      format = "%Y-%m-%dT%H:%M:%S",
+      tz = "UTC"
+    )
+    out$published[i] <- local - tz_offset_seconds(stamp)
+    out$id[i] <- child_text(entries[[i]], "./id")
+  }
+
+  out$id[is.na(out$id)] <- out$link[is.na(out$id)]
+  out
+}
```

**Target (gold region)**

```r
    stamp <- child_text(entries[[i]], "./updated")
    local <- as.POSIXct(
      substr(stamp, 1, 19),
      format = "%Y-%m-%dT%H:%M:%S",
      tz = "UTC"
    )
    out$published[i] <- local - tz_offset_seconds(stamp)
    out$id[i] <- child_text(entries[[i]], "./id")
  }

  out$id[is.na(out$id)] <- out$link[is.na(out$id)]
  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/xml.R
#' @return A numeric vector of offsets in seconds east of UTC.
#' @noRd
tz_offset_seconds <- function(x) {
  m <- regmatches(x, regexec("([+-])([0-9]{2}):?([0-9]{2})$", x))
  vapply(
    m,
    function(p) {
      if (length(p) == 0) {
        return(0)
      }
      sign <- if (p[2] == "-") -1 else 1
      sign * (as.numeric(p[3]) * 3600 + as.numeric(p[4]) * 60)
    },
    numeric(1)
  )
}

#' Read a sitemap
#'
#' Parses a `urlset` sitemap into one row per URL. Namespaces are stripped
#' so that sitemaps with and without the standard namespace both work.
#'
#' @param x A path, URL or literal XML string.
#' @return A data frame with columns `loc`, `lastmod`, `changefreq` and
#'   `priority`.
#' @export
read_sitemap <- function(x) {
  doc <- xml2::read_xml(x) %>% xml2::xml_ns_strip()
  nodes <- xml2::xml_find_all(doc, "/urlset/url")

  out <- data.frame(
    loc = character(length(nodes)),
    lastmod = as.Date(rep(NA_character_, length(nodes))),
    changefreq = character(length(nodes)),
    priority = numeric(length(nodes)),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(nodes)) {
    out$loc[i] <- child_text(nodes[[i]], "./loc")
    # lastmod may be a full W3C datetime; the date part is all we keep
    out$lastmod[i] <- as.Date(substr(
      child_text(nodes[[i]], "./lastmod"),
      1,
      10
    ))
    out$changefreq[i] <- child_text(nodes[[i]], "./changefreq")
    out$priority[i] <- as.numeric(child_text(nodes[[i]], "./priority"))
  }

  # 0.5 is the default priority defined by the sitemap protocol
  out$priority[is.na(out$priority)] <- 0.5
  out
}

#' Read a sitemap index
#'
#' Parses a `sitemapindex` document into one row per child sitemap.
#'
#' @param x A path, URL or literal XML string.
#' @return A data frame with columns `loc` and `lastmod`.
#' @export
read_sitemap_index <- function(x) {
  doc <- xml2::read_xml(x) %>% xml2::xml_ns_strip()
  nodes <- xml2::xml_find_all(doc, "/sitemapindex/sitemap")

  out <- data.frame(
    loc = character(length(nodes)),
    lastmod = as.Date(rep(NA_character_, length(nodes))),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(nodes)) {
    out$loc[i] <- child_text(nodes[[i]], "./loc")
    out$lastmod[i] <- as.Date(substr(
      child_text(nodes[[i]], "./lastmod"),
      1,
      10
    ))
  }

  out
}

#' Parse the items of an RSS feed
#'
#' Reads one row per `<item>`, falling back to the link when an item has
#' no `<guid>`. Publication times are converted to UTC using the offset at
#' the end of `<pubDate>`.
#'
#' @param doc An `xml_document` holding an RSS 2.0 feed with namespaces
#'   stripped.
#' @return A data frame with columns `title`, `link`, `published` and `id`.
#' @keywords internal
parse_rss_items <- function(doc) {
  items <- xml2::xml_find_all(doc, "/rss/channel/item")

  out <- data.frame(
    title = character(length(items)),
    link = character(length(items)),
    published = as.POSIXct(rep(NA_real_, length(items)), tz = "UTC"),
    id = character(length(items)),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(items)) {
    out$title[i] <- child_text(items[[i]], "./title")
    out$link[i] <- child_text(items[[i]], "./link")
    # RFC 822 dates carry a zone offset after the time; parse the local
    # time first, then shift it to UTC.
    stamp <- child_text(items[[i]], "./pubDate")
    local <- as.POSIXct(stamp, format = "%a, %d %b %Y %H:%M:%S", tz = "UTC")
    out$published[i] <- local - tz_offset_seconds(stamp)
    out$id[i] <- child_text(items[[i]], "./guid")
  }

  out$id[is.na(out$id)] <- out$link[is.na(out$id)]
  out
}

#' Parse the entries of an Atom feed
#'
#' Reads one row per `<entry>`, falling back to the link when an entry has
#' no `<id>`. Times are converted to UTC like in [parse_rss_items()].
#'
#' @param doc An `xml_document` holding an Atom feed with namespaces
#'   stripped.
#' @return A data frame with columns `title`, `link`, `published` and `id`.
#' @keywords internal
parse_atom_entries <- function(doc) {
  entries <- xml2::xml_find_all(doc, "/feed/entry")

  out <- data.frame(
    title = character(length(entries)),
    link = character(length(entries)),
    published = as.POSIXct(rep(NA_real_, length(entries)), tz = "UTC"),
    id = character(length(entries)),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(entries)) {
    out$title[i] <- child_text(entries[[i]], "./title")
    out$link[i] <- child_attr(
      entries[[i]],
      "./link[not(@rel) or @rel='alternate']",
      "href"
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

## 42. `75dd76c16354746f228a1645` finish_block / sibling:medium

- package `synth:cellarlink`, path `R/sql_build.R`
- generator: `{"behaviour": "no_check", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["str:Refusing", "str:to", "str:without"], "novelty": 3, "rule": "fb_statement", "seed": "b12-db/cellarlink/R/sql_build.R", "signature_region": false, "structural_support": 0.63, "target_lines": 10, "test_file": false, "unit": "build_delete", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 84, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 869, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -60,3 +60,13 @@
 #' @export
 build_delete <- function(con, table, where) {
   if (length(where) == 0) {
+    stop("Refusing to build a DELETE without a WHERE clause.", call. = FALSE)
+  }
+
+  tbl <- quote_table(con, table)
+  sql <- glue::glue_sql(
+    "DELETE FROM {tbl} WHERE {where_clause(con, where)}",
+    .con = con
+  )
+  return(sql)
+}
```

**Target (gold region)**

```r
    stop("Refusing to build a DELETE without a WHERE clause.", call. = FALSE)
  }

  tbl <- quote_table(con, table)
  sql <- glue::glue_sql(
    "DELETE FROM {tbl} WHERE {where_clause(con, where)}",
    .con = con
  )
  return(sql)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/sql_build.R
#' @title Quote a vector of column names
#' @description Quotes each name with [DBI::dbQuoteIdentifier()] and joins
#'   them into a comma-separated list for use in a `SELECT` or `INSERT`
#'   clause.
#' @param con A `DBIConnection` object.
#' @param cols A character vector of column names.
#' @keywords internal
quote_cols <- function(con, cols) {
  if (length(cols) == 0) {
    stop("`cols` must contain at least one column name.", call. = FALSE)
  }
  quoted <- DBI::dbQuoteIdentifier(con, cols)
  return(DBI::SQL(paste(quoted, collapse = ", ")))
}

#' @title Quote a possibly schema-qualified table name
#' @description Splits `"schema.table"` on the first dot and builds a
#'   [DBI::Id()] so both parts are quoted separately.
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @keywords internal
quote_table <- function(con, table) {
  # anything with more than one dot is passed through as a plain name
  parts <- strsplit(table, ".", fixed = TRUE)[[1]]
  id <- if (length(parts) == 2) {
    DBI::Id(schema = parts[1], table = parts[2])
  } else {
    DBI::Id(table = table)
  }
  return(DBI::dbQuoteIdentifier(con, id))
}

#' @title Build a SELECT statement
#' @description Builds a `SELECT` with optional equality filters. Filter
#'   values are interpolated safely with [glue::glue_sql()].
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @param cols A character vector of column names, or `NULL` for all columns.
#' @param where A named list of equality filters; names are columns.
#' @return An `SQL` string.
#' @export
build_select <- function(con, table, cols = NULL, where = list()) {
  tbl <- quote_table(con, table)
  col_sql <- if (is.null(cols)) DBI::SQL("*") else quote_cols(con, cols)

  sql <- glue::glue_sql("SELECT {col_sql} FROM {tbl}", .con = con)
  if (length(where) > 0) {
    sql <- glue::glue_sql("{sql} WHERE {where_clause(con, where)}", .con = con)
  }
  return(sql)
}

#' @title Build a DELETE statement
#' @description Builds a `DELETE` restricted by equality filters. An empty
#'   `where` is refused so that a typo never wipes a whole table.
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @param where A named list of equality filters; names are columns.
#' @return An `SQL` string.
#' @export
build_delete <- function(con, table, where) {
  if (length(where) == 0) {
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

## 43. `3538a03c045f70e20110ca89` finish_block / sibling:long

- package `synth:driftbook`, path `R/smoothing.R`
- generator: `{"behaviour": "checked", "bucket": "long", "determinacy": "sibling", "full_body": false, "novel_tokens": ["prev_level", "prev_trend"], "novelty": 2, "rule": "fb_nested_close", "seed": "b09-finance/driftbook/R/smoothing.R", "signature_region": false, "structural_support": 0.519, "target_lines": 28, "test_file": false, "unit": "holt_filter", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 206, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 651, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -58,3 +58,31 @@
   n <- length(y)
   if (n < 3) {
     stop("Holt smoothing needs at least three observations.", call. = FALSE)
+  }
+
+  level <- numeric(n)
+  trend <- numeric(n)
+  fitted <- numeric(n)
+  prev_level <- y[1]
+  prev_trend <- y[2] - y[1]
+
+  for (t in seq_len(n)) {
+    fitted[t] <- prev_level + prev_trend
+    level[t] <- alpha * y[t] + (1 - alpha) * (prev_level + prev_trend)
+    trend[t] <- beta * (level[t] - prev_level) + (1 - beta) * prev_trend
+    prev_level <- level[t]
+    prev_trend <- trend[t]
+  }
+
+  structure(
+    list(
+      fitted = fitted,
+      level = level,
+      trend = trend,
+      sse = sum((y - fitted)^2),
+      params = c(alpha = alpha, beta = beta),
+      method = "holt"
+    ),
+    class = "drift_smooth"
+  )
+}
```

**Target (gold region)**

```r
  }

  level <- numeric(n)
  trend <- numeric(n)
  fitted <- numeric(n)
  prev_level <- y[1]
  prev_trend <- y[2] - y[1]

  for (t in seq_len(n)) {
    fitted[t] <- prev_level + prev_trend
    level[t] <- alpha * y[t] + (1 - alpha) * (prev_level + prev_trend)
    trend[t] <- beta * (level[t] - prev_level) + (1 - beta) * prev_trend
    prev_level <- level[t]
    prev_trend <- trend[t]
  }

  structure(
    list(
      fitted = fitted,
      level = level,
      trend = trend,
      sse = sum((y - fitted)^2),
      params = c(alpha = alpha, beta = beta),
      method = "holt"
    ),
    class = "drift_smooth"
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/smoothing.R
#' Simple exponential smoothing
#'
#' @description Filters `y` with a single smoothing weight. The level is
#'   initialised at the first observation unless `level0` is given.
#'
#' @param y Numeric vector of observations in time order.
#' @param alpha Smoothing weight for the level, between 0 and 1.
#' @param level0 Optional starting level.
#' @return A list of class `drift_smooth` with `fitted`, `level`, `sse` and
#'   the smoothing parameters.
#' @family smoothing filters
#' @export
ses_filter <- function(y, alpha, level0 = NULL) {
  check_weight(alpha, "alpha")
  n <- length(y)
  if (n == 0) {
    stop("`y` must contain at least one observation.", call. = FALSE)
  }

  level <- numeric(n)
  fitted <- numeric(n)
  prev <- if (is.null(level0)) y[1] else level0
  for (t in seq_len(n)) {
    fitted[t] <- prev
    level[t] <- alpha * y[t] + (1 - alpha) * prev
    prev <- level[t]
  }

  structure(
    list(
      fitted = fitted,
      level = level,
      trend = NULL,
      sse = sum((y - fitted)^2),
      params = c(alpha = alpha),
      method = "ses"
    ),
    class = "drift_smooth"
  )
}

#' Holt linear trend smoothing
#'
#' @description Filters `y` with separate smoothing weights for the level
#'   and the trend. The trend is initialised from the first two
#'   observations.
#'
#' @param y Numeric vector of observations in time order.
#' @param alpha Smoothing weight for the level, between 0 and 1.
#' @param beta Smoothing weight for the trend, between 0 and 1.
#' @return A list of class `drift_smooth` with `fitted`, `level`, `trend`,
#'   `sse` and the smoothing parameters.
#' @family smoothing filters
#' @export
holt_filter <- function(y, alpha, beta) {
  check_weight(alpha, "alpha")
  check_weight(beta, "beta")
  n <- length(y)
  if (n < 3) {
    stop("Holt smoothing needs at least three observations.", call. = FALSE)
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

## 44. `6537fd3d6b25da5f270a7ca3` finish_block / sibling:long

- package `synth:priorcraft`, path `R/gamSpecs.R`
- generator: `{"behaviour": "checked", "bucket": "long", "determinacy": "sibling", "full_body": false, "novel_tokens": ["str:warned", "suppressWarnings"], "novelty": 2, "rule": "fb_nested_close", "seed": "b24-modelling/priorcraft/R/gamSpecs.R", "signature_region": false, "structural_support": 0.509, "target_lines": 21, "test_file": false, "unit": "fitGamSpec", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 102, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1329, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -137,3 +137,24 @@
         ),
         call. = FALSE
       )
+    },
+    warning = function(w) {
+      warning(
+        sprintf(
+          "GAM fit for `%s` warned: %s",
+          deparse1(f),
+          conditionMessage(w)
+        ),
+        call. = FALSE
+      )
+      suppressWarnings(mgcv::gam(
+        f,
+        data = data,
+        family = family,
+        method = method,
+        ...
+      ))
+    }
+  )
+  return(fit)
+}
```

**Target (gold region)**

```r
    },
    warning = function(w) {
      warning(
        sprintf(
          "GAM fit for `%s` warned: %s",
          deparse1(f),
          conditionMessage(w)
        ),
        call. = FALSE
      )
      suppressWarnings(mgcv::gam(
        f,
        data = data,
        family = family,
        method = method,
        ...
      ))
    }
  )
  return(fit)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/gamSpecs.R
#' Build a GAM formula
#'
#' Combines a response, smooth specifications and parametric terms into a
#' formula for [mgcv::gam()].
#'
#' @export
#' @param response Name of the response variable.
#' @param smooths A list of `pcSmooth` objects.
#' @param parametric Character vector of parametric terms.
#' @param env Environment for the created formula.
#' @return A formula.
gamFormula <- function(
  response,
  smooths,
  parametric = character(),
  env = parent.frame()
) {
  if (inherits(smooths, "pcSmooth")) {
    smooths <- list(smooths)
  }

  smoothTerms <- vapply(smooths, format, character(1))
  byVars <- unlist(lapply(smooths, `[[`, "by"))
  # factor-by smooths need the factor's main effect to stay identifiable
  parametric <- union(parametric, byVars)

  rhs <- paste(c(parametric, smoothTerms), collapse = " + ")
  if (!nzchar(rhs)) {
    rhs <- "1"
  }
  return(stats::as.formula(paste(response, "~", rhs), env = env))
}

#' Check basis sizes against the data
#'
#' Compares each smooth's basis dimension with the number of distinct values
#' of its covariate, since mgcv refuses a basis larger than that. Random
#' effect smooths (`bs = "re"`) have no basis dimension and are skipped.
#'
#' @export
#' @param smooths A list of `pcSmooth` objects.
#' @param data A data frame containing the covariates.
#' @return A data frame with one row per covariate and a logical `ok` column.
checkBasisSize <- function(smooths, data) {
  if (inherits(smooths, "pcSmooth")) {
    smooths <- list(smooths)
  }

  rows <- list()
  for (sm in smooths) {
    if (sm$bs == "re") {
      next
    }
    for (j in seq_along(sm$vars)) {
      v <- sm$vars[j]
      if (!v %in% names(data)) {
        stop(sprintf("Covariate `%s` not found in `data`.", v), call. = FALSE)
      }
      nUnique <- length(unique(stats::na.omit(data[[v]])))
      k <- sm$k[j]
      # Mirror mgcv's defaults: 10 for s(), 5 per margin for tensors.
      if (is.na(k) || k == -1) {
        k <- if (sm$type == "s") 10L else 5L
      }
      rows[[length(rows) + 1L]] <- data.frame(
        term = format(sm),
        var = v,
        k = k,
        unique = nUnique,
        ok = k <= nUnique,
        stringsAsFactors = FALSE
      )
    }
  }

  return(do.call(rbind, rows))
}

#' Fit a GAM from specifications
#'
#' Checks basis sizes, shrinks any that are too large, builds the formula and
#' fits it with [mgcv::gam()].
#'
#' @export
#' @param response Name of the response variable.
#' @param smooths A list of `pcSmooth` objects.
#' @param data A data frame containing the covariates.
#' @param family A family object passed to [mgcv::gam()].
#' @param method Smoothing parameter estimation method.
#' @param ... Passed to [mgcv::gam()].
#' @return A fitted `gam` object.
fitGamSpec <- function(
  response,
  smooths,
  data,
  family = stats::gaussian(),
  method = c("REML", "ML", "GCV.Cp"),
  ...
) {
  method <- match.arg(method)
  if (inherits(smooths, "pcSmooth")) {
    smooths <- list(smooths)
  }

  sizes <- checkBasisSize(smooths, data)
  if (!is.null(sizes) && !all(sizes$ok)) {
    tooBig <- sizes[!sizes$ok, , drop = FALSE]
    for (i in seq_along(smooths)) {
      sm <- smooths[[i]]
      hit <- tooBig$term == format(sm)
      if (any(hit)) {
        # Shrink only the offending margins; fill unset tensor margins with
        # the default so the rendered k vector has no NA in it.
        if (sm$type != "s") {
          sm$k[is.na(sm$k)] <- 5L
        }
        j <- match(tooBig$var[hit], sm$vars)
        sm$k[j] <- pmax(3L, tooBig$unique[hit] - 1L)
        smooths[[i]] <- sm
      }
    }
    warning(
      "Reduced basis size for: ",
      paste(unique(tooBig$term), collapse = ", ")
    )
  }

  f <- gamFormula(response, smooths)
  fit <- tryCatch(
    mgcv::gam(f, data = data, family = family, method = method, ...),
    error = function(e) {
      stop(
        sprintf(
          "GAM fit failed for `%s`: %s",
          deparse1(f),
          conditionMessage(e)
        ),
        call. = FALSE
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

## 45. `394ed97e9cd6f9c58e405a2b` finish_block / sibling:long

- package `synth:shiftplan`, path `R/packing.R`
- generator: `{"behaviour": "checked", "bucket": "long", "determinacy": "sibling", "full_body": false, "novel_tokens": ["which.min"], "novelty": 1, "rule": "fb_statement", "seed": "b19-graphs-or/shiftplan/R/packing.R", "signature_region": false, "structural_support": 0.861, "target_lines": 21, "test_file": false, "unit": "binpack.best.fit", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 172, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1448, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -151,3 +151,24 @@
 #' @export
 binpack.best.fit <- function(size, capacity, decreasing = TRUE) {
   if (any(size > capacity)) {
+    stop("Some items are larger than the bin capacity.", call. = FALSE)
+  }
+
+  ord <- if (decreasing) order(size, decreasing = TRUE) else seq_along(size)
+  bin <- integer(length(size))
+  free <- numeric(0)
+
+  for (i in ord) {
+    fit <- which(free >= size[i])
+    if (length(fit) > 0) {
+      # Tightest fit; which.min() keeps the earliest bin on ties
+      b <- fit[which.min(free[fit] - size[i])]
+    } else {
+      free <- c(free, capacity)
+      b <- length(free)
+    }
+    bin[i] <- b
+    free[b] <- free[b] - size[i]
+  }
+  bin
+}
```

**Target (gold region)**

```r
    stop("Some items are larger than the bin capacity.", call. = FALSE)
  }

  ord <- if (decreasing) order(size, decreasing = TRUE) else seq_along(size)
  bin <- integer(length(size))
  free <- numeric(0)

  for (i in ord) {
    fit <- which(free >= size[i])
    if (length(fit) > 0) {
      # Tightest fit; which.min() keeps the earliest bin on ties
      b <- fit[which.min(free[fit] - size[i])]
    } else {
      free <- c(free, capacity)
      b <- length(free)
    }
    bin[i] <- b
    free[b] <- free[b] - size[i]
  }
  bin
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/packing.R
#' @return A list with logical `chosen`, numeric `value` and integer `weight`.
#' @export
knapsack.dp <- function(value, weight, capacity) {
  stopifnot(length(value) == length(weight), capacity >= 0, all(weight >= 0))
  weight <- as.integer(weight)
  capacity <- as.integer(floor(capacity))
  n <- length(value)

  # tab[i + 1, c + 1] is the best value using the first i items within c
  tab <- matrix(0, nrow = n + 1L, ncol = capacity + 1L)
  for (i in seq_len(n)) {
    for (c in 0:capacity) {
      skip <- tab[i, c + 1L]
      if (weight[i] <= c) {
        take <- tab[i, c - weight[i] + 1L] + value[i]
        tab[i + 1L, c + 1L] <- max(skip, take)
      } else {
        tab[i + 1L, c + 1L] <- skip
      }
    }
  }

  # Walk back up the table: a changed cell means item i was taken
  chosen <- logical(n)
  c <- capacity
  for (i in rev(seq_len(n))) {
    if (tab[i + 1L, c + 1L] != tab[i, c + 1L]) {
      chosen[i] <- TRUE
      c <- c - weight[i]
    }
  }

  list(
    chosen = chosen,
    value = sum(value[chosen]),
    weight = sum(weight[chosen])
  )
}

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

  chosen <- sol$solution > 0.5
  list(
    chosen = chosen,
    value = sum(value[chosen]),
    weight = sum(weight[chosen])
  )
}

#' Pack items into bins with first fit
#'
#' Places each item in the first open bin with enough room, opening a new
#' bin when none fits.
#'
#' @param size Numeric vector of item sizes.
#' @param capacity Bin capacity.
#' @param decreasing Logical; sort items by decreasing size first (FFD).
#' @return An integer vector giving the bin of each item, in input order.
#' @export
binpack.first.fit <- function(size, capacity, decreasing = TRUE) {
  if (any(size > capacity)) {
    stop("Some items are larger than the bin capacity.", call. = FALSE)
  }

  ord <- if (decreasing) order(size, decreasing = TRUE) else seq_along(size)
  bin <- integer(length(size))
  # Remaining room in each open bin
  free <- numeric(0)

  for (i in ord) {
    fit <- which(free >= size[i])
    if (length(fit) > 0) {
      b <- fit[1]
    } else {
      free <- c(free, capacity)
      b <- length(free)
    }
    bin[i] <- b
    free[b] <- free[b] - size[i]
  }
  bin
}

#' Pack items into bins with best fit
#'
#' Places each item in the open bin that leaves the least room afterwards,
#' opening a new bin when none fits.
#'
#' @param size Numeric vector of item sizes.
#' @param capacity Bin capacity.
#' @param decreasing Logical; sort items by decreasing size first (BFD).
#' @return An integer vector giving the bin of each item, in input order.
#' @export
binpack.best.fit <- function(size, capacity, decreasing = TRUE) {
  if (any(size > capacity)) {
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

## 46. `07740d55cee9741e1295796f` finish_block / sibling:long

- package `synth:inkplate`, path `R/escape.R`
- generator: `{"behaviour": "checked", "bucket": "long", "determinacy": "sibling", "full_body": false, "novel_tokens": ["str:textasciicircum", "str:textasciitilde", "str:textbackslash"], "novelty": 3, "rule": "fb_comment_lead", "seed": "b10-text/inkplate/R/escape.R", "signature_region": false, "structural_support": 0.623, "target_lines": 21, "test_file": false, "unit": "escape.latex", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 178, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 851, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -88,3 +88,24 @@
 #' @export
 escape.latex <- function(x, na = "") {
   x <- as.character(x)
+  # backslash first, via a placeholder, so later braces are not re-escaped
+  out <- gsub("\\", "\001", x, fixed = TRUE)
+  out <- replace.by.table(
+    out,
+    c(
+      "{" = "\\{",
+      "}" = "\\}",
+      "$" = "\\$",
+      "&" = "\\&",
+      "%" = "\\%",
+      "#" = "\\#",
+      "_" = "\\_",
+      "~" = "\\textasciitilde{}",
+      "^" = "\\textasciicircum{}"
+    )
+  )
+  out <- gsub("\001", "\\textbackslash{}", out, fixed = TRUE)
+
+  out[is.na(x)] <- na
+  out
+}
```

**Target (gold region)**

```r
  # backslash first, via a placeholder, so later braces are not re-escaped
  out <- gsub("\\", "\001", x, fixed = TRUE)
  out <- replace.by.table(
    out,
    c(
      "{" = "\\{",
      "}" = "\\}",
      "$" = "\\$",
      "&" = "\\&",
      "%" = "\\%",
      "#" = "\\#",
      "_" = "\\_",
      "~" = "\\textasciitilde{}",
      "^" = "\\textasciicircum{}"
    )
  )
  out <- gsub("\001", "\\textbackslash{}", out, fixed = TRUE)

  out[is.na(x)] <- na
  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/escape.R
#' Replace characters using a lookup table
#'
#' Applies fixed-string replacements in the order given, which matters when
#' one replacement introduces characters another one would touch.
#'
#' @param x A character vector to escape.
#' @param table A named character vector mapping characters to replacements.
#' @noRd
replace.by.table <- function(x, table) {
  out <- x
  for (key in names(table)) {
    out <- gsub(key, table[[key]], out, fixed = TRUE)
  }
  out
}

#' Escape text for HTML
#'
#' `escape.html()` escapes text for element content. `escape.attr()` also
#' escapes both kinds of quote so the value can sit inside either quoting
#' style.
#'
#' @param x A character vector to escape.
#' @param na How to render missing values.
#' @return A character vector the same length as `x`.
#' @rdname escape.html
#' @export
escape.html <- function(x, na = "") {
  x <- as.character(x)
  # "&" must go first or the other entities would be escaped twice
  out <- replace.by.table(
    x,
    c(
      "&" = "&amp;",
      "<" = "&lt;",
      ">" = "&gt;"
    )
  )
  out[is.na(x)] <- na
  out
}

#' @rdname escape.html
#' @export
escape.attr <- function(x, na = "") {
  x <- as.character(x)
  out <- replace.by.table(
    x,
    c(
      "&" = "&amp;",
      "<" = "&lt;",
      ">" = "&gt;",
      "\"" = "&quot;",
      "'" = "&#39;"
    )
  )
  out[is.na(x)] <- na
  out
}

#' Escape text for inline Markdown
#'
#' Backslash-escapes characters that would otherwise start emphasis, code,
#' links or table cells.
#'
#' @param x A character vector to escape.
#' @param na How to render missing values.
#' @return A character vector the same length as `x`.
#' @export
escape.markdown <- function(x, na = "") {
  x <- as.character(x)
  out <- gsub("([\\\\`*_{}\\[\\]()#+!|<>~])", "\\\\\\1", x, perl = TRUE)

  # line-start markers that would turn the text into a list item; numbers
  # such as "1.50" or "-3" are not list markers and stay as they are
  out <- gsub("^([0-9]+)\\.(\\s|$)", "\\1\\\\.\\2", out, perl = TRUE)
  out <- gsub("^-(\\s|$)", "\\\\-\\1", out, perl = TRUE)

  out[is.na(x)] <- na
  out
}

#' Escape text for LaTeX
#'
#' @param x A character vector to escape.
#' @param na How to render missing values.
#' @return A character vector the same length as `x`.
#' @export
escape.latex <- function(x, na = "") {
  x <- as.character(x)
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

## 47. `fc40fc1dbb2b8ca1d71ac150` finish_block / constrained_short

- package `synth:panelwick`, path `R/lags.R`
- generator: `{"behaviour": "no_check", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["str:diff"], "novelty": 1, "rule": "fb_statement", "seed": "b09-finance/panelwick/R/lags.R", "signature_region": false, "structural_support": 0.432, "target_lines": 8, "test_file": false, "unit": "diffBy", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 71, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 643, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -50,3 +50,11 @@
 diffBy <- function(dt, vars, id, time, n = 1L) {
   checkPanelKeys(dt, id, time)
   data.table::setkeyv(dt, c(id, time))
+  outCols <- paste0(vars, "_diff", n)
+  dt[,
+    (outCols) := lapply(.SD, function(v) v - data.table::shift(v, n)),
+    by = id,
+    .SDcols = vars
+  ]
+  invisible(dt)
+}
```

**Target (gold region)**

```r
  outCols <- paste0(vars, "_diff", n)
  dt[,
    (outCols) := lapply(.SD, function(v) v - data.table::shift(v, n)),
    by = id,
    .SDcols = vars
  ]
  invisible(dt)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/lags.R
#' Lag or lead variables within each panel unit
#'
#' @description Rows are ordered by `time` within `id` before shifting, so
#'   the input order does not matter.
#'
#' @param dt A data.table in long panel format.
#' @param vars Character vector of columns to shift.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @param n Number of periods to shift by.
#' @return `dt`, keyed by `id` and `time` and modified by reference, invisibly.
#' @rdname shiftBy
#' @importFrom data.table := .SD
#' @export
lagBy <- function(dt, vars, id, time, n = 1L) {
  checkPanelKeys(dt, id, time)
  data.table::setkeyv(dt, c(id, time))
  outCols <- paste0(vars, "_lag", n)
  dt[,
    (outCols) := data.table::shift(.SD, n, type = "lag"),
    by = id,
    .SDcols = vars
  ]
  invisible(dt)
}

#' @rdname shiftBy
#' @export
leadBy <- function(dt, vars, id, time, n = 1L) {
  checkPanelKeys(dt, id, time)
  data.table::setkeyv(dt, c(id, time))
  outCols <- paste0(vars, "_lead", n)
  dt[,
    (outCols) := data.table::shift(.SD, n, type = "lead"),
    by = id,
    .SDcols = vars
  ]
  invisible(dt)
}

#' Difference variables within each panel unit
#'
#' @param dt A data.table in long panel format.
#' @param vars Character vector of columns to shift.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @param n Number of periods to shift by.
#' @return `dt`, keyed by `id` and `time` and modified by reference, invisibly.
#' @export
diffBy <- function(dt, vars, id, time, n = 1L) {
  checkPanelKeys(dt, id, time)
  data.table::setkeyv(dt, c(id, time))
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

## 48. `1b7d42af1857da7bc2466fe0` finish_block / constrained_short

- package `synth:markovine`, path `R/montecarlo.R`
- generator: `{"behaviour": "checked", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b23-simulation/markovine/R/montecarlo.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "mc.design", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 12, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 388, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -22,3 +22,5 @@
       seed = base.seed + 1000L * scenario,
       .before = 1
     )
+  return(grid)
+}
```

**Target (gold region)**

```r
  return(grid)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/montecarlo.R
#' @title Build a Monte Carlo design grid
#' @description Crosses all factor levels into a grid of scenarios and
#'   assigns each a reproducible seed.
#' @param ... Named vectors of factor levels, e.g. `n = c(20, 50)`.
#' @param base.seed Seed from which per-scenario seeds are derived.
#' @return A tibble with one row per scenario and columns `scenario`,
#'   `seed` plus one column per factor.
#' @export
mc.design <- function(..., base.seed = 1L) {
  factors <- list(...)
  if (
    length(factors) == 0 || is.null(names(factors)) || any(names(factors) == "")
  ) {
    stop("All design factors must be named.", call. = FALSE)
  }

  grid <- expand.grid(factors, stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)
  # Seeds are spaced 1000 apart so replicate offsets never collide.
  grid <- tibble::as_tibble(grid) |>
    dplyr::mutate(
      scenario = dplyr::row_number(),
      seed = base.seed + 1000L * scenario,
      .before = 1
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

## 49. `bf970712f60cf8d5533c78fb` roxygen_drafting / roxy_full

- package `synth:rangeweft`, path `R/reduce.R`
- generator: `{"behaviour": "checked", "determined_share": 0.667, "rule": "roxy_full", "seed": "b17-bioc/rangeweft/R/reduce.R", "tag_order": "export param return", "target_lines": 6, "unit": "setdiffSpans"}`
- verification: `{"air_idempotent": true, "completion_tokens": 67, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 821, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,6 +28,12 @@
   ))
 }
 
+#' Difference of two interval sets
+#'
+#' @export
+#' @param x,y `SpanSet` objects.
+#' @param ignoreStrand Should strand be ignored?
+#' @return A reduced `SpanSet` covering bases in `x` but not in `y`.
 setdiffSpans <- function(x, y, ignoreStrand = FALSE) {
   rx <- reduceSpans(x, ignoreStrand = ignoreStrand)
   ry <- reduceSpans(y, ignoreStrand = ignoreStrand)
```

**Target (gold region)**

```r
#' Difference of two interval sets
#'
#' @export
#' @param x,y `SpanSet` objects.
#' @param ignoreStrand Should strand be ignored?
#' @return A reduced `SpanSet` covering bases in `x` but not in `y`.
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/reduce.R
#'
#' @export
#' @param x,y `SpanSet` objects.
#' @param ignoreStrand Should strand be ignored?
#' @return A reduced `SpanSet` covering every base in both `x` and `y`.
intersectSpans <- function(x, y, ignoreStrand = FALSE) {
  rx <- reduceSpans(x, ignoreStrand = ignoreStrand)
  ry <- reduceSpans(y, ignoreStrand = ignoreStrand)

  # Strand must match exactly here: "*" is its own group after reduction.
  hits <- findSpanOverlaps(rx, ry, ignoreStrand = TRUE)
  if (!ignoreStrand) {
    hits <- hits[
      rx@strand[hits$queryHits] == ry@strand[hits$subjectHits],
      ,
      drop = FALSE
    ]
  }
  if (nrow(hits) == 0L) {
    return(SpanSet(character(), start = integer(), end = integer()))
  }

  return(SpanSet(
    seqnames = rx@seqnames[hits$queryHits],
    start = pmax(rx@start[hits$queryHits], ry@start[hits$subjectHits]),
    end = pmin(rx@end[hits$queryHits], ry@end[hits$subjectHits]),
    strand = rx@strand[hits$queryHits]
  ))
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
setdiffSpans <- function(x, y, ignoreStrand = FALSE) {
  rx <- reduceSpans(x, ignoreStrand = ignoreStrand)
  ry <- reduceSpans(y, ignoreStrand = ignoreStrand)

  # Same strand rule as intersectSpans(): "*" only cuts "*".
  hits <- findSpanOverlaps(rx, ry, ignoreStrand = TRUE)
  if (!ignoreStrand) {
    hits <- hits[
      rx@strand[hits$queryHits] == ry@strand[hits$subjectHits],
      ,
      drop = FALSE
    ]
  }

  outStart <- integer()
  outEnd <- integer()
  outIdx <- integer()
  for (i in seq_along(rx@start)) {
    cuts <- hits$subjectHits[hits$queryHits == i]
    cuts <- cuts[order(ry@start[cuts])]
    # Walk the cuts left to right, emitting the piece before each one.
    cursor <- rx@start[i]
    for (s in cuts) {
      if (ry@start[s] > cursor) {
        outStart <- c(outStart, cursor)
        outEnd <- c(outEnd, ry@start[s] - 1L)
        outIdx <- c(outIdx, i)
      }
      cursor <- max(cursor, ry@end[s] + 1L)
    }
    if (cursor <= rx@end[i]) {
      outStart <- c(outStart, cursor)
      outEnd <- c(outEnd, rx@end[i])
      outIdx <- c(outIdx, i)
    }
  }

  return(SpanSet(
    seqnames = rx@seqnames[outIdx],
    start = outStart,
    end = outEnd,
    strand = rx@strand[outIdx]
  ))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 50. `6b727b5dd3652648ab22639a` roxygen_drafting / roxy_full

- package `synth:apiclerk`, path `R/cache.R`
- generator: `{"behaviour": "checked", "determined_share": 0.667, "rule": "roxy_full", "seed": "b03-datatable-io/apiclerk/R/cache.R", "tag_order": "rdname param export", "target_lines": 3, "unit": "cache_set"}`
- verification: `{"air_idempotent": true, "completion_tokens": 31, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 465, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,6 +28,9 @@
   return(readRDS(file))
 }
 
+#' @rdname cache_has
+#' @param value Any R object to store.
+#' @export
 cache_set <- function(cache, key, value) {
   file <- file.path(cache$dir, paste0(key, ".rds"))
   # Write to a temporary name first so readers never see a partial file
```

**Target (gold region)**

```r
#' @rdname cache_has
#' @param value Any R object to store.
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/cache.R
#'
#' `cache_has()` checks whether a fresh entry exists, `cache_get()` reads it
#' and `cache_set()` stores a value. Entries older than the cache's TTL are
#' treated as missing.
#'
#' @param cache A `clerk_cache` object.
#' @param key A cache key as returned by `cache_key()`.
#' @return `cache_has()` returns `TRUE` if the entry exists and has not expired.
#' @export
cache_has <- function(cache, key) {
  file <- file.path(cache$dir, paste0(key, ".rds"))
  if (!file.exists(file)) {
    return(FALSE)
  }

  age <- as.numeric(difftime(Sys.time(), file.mtime(file), units = "secs"))
  return(age <= cache$ttl)
}

#' @rdname cache_has
#' @param default Value returned when the entry is missing or expired.
#' @export
cache_get <- function(cache, key, default = NULL) {
  if (!cache_has(cache, key)) {
    return(default)
  }
  file <- file.path(cache$dir, paste0(key, ".rds"))
  return(readRDS(file))
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
cache_set <- function(cache, key, value) {
  file <- file.path(cache$dir, paste0(key, ".rds"))
  # Write to a temporary name first so readers never see a partial file
  tmp <- paste0(file, ".tmp")
  saveRDS(value, tmp)
  file.rename(tmp, file)
  return(invisible(value))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 51. `8ac995256ac88579fcff2acb` roxygen_drafting / roxy_full

- package `synth:terraslab`, path `R/load.R`
- generator: `{"behaviour": "no_check", "determined_share": 0.8, "rule": "roxy_full", "seed": "b13-geo/terraslab/R/load.R", "tag_order": "param noRd", "target_lines": 5, "unit": "asRasterVector"}`
- verification: `{"air_idempotent": true, "completion_tokens": 67, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 465, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,6 +28,11 @@
   clipped[[which(nonNa > 0)]]
 }
 
+#' Convert an area of interest to a SpatVector in the raster CRS
+#'
+#' @param aoi An `sf` object or `SpatVector` giving the area of interest.
+#' @param r A `SpatRaster`.
+#' @noRd
 asRasterVector <- function(aoi, r) {
   v <- if (inherits(aoi, "SpatVector")) aoi else terra::vect(aoi)
   if (!terra::same.crs(v, r)) {
```

**Target (gold region)**

```r
#' Convert an area of interest to a SpatVector in the raster CRS
#'
#' @param aoi An `sf` object or `SpatVector` giving the area of interest.
#' @param r A `SpatRaster`.
#' @noRd
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/load.R
#' @param inverse If `TRUE`, mask cells inside the polygons instead.
#' @return A masked `SpatRaster`.
#' @export
maskToAoi <- function(r, aoi, inverse = FALSE) {
  v <- asRasterVector(aoi, r)
  # Cropping first keeps the mask step from touching the whole raster
  r %>%
    terra::crop(v, snap = "out") %>%
    terra::mask(v, inverse = inverse)
}

#' Clip a raster and drop empty layers
#'
#' Masks to the area of interest and removes layers that contain only `NA`
#' cells afterwards.
#'
#' @param r A `SpatRaster`.
#' @param aoi An `sf` object or `SpatVector` giving the area of interest.
#' @return A masked `SpatRaster`, possibly with fewer layers.
#' @export
clipAndPrune <- function(r, aoi) {
  clipped <- maskToAoi(r, aoi)
  nonNa <- terra::global(!is.na(clipped), "sum")[, 1]

  if (all(nonNa == 0)) {
    stop("Area of interest does not overlap any raster cells.", call. = FALSE)
  }
  clipped[[which(nonNa > 0)]]
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
asRasterVector <- function(aoi, r) {
  v <- if (inherits(aoi, "SpatVector")) aoi else terra::vect(aoi)
  if (!terra::same.crs(v, r)) {
    v <- terra::project(v, crs(r))
  }
  v
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 52. `3e0a1bfa990c03fb17337b8f` roxygen_drafting / roxy_full

- package `synth:taskrelay`, path `R/parallel.R`
- generator: `{"behaviour": "file_checked_unit_unexercised", "determined_share": 0.727, "rule": "roxy_full", "seed": "b14-pipelines/taskrelay/R/parallel.R", "tag_order": "keywords param return", "target_lines": 11, "unit": "par_map_furrr"}`
- verification: `{"air_idempotent": true, "completion_tokens": 125, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 472, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,6 +28,17 @@
   return(out)
 }
 
+#' Map over a list with future workers
+#'
+#' Uses [furrr::future_map()] with a reproducible seed and wraps each call
+#' so that errors come back as conditions instead of stopping the map.
+#'
+#' @keywords internal
+#' @param x A list or vector to iterate over.
+#' @param fun A function applied to each element of `x`.
+#' @param workers Number of workers to use.
+#' @param ... Passed to `fun`.
+#' @return A list the same length as `x`.
 par_map_furrr <- function(x, fun, workers, ...) {
   safe_fun <- function(el, ...) {
     tryCatch(fun(el, ...), error = function(e) e)
```

**Target (gold region)**

```r
#' Map over a list with future workers
#'
#' Uses [furrr::future_map()] with a reproducible seed and wraps each call
#' so that errors come back as conditions instead of stopping the map.
#'
#' @keywords internal
#' @param x A list or vector to iterate over.
#' @param fun A function applied to each element of `x`.
#' @param workers Number of workers to use.
#' @param ... Passed to `fun`.
#' @return A list the same length as `x`.
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/parallel.R
#' Map over a list with forked workers
#'
#' Uses [parallel::mclapply()] and turns `try-error` results into
#' conditions so callers can treat every backend the same way.
#'
#' @keywords internal
#' @param x A list or vector to iterate over.
#' @param fun A function applied to each element of `x`.
#' @param workers Number of workers to use.
#' @param ... Passed to `fun`.
#' @return A list the same length as `x`.
par_map_mclapply <- function(x, fun, workers, ...) {
  if (.Platform$OS.type == "windows") {
    workers <- 1L
  }

  out <- parallel::mclapply(
    x,
    fun,
    ...,
    mc.cores = workers,
    mc.preschedule = TRUE
  )
  failed <- vapply(out, inherits, logical(1), what = "try-error")
  out[failed] <- lapply(out[failed], function(e) attr(e, "condition"))

  names(out) <- names(x)
  return(out)
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
par_map_furrr <- function(x, fun, workers, ...) {
  safe_fun <- function(el, ...) {
    tryCatch(fun(el, ...), error = function(e) e)
  }

  out <- with_workers(
    furrr::future_map(
      x,
      safe_fun,
      ...,
      .options = furrr::furrr_options(seed = TRUE)
    ),
    strategy = "multisession",
    workers = workers
  )
  names(out) <- names(x)
  return(out)
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 53. `15b444946f6a9d823be180a3` roxygen_drafting / roxy_full

- package `synth:taskrelay`, path `R/logging.R`
- generator: `{"behaviour": "checked", "determined_share": 0.714, "rule": "roxy_full", "seed": "b14-pipelines/taskrelay/R/logging.R", "tag_order": "export inheritParams rdname", "target_lines": 7, "unit": "log_error"}`
- verification: `{"air_idempotent": true, "completion_tokens": 57, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 317, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -21,6 +21,13 @@
   return(log_at(logger, "WARN", fmt, ...))
 }
 
+#' Log an error message
+#'
+#' Shorthand for `log_at(logger, "ERROR", ...)`.
+#'
+#' @export
+#' @inheritParams log_at
+#' @rdname log_shortcuts
 log_error <- function(logger, fmt, ...) {
   return(log_at(logger, "ERROR", fmt, ...))
 }
```

**Target (gold region)**

```r
#' Log an error message
#'
#' Shorthand for `log_at(logger, "ERROR", ...)`.
#'
#' @export
#' @inheritParams log_at
#' @rdname log_shortcuts
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/logging.R
#' Log an informational message
#'
#' Shorthand for `log_at(logger, "INFO", ...)`.
#'
#' @export
#' @inheritParams log_at
#' @return `TRUE` if the line was emitted, `FALSE` otherwise, invisibly.
#' @rdname log_shortcuts
log_info <- function(logger, fmt, ...) {
  return(log_at(logger, "INFO", fmt, ...))
}

#' Log a warning message
#'
#' Shorthand for `log_at(logger, "WARN", ...)`.
#'
#' @export
#' @inheritParams log_at
#' @rdname log_shortcuts
log_warn <- function(logger, fmt, ...) {
  return(log_at(logger, "WARN", fmt, ...))
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
log_error <- function(logger, fmt, ...) {
  return(log_at(logger, "ERROR", fmt, ...))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 54. `42998cb7a06224e2a9ddbe6e` roxygen_drafting / roxy_full

- package `synth:foliokit`, path `R/money-vctrs.R`
- generator: `{"behaviour": "no_check", "determined_share": 0.778, "rule": "roxy_full", "seed": "b05-oop/foliokit/R/money-vctrs.R", "tag_order": "param return keywords export", "target_lines": 9, "unit": "vec_ptype2.foliokit_money.double"}`
- verification: `{"air_idempotent": true, "completion_tokens": 84, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 264, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -11,6 +11,15 @@
   new_money(currency = check_same_currency(x, y))
 }
 
+#' Common type of money and a double vector
+#'
+#' Plain doubles are treated as amounts in the currency of the money vector.
+#'
+#' @param x,y Vectors to combine.
+#' @param ... Ignored.
+#' @return A zero-length `foliokit_money` prototype.
+#' @keywords internal
+#' @export
 vec_ptype2.foliokit_money.double <- function(x, y, ...) {
   new_money(currency = attr(x, "currency"))
 }
```

**Target (gold region)**

```r
#' Common type of money and a double vector
#'
#' Plain doubles are treated as amounts in the currency of the money vector.
#'
#' @param x,y Vectors to combine.
#' @param ... Ignored.
#' @return A zero-length `foliokit_money` prototype.
#' @keywords internal
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/money-vctrs.R
#' Common type of two money vectors
#'
#' Two money vectors combine only if they share a currency.
#'
#' @param x,y Vectors to combine.
#' @param ... Ignored.
#' @return A zero-length `foliokit_money` prototype.
#' @keywords internal
#' @export
vec_ptype2.foliokit_money.foliokit_money <- function(x, y, ...) {
  new_money(currency = check_same_currency(x, y))
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
vec_ptype2.foliokit_money.double <- function(x, y, ...) {
  new_money(currency = attr(x, "currency"))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 55. `49de261ea62d1d10aa8b83bc` roxygen_drafting / roxy_full

- package `synth:foliokit`, path `R/money-vctrs.R`
- generator: `{"behaviour": "no_check", "determined_share": 0.75, "rule": "roxy_full", "seed": "b05-oop/foliokit/R/money-vctrs.R", "tag_order": "param return keywords export", "target_lines": 8, "unit": "vec_cast.double.foliokit_money"}`
- verification: `{"air_idempotent": true, "completion_tokens": 71, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 395, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,6 +28,14 @@
   money(x, currency = attr(to, "currency"))
 }
 
+#' Cast money to a double vector
+#'
+#' @param x Vector to cast.
+#' @param to Prototype to cast to.
+#' @param ... Ignored.
+#' @return A double vector of amounts with the currency dropped.
+#' @keywords internal
+#' @export
 vec_cast.double.foliokit_money <- function(x, to, ...) {
   vctrs::vec_data(x)
 }
```

**Target (gold region)**

```r
#' Cast money to a double vector
#'
#' @param x Vector to cast.
#' @param to Prototype to cast to.
#' @param ... Ignored.
#' @return A double vector of amounts with the currency dropped.
#' @keywords internal
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/money-vctrs.R
#' @export
vec_ptype2.double.foliokit_money <- function(x, y, ...) {
  new_money(currency = attr(y, "currency"))
}

#' Cast money to money
#'
#' @param x Vector to cast.
#' @param to Prototype to cast to.
#' @param ... Ignored.
#' @return A `foliokit_money` vector in the currency of `to`.
#' @keywords internal
#' @export
vec_cast.foliokit_money.foliokit_money <- function(x, to, ...) {
  check_same_currency(x, to)
  x
}

#' Cast a double vector to money
#'
#' @param x Vector to cast.
#' @param to Prototype to cast to.
#' @param ... Ignored.
#' @return A `foliokit_money` vector in the currency of `to`.
#' @keywords internal
#' @export
vec_cast.foliokit_money.double <- function(x, to, ...) {
  money(x, currency = attr(to, "currency"))
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
vec_cast.double.foliokit_money <- function(x, to, ...) {
  vctrs::vec_data(x)
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 56. `8375b8c8986df690f4732e7b` roxygen_drafting / roxy_full

- package `synth:rangeweft`, path `R/overlaps.R`
- generator: `{"behaviour": "checked", "determined_share": 0.667, "rule": "roxy_full", "seed": "b17-bioc/rangeweft/R/overlaps.R", "tag_order": "export inheritParams param return", "target_lines": 6, "unit": "overlapsAnySpan"}`
- verification: `{"air_idempotent": true, "completion_tokens": 65, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 273, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -10,6 +10,12 @@
   return(counts)
 }
 
+#' Does each query interval overlap any subject interval?
+#'
+#' @export
+#' @inheritParams findSpanOverlaps
+#' @param ... Passed to [findSpanOverlaps()].
+#' @return A logical vector with one element per query interval.
 overlapsAnySpan <- function(query, subject, ...) {
   hits <- findSpanOverlaps(query, subject, ...)
   out <- seq_along(query@start) %in% hits$queryHits
```

**Target (gold region)**

```r
#' Does each query interval overlap any subject interval?
#'
#' @export
#' @inheritParams findSpanOverlaps
#' @param ... Passed to [findSpanOverlaps()].
#' @return A logical vector with one element per query interval.
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/overlaps.R
#' Count overlaps per query interval
#'
#' @export
#' @inheritParams findSpanOverlaps
#' @param ... Passed to [findSpanOverlaps()].
#' @return An integer vector with one count per query interval.
countSpanOverlaps <- function(query, subject, ...) {
  hits <- findSpanOverlaps(query, subject, ...)
  counts <- tabulate(hits$queryHits, nbins = length(query@start))
  return(counts)
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
overlapsAnySpan <- function(query, subject, ...) {
  hits <- findSpanOverlaps(query, subject, ...)
  out <- seq_along(query@start) %in% hits$queryHits
  return(out)
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 57. `ed8ed4cc2b1fd7707dbbb322` roxygen_drafting / roxy_full

- package `synth:taskrota`, path `R/hooks.R`
- generator: `{"behaviour": "no_check", "determined_share": 0.667, "rule": "roxy_full", "seed": "b05-oop/taskrota/R/hooks.R", "tag_order": "noRd", "target_lines": 3, "unit": "hook_events"}`
- verification: `{"air_idempotent": true, "completion_tokens": 20, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 139, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -1,3 +1,6 @@
+#' Supported hook events
+#'
+#' @noRd
 hook_events <- function() {
   c("job_started", "job_succeeded", "job_failed", "queue_empty")
 }
```

**Target (gold region)**

```r
#' Supported hook events
#'
#' @noRd
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/hooks.R
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
hook_events <- function() {
  c("job_started", "job_succeeded", "job_failed", "queue_empty")
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 58. `8e50f7da258b95fbb91a7842` roxygen_drafting / roxy_full

- package `synth:relaydesk`, path `R/validate.R`
- generator: `{"behaviour": "checked", "determined_share": 0.667, "rule": "roxy_full", "seed": "b18-web/relaydesk/R/validate.R", "tag_order": "rdname param keywords", "target_lines": 3, "unit": "requireChoice"}`
- verification: `{"air_idempotent": true, "completion_tokens": 32, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 513, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -23,6 +23,9 @@
   list(value = as.integer(number), error = NULL)
 }
 
+#' @rdname requireString
+#' @param choices Character vector of allowed values.
+#' @keywords internal
 requireChoice <- function(value, name, choices) {
   if (is.null(value) || length(value) != 1 || is.na(value)) {
     return(list(value = NULL, error = sprintf("`%s` is required", name)))
```

**Target (gold region)**

```r
#' @rdname requireString
#' @param choices Character vector of allowed values.
#' @keywords internal
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/validate.R
#' @rdname requireString
#' @param min,max Inclusive bounds for the value.
#' @keywords internal
requireInteger <- function(value, name, min = 1L, max = .Machine$integer.max) {
  if (is.null(value) || length(value) != 1 || is.na(value)) {
    return(list(value = NULL, error = sprintf("`%s` is required", name)))
  }

  # Query parameters arrive as strings, so parse before checking the range.
  number <- suppressWarnings(as.numeric(value))
  if (is.na(number) || number != round(number)) {
    return(list(
      value = NULL,
      error = sprintf("`%s` must be a whole number", name)
    ))
  }
  if (number < min || number > max) {
    return(list(
      value = NULL,
      error = sprintf("`%s` must be between %d and %d", name, min, max)
    ))
  }
  list(value = as.integer(number), error = NULL)
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
requireChoice <- function(value, name, choices) {
  if (is.null(value) || length(value) != 1 || is.na(value)) {
    return(list(value = NULL, error = sprintf("`%s` is required", name)))
  }

  # Match case-insensitively but hand back the canonical spelling.
  value <- tolower(trimws(as.character(value)))
  hit <- match(value, tolower(choices))
  if (is.na(hit)) {
    allowed <- paste(choices, collapse = ", ")
    return(list(
      value = NULL,
      error = sprintf("`%s` must be one of: %s", name, allowed)
    ))
  }
  list(value = choices[[hit]], error = NULL)
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 59. `1305f77b1a121b0d0d666a5e` roxygen_drafting / roxy_full

- package `synth:trophicstan`, path `R/standata.R`
- generator: `{"behaviour": "checked", "determined_share": 0.8, "rule": "roxy_full", "seed": "b08-rcpp-science/trophicstan/R/standata.R", "tag_order": "param noRd", "target_lines": 5, "unit": "check_exposure"}`
- verification: `{"air_idempotent": true, "completion_tokens": 47, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 380, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -21,6 +21,11 @@
   return(as.integer(x))
 }
 
+#' Check a vector of positive exposures
+#'
+#' @param x Vector to check.
+#' @param arg Name of the argument, used in error messages.
+#' @noRd
 check_exposure <- function(x, arg) {
   if (!is.numeric(x) || anyNA(x)) {
     stop(
```

**Target (gold region)**

```r
#' Check a vector of positive exposures
#'
#' @param x Vector to check.
#' @param arg Name of the argument, used in error messages.
#' @noRd
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/standata.R
#' Check a vector of non-negative counts
#'
#' @param x Vector to check.
#' @param arg Name of the argument, used in error messages.
#' @return `x` as an integer vector, if all checks pass.
#' @noRd
check_counts <- function(x, arg) {
  if (!is.numeric(x) || anyNA(x)) {
    stop(
      sprintf("`%s` must be a numeric vector without missing values.", arg),
      call. = FALSE
    )
  }
  if (any(x < 0) || any(x != round(x))) {
    stop(
      sprintf("`%s` must contain non-negative whole numbers.", arg),
      call. = FALSE
    )
  }

  return(as.integer(x))
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
check_exposure <- function(x, arg) {
  if (!is.numeric(x) || anyNA(x)) {
    stop(
      sprintf("`%s` must be a numeric vector without missing values.", arg),
      call. = FALSE
    )
  }
  if (any(x <= 0)) {
    stop(sprintf("`%s` must be strictly positive.", arg), call. = FALSE)
  }
  return(as.double(x))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 60. `19aa1b5d130b9e3e4149c555` roxygen_drafting / roxy_full

- package `synth:diffledger`, path `R/de-results.R`
- generator: `{"behaviour": "no_check", "determined_share": 0.714, "rule": "roxy_full", "seed": "b17-bioc/diffledger/R/de-results.R", "tag_order": "param return export", "target_lines": 7, "unit": "summarise_de"}`
- verification: `{"air_idempotent": true, "completion_tokens": 82, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 459, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,6 +28,13 @@
     )
 }
 
+#' Count features per direction
+#'
+#' @param de A tibble of standardised differential-expression results.
+#' @param alpha Significance threshold on adjusted p-values.
+#' @param lfc_min Minimum absolute log2 fold change.
+#' @return A tibble with one row per direction and columns `n` and `pct`.
+#' @export
 summarise_de <- function(de, alpha = 0.05, lfc_min = 1) {
   de |>
     classify_de(alpha = alpha, lfc_min = lfc_min) |>
```

**Target (gold region)**

```r
#' Count features per direction
#'
#' @param de A tibble of standardised differential-expression results.
#' @param alpha Significance threshold on adjusted p-values.
#' @param lfc_min Minimum absolute log2 fold change.
#' @return A tibble with one row per direction and columns `n` and `pct`.
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/de-results.R
#' Classify features as up, down or unchanged
#'
#' @description Features without an adjusted p-value (for example those
#'   removed by independent filtering) are labelled `"untested"`.
#'
#' @param de A tibble of standardised differential-expression results.
#' @param alpha Significance threshold on adjusted p-values.
#' @param lfc_min Minimum absolute log2 fold change.
#' @return `de` with a factor column `direction`.
#' @export
classify_de <- function(de, alpha = 0.05, lfc_min = 1) {
  if (!"padj" %in% names(de)) {
    de <- add_padj(de)
  }

  de |>
    dplyr::mutate(
      direction = dplyr::case_when(
        is.na(padj) ~ "untested",
        padj < alpha & log2fc >= lfc_min ~ "up",
        padj < alpha & log2fc <= -lfc_min ~ "down",
        TRUE ~ "unchanged"
      ),
      direction = factor(
        direction,
        levels = c("up", "down", "unchanged", "untested")
      )
    )
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
summarise_de <- function(de, alpha = 0.05, lfc_min = 1) {
  de |>
    classify_de(alpha = alpha, lfc_min = lfc_min) |>
    dplyr::count(direction, .drop = FALSE) |>
    dplyr::mutate(pct = round(100 * n / sum(n), 1))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>
