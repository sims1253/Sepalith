# SYNTH-v3.1 audit sample

Bound to `/mnt/e/sepalith/campaign-20260915/data-work/SYNTH-v3.1/manifest.json` sha256 `c15d1d8321ea6987ea79898875d4865dbb019378c09e9c466b0850b777199538` (build-summary.json `f02f45c878e3c536d0ee3aacd028b5811e110607d731d18c4c7ac5335172cf86`).

Stratified random sample (seed 20261022) of 60 of 1515 cases. Strata: {"finish_block": {"constrained_short": 22, "sibling:short": 6, "sibling:medium": 16, "sibling:long": 4}, "roxygen_drafting": {"roxy_full": 12}}.

Each case shows the edited window as a diff (prefix + region + suffix, before and after the gold edit),
the gold target, the verification record and the exact rendered prompt of the drop-in TRAIN row.

## 1. `905daa0dd0d1ec5050671757` finish_block / constrained_short

- package `synth:confloom`, path `R/dcf.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b30-config/confloom/R/dcf.R", "signature_region": false, "structural_support": 0.0, "target_lines": 3, "test_file": false, "unit": "validateJobRecord", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 12, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 550, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -35,3 +35,6 @@
   # Enabled: optional, yes or no
   if (!is.null(job$Enabled) && !is.logical(job$Enabled)) {
     problems <- c(problems, "Field 'Enabled' must be yes or no.")
+  }
+  problems
+}
```

**Target (gold region)**

```r
  }
  problems
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/dcf.R
validateJobRecord <- function(job) {
  problems <- character()

  # Name: required, a single identifier
  if (is.null(job$Name)) {
    problems <- c(problems, "Field 'Name' is missing.")
  } else if (!grepl("^[A-Za-z][A-Za-z0-9_.-]*$", job$Name)) {
    problems <- c(problems, "Field 'Name' is not a valid identifier.")
  }

  # Command: required, non-empty text
  if (is.null(job$Command)) {
    problems <- c(problems, "Field 'Command' is missing.")
  } else if (!nzchar(trimws(job$Command))) {
    problems <- c(problems, "Field 'Command' is empty.")
  }

  # Schedule: required, one of the known keywords
  if (is.null(job$Schedule)) {
    problems <- c(problems, "Field 'Schedule' is missing.")
  } else if (!job$Schedule %in% c("hourly", "daily", "weekly", "monthly")) {
    problems <- c(problems, "Field 'Schedule' is not a known schedule.")
  }

  # Retries: optional, a non-negative integer
  if (!is.null(job$Retries) && (!is.numeric(job$Retries) || job$Retries < 0)) {
    problems <- c(problems, "Field 'Retries' must be a non-negative number.")
  }

  # Timeout: optional, a positive number
  if (!is.null(job$Timeout) && (!is.numeric(job$Timeout) || job$Timeout <= 0)) {
    problems <- c(problems, "Field 'Timeout' must be a positive number.")
  }

  # Enabled: optional, yes or no
  if (!is.null(job$Enabled) && !is.logical(job$Enabled)) {
    problems <- c(problems, "Field 'Enabled' must be yes or no.")
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

## 2. `6f2b023696cf4adb2008f6f0` finish_block / constrained_short

- package `synth:tileweave`, path `R/quadkeys.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["elt:path"], "novelty": 1, "rule": "fb_statement", "seed": "b42-geo/tileweave/R/quadkeys.R", "signature_region": false, "structural_support": 0.0, "target_lines": 6, "test_file": false, "unit": "describeTileKeys", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 63, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 387, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -27,3 +27,9 @@
   )
 
   # Column quadkey
+  out$quadkey <- encodeTileKey(tiles$x, tiles$y, tiles$zoom, "quadkey")
+
+  # Column path
+  out$path <- encodeTileKey(tiles$x, tiles$y, tiles$zoom, "path")
+  return(out)
+}
```

**Target (gold region)**

```r
  out$quadkey <- encodeTileKey(tiles$x, tiles$y, tiles$zoom, "quadkey")

  # Column path
  out$path <- encodeTileKey(tiles$x, tiles$y, tiles$zoom, "path")
  return(out)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/quadkeys.R
#' Describe a vector of keys
#'
#' @param key A character vector of keys.
#' @param format Key format accepted by [decodeTileKey()].
#' @return A data frame with columns `key`, `x`, `y`, `zoom`, `parent`,
#'   `quadkey` and `path`, where `parent`, `quadkey` and `path` are keys in
#'   their own formats (`parent` in `format`).
#' @export
describeTileKeys <- function(key, format = "quadkey") {
  isTileKey(key, format, error = TRUE)
  tiles <- decodeTileKey(key, format)
  out <- data.frame(key = key, stringsAsFactors = FALSE)

  # Columns x, y, zoom
  out$x <- tiles$x
  out$y <- tiles$y
  out$zoom <- tiles$zoom

  # Column parent
  hasParent <- tiles$zoom > 0
  out$parent <- NA_character_
  out$parent[hasParent] <- encodeTileKey(
    tiles$x[hasParent] %/% 2,
    tiles$y[hasParent] %/% 2,
    tiles$zoom[hasParent] - 1,
    format
  )

  # Column quadkey
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

## 3. `42055a4900da04eb00164325` finish_block / constrained_short

- package `synth:terraslab`, path `R/tiles.R`
- generator: `{"behaviour": "no_check", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["merge", "sprc"], "novelty": 2, "rule": "fb_statement", "seed": "b13-geo/terraslab/R/tiles.R", "signature_region": false, "structural_support": 0.0, "target_lines": 8, "test_file": false, "unit": "applyByTile", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 63, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 207, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -7,3 +7,11 @@
     padded <- terra::crop(r, tileExtent(r, tile, buffer = buffer))
     result <- fun(padded)
     # Trim back to the unbuffered tile so neighbours do not overlap
+    pieces[[i]] <- terra::crop(result, tileExtent(r, tile))
+  }
+
+  if (length(pieces) == 1) {
+    return(pieces[[1]])
+  }
+  return(terra::merge(terra::sprc(pieces)))
+}
```

**Target (gold region)**

```r
    pieces[[i]] <- terra::crop(result, tileExtent(r, tile))
  }

  if (length(pieces) == 1) {
    return(pieces[[1]])
  }
  return(terra::merge(terra::sprc(pieces)))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/tiles.R
applyByTile <- function(r, fun, buffer = 0, ...) {
  scheme <- tileScheme(r, ...)
  pieces <- vector("list", nrow(scheme))

  for (i in seq_len(nrow(scheme))) {
    tile <- scheme[i, ]
    padded <- terra::crop(r, tileExtent(r, tile, buffer = buffer))
    result <- fun(padded)
    # Trim back to the unbuffered tile so neighbours do not overlap
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

## 4. `10f308084989b80bdfbd52ab` finish_block / constrained_short

- package `synth:respkit`, path `R/records.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["col", "values"], "novelty": 2, "rule": "fb_statement", "seed": "b41-webapi/respkit/R/records.R", "signature_region": false, "structural_support": 0.0, "target_lines": 7, "test_file": false, "unit": "records.to.frame", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 65, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 199, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -5,3 +5,10 @@
   flat <- lapply(items, flatten.record, sep = sep, collapse = collapse)
   columns <- unique(unlist(lapply(flat, names), use.names = FALSE))
   out <- data.frame(row.names = seq_along(flat))
+  for (col in columns) {
+    values <- lapply(flat, function(r) if (is.null(r[[col]])) NA else r[[col]])
+    out[[col]] <- unlist(values, use.names = FALSE)
+  }
+
+  out
+}
```

**Target (gold region)**

```r
  for (col in columns) {
    values <- lapply(flat, function(r) if (is.null(r[[col]])) NA else r[[col]])
    out[[col]] <- unlist(values, use.names = FALSE)
  }

  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/records.R
records.to.frame <- function(items, sep = ".", collapse = ",") {
  if (length(items) == 0L) {
    return(data.frame())
  }
  flat <- lapply(items, flatten.record, sep = sep, collapse = collapse)
  columns <- unique(unlist(lapply(flat, names), use.names = FALSE))
  out <- data.frame(row.names = seq_along(flat))
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

## 5. `60e6f319dbaaa4267db81232` finish_block / constrained_short

- package `synth:tidyscrub`, path `R/impute.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["source"], "novelty": 1, "rule": "fb_statement", "seed": "b25-validation/tidyscrub/R/impute.R", "signature_region": false, "structural_support": 0.0, "target_lines": 7, "test_file": false, "unit": "fill.forward", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 76, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 259, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -13,3 +13,10 @@
     return(x)
   }
   index <- findInterval(seq_along(x), observed)
+  source <- rep(NA_integer_, length(x))
+  source[index > 0L] <- observed[index]
+  gap <- seq_along(x) - source
+  fill <- is.na(x) & !is.na(source) & gap <= max.gap
+  x[fill] <- x[source[fill]]
+  x
+}
```

**Target (gold region)**

```r
  source <- rep(NA_integer_, length(x))
  source[index > 0L] <- observed[index]
  gap <- seq_along(x) - source
  fill <- is.na(x) & !is.na(source) & gap <= max.gap
  x[fill] <- x[source[fill]]
  x
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/impute.R
#' Carry values forward or backward over gaps
#'
#' @param x A vector with missing values.
#' @param max.gap Longest run of missing values to fill; longer runs are
#'   left as they are.
#' @return `x` with gaps filled from the previous (`fill.forward()`) or the
#'   next (`fill.backward()`) observed value.
#' @rdname fill
#' @export
fill.forward <- function(x, max.gap = Inf) {
  observed <- which(!is.na(x))
  if (length(observed) == 0L) {
    return(x)
  }
  index <- findInterval(seq_along(x), observed)
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

## 6. `5afe43a39b486f862358965f` finish_block / constrained_short

- package `synth:quaryl`, path `R/retry.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b41-webapi/quaryl/R/retry.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "http_condition", "variant": "outline"}`
- verification: `{"air_idempotent": true, "completion_tokens": 12, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 427, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -36,3 +36,5 @@
       retry_after = retry_after
     )
   )
+  stop(cond)
+}
```

**Target (gold region)**

```r
  stop(cond)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/retry.R
# ---- outline --------------------------------
# fn: http_condition(status, message, retry_after)
# plan: 5 top-level statements; 4 typed above the cursor
# docs: yes
# ----------------------------------------------
#' Signal a classed HTTP error
#'
#' @param status HTTP status code, or `NA` when no response arrived.
#' @param message Error message.
#' @param retry_after Seconds from a `Retry-After` header, or `NA`.
#' @return Does not return; signals a condition with classes
#'   `quaryl_<error class>`, `quaryl_http_error`, `error` and `condition`.
#' @keywords internal
http_condition <- function(status, message = "", retry_after = NA_real_) {
  error_class <- classify_http_error(status, message)
  if (identical(error_class, "ok")) {
    stop("`status` does not describe an error.")
  }
  text <- if (is.na(status)) {
    message
  } else {
    sprintf("HTTP %d: %s", status, message)
  }
  cond <- structure(
    class = c(
      paste0("quaryl_", error_class),
      "quaryl_http_error",
      "error",
      "condition"
    ),
    list(
      message = text,
      call = NULL,
      status = status,
      error_class = error_class,
      retry_after = retry_after
    )
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

## 7. `5dfb493efed7f98774198c69` finish_block / constrained_short

- package `synth:scenariokit`, path `R/scenarios.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b38-simulation/scenariokit/R/scenarios.R", "signature_region": false, "structural_support": 0.0, "target_lines": 5, "test_file": false, "unit": "scenario.grid", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 39, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 225, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -9,3 +9,8 @@
   for (i in seq_len(nrow(grid))) {
     p <- as.list(grid[i, ])
     p$label <- sprintf("beta=%g/gamma=%g/immune=%g", p$beta, p$gamma, p$immune)
+    out[[i]] <- p
+  }
+  names(out) <- vapply(out, `[[`, character(1), "label")
+  out
+}
```

**Target (gold region)**

```r
    out[[i]] <- p
  }
  names(out) <- vapply(out, `[[`, character(1), "label")
  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/scenarios.R
scenario.grid <- function(beta, gamma, immune = 0) {
  grid <- expand.grid(
    beta = beta,
    gamma = gamma,
    immune = immune,
    KEEP.OUT.ATTRS = FALSE
  )
  out <- vector("list", nrow(grid))
  for (i in seq_len(nrow(grid))) {
    p <- as.list(grid[i, ])
    p$label <- sprintf("beta=%g/gamma=%g/immune=%g", p$beta, p$gamma, p$immune)
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

## 8. `0089f5e4316773b3d4da1623` finish_block / constrained_short

- package `synth:pagesieve`, path `R/extract.R`
- generator: `{"behaviour": "file_checked_not_entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["as.data.frame"], "novelty": 1, "rule": "fb_nested_close", "seed": "b18-web/pagesieve/R/extract.R", "signature_region": false, "structural_support": 0.0, "target_lines": 5, "test_file": false, "unit": "extract_records", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 24, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 397, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -32,3 +32,8 @@
       if (!is.na(value) && nzchar(value)) {
         out[i, j] <- value
       }
+    }
+  }
+
+  as.data.frame(out, stringsAsFactors = FALSE)
+}
```

**Target (gold region)**

```r
    }
  }

  as.data.frame(out, stringsAsFactors = FALSE)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/extract.R
extract_records <- function(html, item_css, fields) {
  if (is.null(names(fields)) || any(!nzchar(names(fields)))) {
    stop("`fields` must be a fully named character vector.", call. = FALSE)
  }

  items <- rvest::html_elements(html, item_css)
  specs <- lapply(fields, parse_field_spec)
  out <- matrix(
    NA_character_,
    nrow = length(items),
    ncol = length(specs),
    dimnames = list(NULL, names(fields))
  )

  for (i in seq_along(items)) {
    for (j in seq_along(specs)) {
      spec <- specs[[j]]
      node <- if (is.null(spec$css)) {
        items[[i]]
      } else {
        rvest::html_element(items[[i]], spec$css)
      }
      if (inherits(node, "xml_missing")) {
        next
      }

      value <- if (is.null(spec$attr)) {
        rvest::html_text2(node)
      } else {
        rvest::html_attr(node, spec$attr)
      }
      if (!is.na(value) && nzchar(value)) {
        out[i, j] <- value
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

## 9. `f61d7232580f0354f8e778a9` finish_block / constrained_short

- package `synth:termsmith`, path `R/marginal.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b24-modelling/termsmith/R/marginal.R", "signature_region": false, "structural_support": 0.0, "target_lines": 4, "test_file": false, "unit": "predict_safely", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 17, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 244, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -10,3 +10,7 @@
     warning = function(w) {
       warning(sprintf("Prediction warning: %s", conditionMessage(w)))
       suppressWarnings(stats::predict(model, newdata = newdata, type = type))
+    }
+  )
+  as.numeric(pred)
+}
```

**Target (gold region)**

```r
    }
  )
  as.numeric(pred)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/marginal.R
predict_safely <- function(model, newdata, type = c("response", "link")) {
  type <- match.arg(type)

  pred <- tryCatch(
    stats::predict(model, newdata = newdata, type = type),
    error = function(e) {
      stop(sprintf("Prediction failed: %s", conditionMessage(e)))
    },
    # Re-raise the warning with a prefix, then redo the prediction quietly.
    warning = function(w) {
      warning(sprintf("Prediction warning: %s", conditionMessage(w)))
      suppressWarnings(stats::predict(model, newdata = newdata, type = type))
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

## 10. `30020e63afbcfd7cd97def82` finish_block / constrained_short

- package `synth:confloom`, path `R/ini.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["writeLines"], "novelty": 1, "rule": "fb_statement", "seed": "b30-config/confloom/R/ini.R", "signature_region": false, "structural_support": 0.676, "target_lines": 8, "test_file": false, "unit": "writeIniConfig", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 52, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 465, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -34,3 +34,11 @@
       lines <- c(lines, "")
     }
     lines <- c(lines, sprintf("[%s]", section))
+    if (length(config[[section]]) > 0) {
+      lines <- c(lines, formatPairs(config[[section]]))
+    }
+  }
+
+  writeLines(lines, path)
+  invisible(path)
+}
```

**Target (gold region)**

```r
    if (length(config[[section]]) > 0) {
      lines <- c(lines, formatPairs(config[[section]]))
    }
  }

  writeLines(lines, path)
  invisible(path)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/ini.R
#' Write an INI-like configuration file
#'
#' @description
#' The inverse of [readIniConfig()]. The section named by `global` is
#' written first without a header; every other section gets a `[name]`
#' header followed by its `key = value` lines. Values containing newlines
#' are written as indented continuation lines.
#'
#' @param config A named list of sections, each a named list of values.
#' @param path Path to the file.
#' @param global Name of the section for keys before the first header.
#' @param header Optional comment lines written at the top of the file.
#' @export
writeIniConfig <- function(config, path, global = "global", header = NULL) {
  formatPairs <- function(values) {
    values <- vapply(
      values,
      function(v) paste(format(v), collapse = ", "),
      character(1)
    )
    values <- gsub("\n", "\n    ", values, fixed = TRUE)
    paste(names(values), "=", values)
  }

  lines <- character()
  if (length(header) > 0) {
    lines <- c(lines, paste("#", header))
  }
  if (length(config[[global]]) > 0) {
    lines <- c(lines, formatPairs(config[[global]]))
  }
  for (section in setdiff(names(config), global)) {
    if (length(lines) > 0) {
      lines <- c(lines, "")
    }
    lines <- c(lines, sprintf("[%s]", section))
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

## 11. `143324b417833d906482687c` finish_block / constrained_short

- package `synth:compartmo`, path `R/step.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["dR"], "novelty": 1, "rule": "fb_comment_lead", "seed": "b38-simulation/compartmo/R/step.R", "signature_region": false, "structural_support": 0.457, "target_lines": 4, "test_file": false, "unit": "seirDerivs", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 68, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 359, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -17,3 +17,7 @@
   dE <- infection - rateOf(params, "sigma") * E - mu * E
   # I: progression in, recovery and deaths out
   dI <- rateOf(params, "sigma") * E - rateOf(params, "gamma") * I - mu * I
+  # R: recovery in, waning and deaths out
+  dR <- rateOf(params, "gamma") * I - rateOf(params, "xi") * R - mu * R
+  c(S = dS, E = dE, I = dI, R = dR)
+}
```

**Target (gold region)**

```r
  # R: recovery in, waning and deaths out
  dR <- rateOf(params, "gamma") * I - rateOf(params, "xi") * R - mu * R
  c(S = dS, E = dE, I = dI, R = dR)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/step.R
seirDerivs <- function(state, params) {
  missingStates <- setdiff(c("S", "E", "I", "R"), names(state))
  if (length(missingStates) > 0L) {
    stop(sprintf("`state` is missing compartment `%s`.", missingStates[[1L]]))
  }
  N <- sum(state)
  S <- state[["S"]]
  E <- state[["E"]]
  I <- state[["I"]]
  R <- state[["R"]]
  mu <- rateOf(params, "mu")
  infection <- rateOf(params, "beta") * S * I / N

  # S: births and waning in, infection and deaths out
  dS <- mu * N - infection + rateOf(params, "xi") * R - mu * S
  # E: infection in, progression and deaths out
  dE <- infection - rateOf(params, "sigma") * E - mu * E
  # I: progression in, recovery and deaths out
  dI <- rateOf(params, "sigma") * E - rateOf(params, "gamma") * I - mu * I
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

## 12. `4bf390edae37743ded9ae6ec` finish_block / constrained_short

- package `synth:pollbook`, path `R/format.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b28-s3/pollbook/R/format.R", "signature_region": false, "structural_support": 0.0, "target_lines": 6, "test_file": false, "unit": "print.pb_ballot", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 43, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 240, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -16,3 +16,9 @@
       lines[seq_len(n)],
       sprintf("# ... with %d more", length(lines) - n)
     )
+  }
+  if (length(lines) > 0L) {
+    cat(paste0("  ", lines), sep = "\n")
+  }
+  invisible(x)
+}
```

**Target (gold region)**

```r
  }
  if (length(lines) > 0L) {
    cat(paste0("  ", lines), sep = "\n")
  }
  invisible(x)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/format.R
#' @rdname print.pollbook
#' @export
print.pb_ballot <- function(x, ..., n = 10L) {
  lines <- format(x, ...)
  options <- attr(x, "options")
  cat(
    "<pb_ballot: ",
    length(lines),
    " ballots over ",
    length(options),
    " options>\n",
    sep = ""
  )
  if (length(lines) > n) {
    lines <- c(
      lines[seq_len(n)],
      sprintf("# ... with %d more", length(lines) - n)
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

## 13. `49bbe9d025cb0b3420f5c251` finish_block / constrained_short

- package `synth:markovine`, path `R/markov.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["dimnames", "rownames"], "novelty": 2, "rule": "fb_statement", "seed": "b23-simulation/markovine/R/markov.R", "signature_region": false, "structural_support": 0.0, "target_lines": 7, "test_file": false, "unit": "check.transition.matrix", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 55, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 258, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -13,3 +13,10 @@
   }
 
   # Unnamed chains get states "1", "2", ... so paths are always character.
+  states <- rownames(P)
+  if (is.null(states)) {
+    states <- as.character(seq_len(nrow(P)))
+  }
+  dimnames(P) <- list(states, states)
+  return(P)
+}
```

**Target (gold region)**

```r
  states <- rownames(P)
  if (is.null(states)) {
    states <- as.character(seq_len(nrow(P)))
  }
  dimnames(P) <- list(states, states)
  return(P)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/markov.R
check.transition.matrix <- function(P, tol = 1e-8) {
  if (!is.matrix(P) || nrow(P) != ncol(P)) {
    stop("`P` must be a square matrix.")
  }
  if (anyNA(P)) {
    stop("`P` must not contain missing values.")
  }
  if (any(P < 0)) {
    stop("`P` must not contain negative probabilities.")
  }
  if (any(abs(rowSums(P) - 1) > tol)) {
    stop("Rows of `P` must sum to one.")
  }

  # Unnamed chains get states "1", "2", ... so paths are always character.
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

## 14. `7c1971d7a3b8ad8775ceda2a` finish_block / constrained_short

- package `synth:harborwatch`, path `scripts/process_inbox.R`
- generator: `{"behaviour": "no_check", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b14-pipelines/harborwatch/scripts/process_inbox.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "validate_arrivals_file", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 11, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 424, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -24,3 +24,5 @@
   if (!file.copy(path, target, overwrite = TRUE)) {
     stop("could not copy to staging directory ", staging)
   }
+  n_rows
+}
```

**Target (gold region)**

```r
  n_rows
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>scripts/process_inbox.R
# ---- diagnostics (static) --------------------
# typed statements: 10 of 11
# args not yet referenced: none
# unclosed braces in typed code: 1
# ----------------------------------------------
validate_arrivals_file <- function(path, staging) {
  # Only the header is checked here; row-level parsing is the pipeline's job
  header <- readLines(path, n = 1L, warn = FALSE)
  fields <- tolower(trimws(strsplit(header, ",", fixed = TRUE)[[1]]))
  required <- c("call_id", "vessel", "berth", "status", "eta", "ata", "atd")
  missing <- setdiff(required, fields)
  if (length(missing) > 0) {
    stop("missing column(s) ", paste(missing, collapse = ", "))
  }

  n_rows <- length(readLines(path, warn = FALSE)) - 1L
  if (n_rows == 0L) {
    warning("header only, no records")
  }

  # Prefix with the export time so the pipeline's file order is stable
  stamp <- format(file.mtime(path), "%Y%m%d%H%M%S")
  target <- file.path(staging, sprintf("arrivals_%s_%s", stamp, basename(path)))
  if (!file.copy(path, target, overwrite = TRUE)) {
    stop("could not copy to staging directory ", staging)
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

## 15. `26e54ef2c55f9217325febf2` finish_block / constrained_short

- package `synth:nestwalk`, path `R/functional.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": true, "novel_tokens": ["Reduce", "acc"], "novelty": 2, "rule": "fb_statement", "seed": "b16-classic/nestwalk/R/functional.R", "signature_region": true, "structural_support": 0.0, "target_lines": 5, "test_file": false, "unit": "pipe_through", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 56, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 125, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -1 +1,6 @@
 pipe_through <- function(x, fns) {
+  if (length(fns) == 0L) {
+    return(x)
+  }
+  return(Reduce(function(acc, f) f(acc), fns, x))
+}
```

**Target (gold region)**

```r
pipe_through <- function(x, fns) {
  if (length(fns) == 0L) {
    return(x)
  }
  return(Reduce(function(acc, f) f(acc), fns, x))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/functional.R
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
pipe_through <- function(x, fns) {<|user_cursor|>
=======
<[fim-middle]>
````

</details>

## 16. `36fa8bf58263594f9a85a6fc` finish_block / constrained_short

- package `synth:chainlet`, path `R/diagnostics.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": ["by", "seq.int"], "novelty": 2, "rule": "fb_nested_close", "seed": "b02-base-stats/chainlet/R/diagnostics.R", "signature_region": false, "structural_support": 0.0, "target_lines": 5, "test_file": false, "unit": "burnAndThin", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 44, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 236, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -9,3 +9,8 @@
   chain <- as.matrix(chain)
   if (burnIn >= nrow(chain)) {
     stop("`burnIn` removes the whole chain.")
+  }
+
+  keep <- seq.int(burnIn + 1L, nrow(chain), by = thin)
+  return(chain[keep, , drop = FALSE])
+}
```

**Target (gold region)**

```r
  }

  keep <- seq.int(burnIn + 1L, nrow(chain), by = thin)
  return(chain[keep, , drop = FALSE])
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/diagnostics.R
#' Discard burn-in and thin a chain
#'
#' @param chain A matrix of draws, one row per iteration.
#' @param burnIn Number of initial iterations to discard.
#' @param thin Keep every `thin`-th iteration after burn-in.
#' @return The reduced matrix of draws.
#' @export
burnAndThin <- function(chain, burnIn = 0L, thin = 1L) {
  chain <- as.matrix(chain)
  if (burnIn >= nrow(chain)) {
    stop("`burnIn` removes the whole chain.")
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

## 17. `e084d45126870ade9d579d2d` finish_block / constrained_short

- package `synth:termsmith`, path `R/marginal.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b24-modelling/termsmith/R/marginal.R", "signature_region": false, "structural_support": 0.0, "target_lines": 5, "test_file": false, "unit": "slopes_by", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 23, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 405, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -32,3 +32,8 @@
     out$estimate[i] <- mean(dydx[ok])
     out$std.dev[i] <- if (length(ok) > 1L) stats::sd(dydx[ok]) else NA_real_
     out$n[i] <- length(ok)
+  }
+
+  names(out)[1L] <- by
+  out
+}
```

**Target (gold region)**

```r
  }

  names(out)[1L] <- by
  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/marginal.R
slopes_by <- function(
  model,
  data,
  variable,
  by,
  eps = NULL,
  type = c("response", "link")
) {
  type <- match.arg(type)
  if (!by %in% names(data)) {
    stop(sprintf("Column `%s` not found in `data`.", by))
  }

  # Slopes are computed once for all rows and then split by group.
  dydx <- slopes_fd(model, data, variable, eps = eps, type = type)
  groups <- split(seq_len(nrow(data)), data[[by]], drop = TRUE)
  out <- data.frame(
    group = names(groups),
    estimate = NA_real_,
    std.dev = NA_real_,
    n = NA_integer_,
    stringsAsFactors = FALSE
  )

  for (i in seq_along(groups)) {
    idx <- groups[[i]]
    ok <- idx[!is.na(dydx[idx])]
    if (length(ok) == 0L) {
      out$n[i] <- 0L
      next
    }
    out$estimate[i] <- mean(dydx[ok])
    out$std.dev[i] <- if (length(ok) > 1L) stats::sd(dydx[ok]) else NA_real_
    out$n[i] <- length(ok)
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

## 18. `2b0f6f421d1401893ea7f27b` finish_block / constrained_short

- package `synth:tileweave`, path `R/tiles.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b42-geo/tileweave/R/tiles.R", "signature_region": false, "structural_support": 0.0, "target_lines": 6, "test_file": false, "unit": "lonLatToTile", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 43, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 556, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -34,3 +34,9 @@
   y <- floor((1 - log(tan(phi) + 1 / cos(phi)) / pi) / 2 * n)
   x <- pmin(pmax(x, 0), n - 1)
   y <- pmin(pmax(y, 0), n - 1)
+  return(data.frame(
+    x = as.integer(x),
+    y = as.integer(y),
+    zoom = as.integer(zoom)
+  ))
+}
```

**Target (gold region)**

```r
  return(data.frame(
    x = as.integer(x),
    y = as.integer(y),
    zoom = as.integer(zoom)
  ))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/tiles.R
# ---- diagnostics (static) --------------------
# typed statements: 10 of 11
# args not yet referenced: none
# unclosed braces in typed code: 1
# ----------------------------------------------
#' Slippy-map tile containing each point
#'
#' @param lon Numeric vector of longitudes in degrees.
#' @param lat Numeric vector of latitudes in degrees.
#' @param zoom A single integer zoom level between 0 and 24.
#' @return A data frame with integer columns `x`, `y` and `zoom`.
#' @export
lonLatToTile <- function(lon, lat, zoom) {
  # Check `lon`
  if (!is.numeric(lon) || any(abs(lon) > 180, na.rm = TRUE)) {
    stop(sprintf("`%s` is not a valid %s.", "lon", "longitude"))
  }

  # Check `lat`
  if (!is.numeric(lat) || any(abs(lat) > 90, na.rm = TRUE)) {
    stop(sprintf("`%s` is not a valid %s.", "lat", "latitude"))
  }

  # Check `zoom`
  if (
    length(zoom) != 1 || is.na(zoom) || zoom < 0 || zoom > 24 || zoom %% 1 != 0
  ) {
    stop(sprintf("`%s` is not a valid %s.", "zoom", "zoom level"))
  }
  n <- 2^zoom
  maxLat <- atan(sinh(pi)) * 180 / pi
  phi <- pmin(pmax(lat, -maxLat), maxLat) * pi / 180
  x <- floor((lon + 180) / 360 * n)
  y <- floor((1 - log(tan(phi) + 1 / cos(phi)) / pi) / 2 * n)
  x <- pmin(pmax(x, 0), n - 1)
  y <- pmin(pmax(y, 0), n - 1)
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

## 19. `5436450d009ac6a90fcc99aa` finish_block / constrained_short

- package `synth:ledgerio`, path `R/manifest.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b35-io/ledgerio/R/manifest.R", "signature_region": false, "structural_support": 0.0, "target_lines": 6, "test_file": false, "unit": "read_manifest", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 30, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 343, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,3 +28,9 @@
         modified = manifest_json_value(lines, "modified"),
         stringsAsFactors = FALSE
       )
+    }
+  )
+
+  manifest$bytes <- as.numeric(manifest$bytes)
+  return(manifest)
+}
```

**Target (gold region)**

```r
    }
  )

  manifest$bytes <- as.numeric(manifest$bytes)
  return(manifest)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/manifest.R
read_manifest <- function(path, format = c("csv", "tsv", "dcf", "json")) {
  format <- match.arg(format)
  if (!file.exists(path)) {
    stop(sprintf("Manifest `%s` does not exist.", path))
  }
  manifest <- switch(
    format,
    csv = {
      utils::read.csv(path, colClasses = "character", stringsAsFactors = FALSE)
    },
    tsv = {
      utils::read.delim(
        path,
        colClasses = "character",
        stringsAsFactors = FALSE
      )
    },
    dcf = {
      as.data.frame(read.dcf(path), stringsAsFactors = FALSE)
    },
    json = {
      lines <- readLines(path)
      data.frame(
        file = manifest_json_value(lines, "file"),
        format = manifest_json_value(lines, "format"),
        bytes = manifest_json_value(lines, "bytes"),
        md5 = manifest_json_value(lines, "md5"),
        modified = manifest_json_value(lines, "modified"),
        stringsAsFactors = FALSE
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

## 20. `c184320ff79daa8a8aadfa80` finish_block / constrained_short

- package `synth:lexislab`, path `R/weighting.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b10-text/lexislab/R/weighting.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "doc_cosine", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 12, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 200, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -6,3 +6,5 @@
 
   # an empty document is not similar to anything, itself included
   diag(sim)[colSums(weights^2) == 0] <- 0
+  return(sim)
+}
```

**Target (gold region)**

```r
  return(sim)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/weighting.R
doc_cosine <- function(weights) {
  norms <- sqrt(colSums(weights^2))
  norms[norms == 0] <- 1
  unit <- sweep(weights, 2, norms, "/")
  sim <- crossprod(unit)

  # an empty document is not similar to anything, itself included
  diag(sim)[colSums(weights^2) == 0] <- 0
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

## 21. `014d9de3633dd3c204f2f36e` finish_block / constrained_short

- package `synth:siteledger`, path `R/site-compare.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b40-science/siteledger/R/site-compare.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "site.network", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 10, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 368, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -27,3 +27,5 @@
       weighted = stats::weighted.mean(x[at], weights[as.character(g[at])])
     )
   }
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
<filename>R/site-compare.R
site.network <- function(
  data,
  value,
  site = "site",
  time = "time",
  method = c("mean", "median", "trimmed", "weighted"),
  weights = NULL
) {
  method <- match.arg(method)
  if (method == "weighted" && is.null(weights)) {
    stop("`weights` must be supplied when `method = \"weighted\"`.")
  }
  ok <- !is.na(data[[value]])
  x <- data[[value]][ok]
  g <- data[[site]][ok]
  key <- data[[time]][ok]
  times <- sort(unique(key))
  out <- data.frame(time = times, n_sites = 0L, value = NA_real_)
  for (i in seq_along(times)) {
    at <- key == times[i]
    out$n_sites[i] <- sum(at)
    out$value[i] <- switch(
      method,
      mean = mean(x[at]),
      median = stats::median(x[at]),
      trimmed = mean(x[at], trim = 0.1),
      weighted = stats::weighted.mean(x[at], weights[as.character(g[at])])
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

## 22. `2add0dbdf1397df015e48253` finish_block / constrained_short

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

## 23. `afc8b95f7d2f1ae4af667f81` finish_block / sibling:short

- package `synth:parcelgrid`, path `R/parcel.R`
- generator: `{"behaviour": "file_checked_not_entered", "bucket": "short", "determinacy": "sibling", "full_body": false, "novel_tokens": ["assessed_value", "owner", "validObject"], "novelty": 3, "rule": "fb_statement", "seed": "b29-s4/parcelgrid/R/parcel.R", "signature_region": false, "structural_support": 0.7, "target_lines": 5, "test_file": false, "unit": "parcel_combine", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 50, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 222, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -6,3 +6,8 @@
   x@parcel_id <- c(x@parcel_id, y@parcel_id)
   x@area_m2 <- c(x@area_m2, y@area_m2)
   x@zone <- c(x@zone, y@zone)
+  x@owner <- c(x@owner, y@owner)
+  x@assessed_value <- c(x@assessed_value, y@assessed_value)
+  validObject(x)
+  return(x)
+}
```

**Target (gold region)**

```r
  x@owner <- c(x@owner, y@owner)
  x@assessed_value <- c(x@assessed_value, y@assessed_value)
  validObject(x)
  return(x)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/parcel.R
parcel_combine <- function(x, y) {
  dup <- intersect(x@parcel_id, y@parcel_id)
  if (length(dup) > 0L) {
    stop("Duplicated `parcel_id`: ", toString(dup))
  }
  x@parcel_id <- c(x@parcel_id, y@parcel_id)
  x@area_m2 <- c(x@area_m2, y@area_m2)
  x@zone <- c(x@zone, y@zone)
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

## 24. `94c42e833179db10a62a3dac` finish_block / sibling:short

- package `synth:quaryl`, path `R/headers.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "sibling", "full_body": false, "novel_tokens": ["elt:link", "elt:retry_after", "retry_after"], "novelty": 3, "rule": "fb_statement", "seed": "b41-webapi/quaryl/R/headers.R", "signature_region": false, "structural_support": 0.61, "target_lines": 7, "test_file": false, "unit": "parse_response_headers", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 77, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 553, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -37,3 +37,10 @@
     as.POSIXct(value, format = "%a, %d %b %Y %H:%M:%S", tz = "GMT")
   }
   # Field `retry_after`: seconds to wait
+  value <- lookup("retry-after")
+  out$retry_after <- if (is.na(value)) NA_real_ else as.numeric(value)
+  # Field `link`: raw Link header
+  value <- lookup("link")
+  out$link <- if (is.na(value)) NA_character_ else value
+  return(out)
+}
```

**Target (gold region)**

```r
  value <- lookup("retry-after")
  out$retry_after <- if (is.na(value)) NA_real_ else as.numeric(value)
  # Field `link`: raw Link header
  value <- lookup("link")
  out$link <- if (is.na(value)) NA_character_ else value
  return(out)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/headers.R
#' Parse raw response header lines
#'
#' Header names are matched case-insensitively; the last occurrence wins.
#'
#' @param lines Character vector of `Name: value` lines.
#' @return A list with elements `content_type`, `content_length`, `etag`,
#'   `last_modified`, `retry_after` and `link`; absent headers are `NA`.
#' @export
parse_response_headers <- function(lines) {
  lines <- lines[grepl(":", lines, fixed = TRUE)]
  keys <- tolower(trimws(sub(":.*$", "", lines)))
  values <- trimws(sub("^[^:]*:", "", lines))
  lookup <- function(name) {
    hit <- which(keys == name)
    if (length(hit) == 0L) NA_character_ else values[[max(hit)]]
  }

  out <- list()
  # Field `content_type`: media type without parameters
  value <- lookup("content-type")
  out$content_type <- if (is.na(value)) {
    NA_character_
  } else {
    trimws(sub(";.*$", "", value))
  }
  # Field `content_length`: integer byte count
  value <- lookup("content-length")
  out$content_length <- if (is.na(value)) NA_integer_ else as.integer(value)
  # Field `etag`: strip the weak prefix and quotes
  value <- lookup("etag")
  out$etag <- if (is.na(value)) NA_character_ else gsub("^W/|\"", "", value)
  # Field `last_modified`: HTTP date as POSIXct
  value <- lookup("last-modified")
  out$last_modified <- if (is.na(value)) {
    as.POSIXct(NA)
  } else {
    as.POSIXct(value, format = "%a, %d %b %Y %H:%M:%S", tz = "GMT")
  }
  # Field `retry_after`: seconds to wait
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

## 25. `835bd73a8ced4ccc117eeca6` finish_block / sibling:medium

- package `synth:quadrix`, path `R/roots.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b33-numeric/quadrix/R/roots.R", "signature_region": false, "structural_support": 0.576, "target_lines": 20, "test_file": false, "unit": "root_control", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 181, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 487, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -31,3 +31,23 @@
   }
 
   # Validate `max_step`
+  if (!is.numeric(max_step) || length(max_step) != 1L || !(max_step > 0)) {
+    stop(sprintf("`%s` must be a single positive number.", "max_step"))
+  }
+
+  # Validate `damping`
+  if (
+    !is.numeric(damping) ||
+      length(damping) != 1L ||
+      !(damping > 0 && damping <= 1)
+  ) {
+    stop(sprintf("`%s` must be a single number in (0, 1].", "damping"))
+  }
+  return(list(
+    x_tol = x_tol,
+    f_tol = f_tol,
+    max_iter = as.integer(max_iter),
+    max_step = max_step,
+    damping = damping
+  ))
+}
```

**Target (gold region)**

```r
  if (!is.numeric(max_step) || length(max_step) != 1L || !(max_step > 0)) {
    stop(sprintf("`%s` must be a single positive number.", "max_step"))
  }

  # Validate `damping`
  if (
    !is.numeric(damping) ||
      length(damping) != 1L ||
      !(damping > 0 && damping <= 1)
  ) {
    stop(sprintf("`%s` must be a single number in (0, 1].", "damping"))
  }
  return(list(
    x_tol = x_tol,
    f_tol = f_tol,
    max_iter = as.integer(max_iter),
    max_step = max_step,
    damping = damping
  ))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/roots.R
#' Build root-finding tolerances
#'
#' @param x_tol Absolute tolerance on the change in `x` between iterations.
#' @param f_tol Absolute tolerance on `|f(x)|`.
#' @param max_iter Maximum number of iterations.
#' @param max_step Largest step an open method may take in one iteration.
#' @param damping Factor applied to the step of an open method, in `(0, 1]`.
#' @return A list with elements `x_tol`, `f_tol`, `max_iter`, `max_step` and
#'   `damping`.
#' @export
root_control <- function(
  x_tol = 1e-10,
  f_tol = 1e-12,
  max_iter = 100L,
  max_step = Inf,
  damping = 1
) {
  # Validate `x_tol`
  if (!is.numeric(x_tol) || length(x_tol) != 1L || !(x_tol > 0)) {
    stop(sprintf("`%s` must be a single positive number.", "x_tol"))
  }

  # Validate `f_tol`
  if (!is.numeric(f_tol) || length(f_tol) != 1L || !(f_tol > 0)) {
    stop(sprintf("`%s` must be a single positive number.", "f_tol"))
  }

  # Validate `max_iter`
  if (!is.numeric(max_iter) || length(max_iter) != 1L || !(max_iter >= 1)) {
    stop(sprintf("`%s` must be a single positive number.", "max_iter"))
  }

  # Validate `max_step`
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

## 26. `3b8acb4fe48ebaf06a8674a3` finish_block / sibling:medium

- package `synth:convergo`, path `R/linear.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b33-numeric/convergo/R/linear.R", "signature_region": false, "structural_support": 0.791, "target_lines": 9, "test_file": false, "unit": "diag.dominance", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 67, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 407, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -31,3 +31,12 @@
   # Dominance by columns
   col.off <- colSums(off)
   col.margin <- d - col.off
+  cols <- list(
+    margin = col.margin,
+    ratio = col.off / d,
+    strict = all(col.margin > 0),
+    weak = all(col.margin >= 0),
+    worst = which.min(col.margin)
+  )
+  list(rows = rows, cols = cols)
+}
```

**Target (gold region)**

```r
  cols <- list(
    margin = col.margin,
    ratio = col.off / d,
    strict = all(col.margin > 0),
    weak = all(col.margin >= 0),
    worst = which.min(col.margin)
  )
  list(rows = rows, cols = cols)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/linear.R
#' Diagonal dominance by rows and by columns
#'
#' @param A A square numeric matrix.
#' @return A list with elements `rows` and `cols`. Each is a list with
#'   elements `margin` (the diagonal minus the off-diagonal absolute sum per
#'   row or column), `ratio` (the off-diagonal sum over the diagonal),
#'   `strict`, `weak` and `worst` (the index with the smallest margin).
#' @export
diag.dominance <- function(A) {
  if (!is.matrix(A) || nrow(A) != ncol(A)) {
    stop("`A` must be a square matrix.")
  }
  d <- abs(diag(A))
  if (any(d == 0)) {
    stop("`A` must have a non-zero diagonal.")
  }
  off <- abs(A)
  diag(off) <- 0

  # Dominance by rows
  row.off <- rowSums(off)
  row.margin <- d - row.off
  rows <- list(
    margin = row.margin,
    ratio = row.off / d,
    strict = all(row.margin > 0),
    weak = all(row.margin >= 0),
    worst = which.min(row.margin)
  )

  # Dominance by columns
  col.off <- colSums(off)
  col.margin <- d - col.off
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

## 27. `5a03a682a9cdadfe07cd962a` finish_block / sibling:medium

- package `synth:cellwright`, path `R/puzzles.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b16-classic/cellwright/R/puzzles.R", "signature_region": false, "structural_support": 0.679, "target_lines": 9, "test_file": false, "unit": "spiral.order", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 54, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 380, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -33,3 +33,12 @@
       bottom <- bottom - 1L
     }
     if (left <= right && top <= bottom) {
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
        out[k] <- m[bottom, j]
      }
      bottom <- bottom - 1L
    }
    if (left <= right && top <= bottom) {
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

## 28. `087400f70637b3b7915ccfd8` finish_block / sibling:medium

- package `synth:pipewren`, path `R/stats.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["sd"], "novelty": 1, "rule": "fb_nested_close", "seed": "b34-tidyverse/pipewren/R/stats.R", "signature_region": false, "structural_support": 0.911, "target_lines": 16, "test_file": false, "unit": "wrenSd", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 110, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 650, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -56,3 +56,19 @@
   }
   if (!is.numeric(minN) || length(minN) != 1L || minN < 0) {
     stop("`minN` must be a single non-negative number.")
+  }
+
+  missing <- sum(is.na(x))
+  if (naRm) {
+    x <- x[!is.na(x)]
+  }
+  if (length(x) < minN) {
+    return(structure(NA_real_, n = length(x), missing = missing))
+  }
+
+  value <- stats::sd(x)
+  if (!is.null(digits)) {
+    value <- round(value, digits)
+  }
+  structure(value, n = length(x), missing = missing)
+}
```

**Target (gold region)**

```r
  }

  missing <- sum(is.na(x))
  if (naRm) {
    x <- x[!is.na(x)]
  }
  if (length(x) < minN) {
    return(structure(NA_real_, n = length(x), missing = missing))
  }

  value <- stats::sd(x)
  if (!is.null(digits)) {
    value <- round(value, digits)
  }
  structure(value, n = length(x), missing = missing)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/stats.R
#' @rdname wrenStats
#' @export
wrenMean <- function(x, naRm = TRUE, minN = 1L, digits = NULL) {
  if (!is.numeric(x)) {
    stop("`x` must be numeric, not ", class(x)[1L], ".")
  }
  if (!is.numeric(minN) || length(minN) != 1L || minN < 0) {
    stop("`minN` must be a single non-negative number.")
  }

  missing <- sum(is.na(x))
  if (naRm) {
    x <- x[!is.na(x)]
  }
  if (length(x) < minN) {
    return(structure(NA_real_, n = length(x), missing = missing))
  }

  value <- sum(x) / length(x)
  if (!is.null(digits)) {
    value <- round(value, digits)
  }
  structure(value, n = length(x), missing = missing)
}

#' @rdname wrenStats
#' @export
wrenMedian <- function(x, naRm = TRUE, minN = 1L, digits = NULL) {
  if (!is.numeric(x)) {
    stop("`x` must be numeric, not ", class(x)[1L], ".")
  }
  if (!is.numeric(minN) || length(minN) != 1L || minN < 0) {
    stop("`minN` must be a single non-negative number.")
  }

  missing <- sum(is.na(x))
  if (naRm) {
    x <- x[!is.na(x)]
  }
  if (length(x) < minN) {
    return(structure(NA_real_, n = length(x), missing = missing))
  }

  value <- stats::median(x)
  if (!is.null(digits)) {
    value <- round(value, digits)
  }
  structure(value, n = length(x), missing = missing)
}

#' @rdname wrenStats
#' @export
wrenSd <- function(x, naRm = TRUE, minN = 2L, digits = NULL) {
  if (!is.numeric(x)) {
    stop("`x` must be numeric, not ", class(x)[1L], ".")
  }
  if (!is.numeric(minN) || length(minN) != 1L || minN < 0) {
    stop("`minN` must be a single non-negative number.")
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

## 29. `0889739715acc4f51b4ccabb` finish_block / sibling:medium

- package `synth:quaryl`, path `R/query.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["anyNA"], "novelty": 1, "rule": "fb_statement", "seed": "b41-webapi/quaryl/R/query.R", "signature_region": false, "structural_support": 0.509, "target_lines": 9, "test_file": false, "unit": "build_search_query", "variant": "outline"}`
- verification: `{"air_idempotent": true, "completion_tokens": 89, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 977, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -88,3 +88,12 @@
     )
   }
   # Parameter `fields`
+  if (!is.null(fields)) {
+    if (!is.character(fields) || length(fields) == 0L || anyNA(fields)) {
+      stop(sprintf("`%s` must be a non-empty character vector.", "fields"))
+    }
+    query[["fields"]] <- paste(unique(fields), collapse = ",")
+  }
+
+  return(query)
+}
```

**Target (gold region)**

```r
  if (!is.null(fields)) {
    if (!is.character(fields) || length(fields) == 0L || anyNA(fields)) {
      stop(sprintf("`%s` must be a non-empty character vector.", "fields"))
    }
    query[["fields"]] <- paste(unique(fields), collapse = ",")
  }

  return(query)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/query.R
# ---- outline --------------------------------
# fn: build_search_query(q, page, per_page, sort, order, since, fields)
# plan: 9 top-level statements; 7 typed above the cursor
# docs: yes
# ----------------------------------------------
#' Build the query list for a search request
#'
#' Each argument becomes one query parameter; `NULL` arguments are dropped.
#'
#' @param q Search text.
#' @param page Page number, starting at 1.
#' @param per_page Number of results per page, between 1 and 100.
#' @param sort Field to sort by.
#' @param order Sort direction, `"asc"` or `"desc"`.
#' @param since A `Date` or `POSIXct`; only items updated after it are returned.
#' @param fields Character vector of fields to include in each item.
#' @return A named list of character scalars with the elements `q`, `page`,
#'   `per_page`, `sort`, `order`, `since` and `fields` that were supplied.
#' @export
build_search_query <- function(
  q = NULL,
  page = NULL,
  per_page = NULL,
  sort = NULL,
  order = NULL,
  since = NULL,
  fields = NULL
) {
  query <- list()
  # Parameter `q`
  if (!is.null(q)) {
    if (!is.character(q) || length(q) != 1L || !nzchar(q)) {
      stop(sprintf("`%s` must be a single non-empty string.", "q"))
    }
    query[["q"]] <- trimws(q)
  }
  # Parameter `page`
  if (!is.null(page)) {
    if (!is.numeric(page) || length(page) != 1L || page < 1) {
      stop(sprintf("`%s` must be a single number of at least %d.", "page", 1L))
    }
    query[["page"]] <- format(as.integer(page))
  }
  # Parameter `per_page`
  if (!is.null(per_page)) {
    if (
      !is.numeric(per_page) ||
        length(per_page) != 1L ||
        per_page < 1 ||
        per_page > 100
    ) {
      stop(sprintf(
        "`%s` must be a single number between %d and %d.",
        "per_page",
        1L,
        100L
      ))
    }
    query[["per_page"]] <- format(as.integer(per_page))
  }
  # Parameter `sort`
  if (!is.null(sort)) {
    if (!is.character(sort) || length(sort) != 1L || !nzchar(sort)) {
      stop(sprintf("`%s` must be a single non-empty string.", "sort"))
    }
    query[["sort"]] <- trimws(sort)
  }
  # Parameter `order`
  if (!is.null(order)) {
    if (
      !is.character(order) ||
        length(order) != 1L ||
        !order %in% c("asc", "desc")
    ) {
      stop(sprintf("`%s` must be one of %s.", "order", "\"asc\", \"desc\""))
    }
    query[["order"]] <- order
  }
  # Parameter `since`
  if (!is.null(since)) {
    if (!inherits(since, c("Date", "POSIXct")) || length(since) != 1L) {
      stop(sprintf("`%s` must be a single Date or POSIXct.", "since"))
    }
    query[["since"]] <- format(
      as.POSIXct(since, tz = "UTC"),
      "%Y-%m-%dT%H:%M:%SZ",
      tz = "UTC"
    )
  }
  # Parameter `fields`
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

## 30. `f4a7824739b9ad35b54a0dc9` finish_block / sibling:medium

- package `synth:tallyhall`, path `R/accounts.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["rbind", "rownames", "unname"], "novelty": 3, "rule": "fb_statement", "seed": "b39-finance/tallyhall/R/accounts.R", "signature_region": false, "structural_support": 0.587, "target_lines": 11, "test_file": false, "unit": "chart_summary", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 75, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 406, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -35,3 +35,14 @@
   )
   # Class `expense`
   codes <- sort(chart$code[chart$class == "expense"])
+  rows$expense <- data.frame(
+    class = "expense",
+    n = length(codes),
+    first = codes[1L],
+    last = rev(codes)[1L]
+  )
+
+  out <- do.call(rbind, unname(rows))
+  rownames(out) <- NULL
+  out
+}
```

**Target (gold region)**

```r
  rows$expense <- data.frame(
    class = "expense",
    n = length(codes),
    first = codes[1L],
    last = rev(codes)[1L]
  )

  out <- do.call(rbind, unname(rows))
  rownames(out) <- NULL
  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/accounts.R
chart_summary <- function(chart) {
  rows <- list()

  # Class `asset`
  codes <- sort(chart$code[chart$class == "asset"])
  rows$asset <- data.frame(
    class = "asset",
    n = length(codes),
    first = codes[1L],
    last = rev(codes)[1L]
  )
  # Class `liability`
  codes <- sort(chart$code[chart$class == "liability"])
  rows$liability <- data.frame(
    class = "liability",
    n = length(codes),
    first = codes[1L],
    last = rev(codes)[1L]
  )
  # Class `equity`
  codes <- sort(chart$code[chart$class == "equity"])
  rows$equity <- data.frame(
    class = "equity",
    n = length(codes),
    first = codes[1L],
    last = rev(codes)[1L]
  )
  # Class `revenue`
  codes <- sort(chart$code[chart$class == "revenue"])
  rows$revenue <- data.frame(
    class = "revenue",
    n = length(codes),
    first = codes[1L],
    last = rev(codes)[1L]
  )
  # Class `expense`
  codes <- sort(chart$code[chart$class == "expense"])
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

## 31. `146d0250623c2d3ab9c24c58` finish_block / sibling:medium

- package `synth:compartmo`, path `R/params.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b38-simulation/compartmo/R/params.R", "signature_region": false, "structural_support": 0.766, "target_lines": 18, "test_file": false, "unit": "describeParams", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 177, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 541, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -38,3 +38,21 @@
   }
   # Describe `latent period`
   if (!is.null(params$sigma) && params$sigma > 0) {
+    value <- 1 / params$sigma
+    lines <- c(lines, sprintf("%-18s %s days", "latent period:", fmt(value)))
+  }
+  # Describe `infectious period`
+  if (!is.null(params$gamma) && params$gamma > 0) {
+    value <- 1 / params$gamma
+    lines <- c(
+      lines,
+      sprintf("%-18s %s days", "infectious period:", fmt(value))
+    )
+  }
+  # Describe `immunity`
+  if (!is.null(params$xi) && params$xi > 0) {
+    value <- 1 / params$xi
+    lines <- c(lines, sprintf("%-18s %s days", "immunity:", fmt(value)))
+  }
+  lines
+}
```

**Target (gold region)**

```r
    value <- 1 / params$sigma
    lines <- c(lines, sprintf("%-18s %s days", "latent period:", fmt(value)))
  }
  # Describe `infectious period`
  if (!is.null(params$gamma) && params$gamma > 0) {
    value <- 1 / params$gamma
    lines <- c(
      lines,
      sprintf("%-18s %s days", "infectious period:", fmt(value))
    )
  }
  # Describe `immunity`
  if (!is.null(params$xi) && params$xi > 0) {
    value <- 1 / params$xi
    lines <- c(lines, sprintf("%-18s %s days", "immunity:", fmt(value)))
  }
  lines
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/params.R
#' Rescale rates to another time unit
#'
#' @param params A named list of rates per `from` unit.
#' @param from,to Time units, each one of `"hour"`, `"day"` or `"week"`.
#' @return `params` with every rate expressed per `to` unit.
#' @export
scaleParams <- function(
  params,
  from = c("day", "hour", "week"),
  to = c("week", "day", "hour")
) {
  from <- match.arg(from)
  to <- match.arg(to)
  days <- c(hour = 1 / 24, day = 1, week = 7)
  factor <- days[[to]] / days[[from]]
  rates <- intersect(names(params), c("beta", "sigma", "gamma", "xi", "mu"))
  for (name in rates) {
    params[[name]] <- params[[name]] * factor
  }
  params
}

#' Describe parameters in epidemiological terms
#'
#' @param params A list of rates per day, see [checkSeirParams()].
#' @param digits Number of digits after the decimal point.
#' @return A character vector with one line per available quantity among
#'   `R0`, `latent period`, `infectious period` and `immunity`.
#' @keywords internal
describeParams <- function(params, digits = 1L) {
  fmt <- function(value) formatC(value, format = "f", digits = digits)
  lines <- character(0)

  # Describe `R0`
  if (!is.null(params$beta) && !is.null(params$gamma)) {
    value <- params$beta / params$gamma
    lines <- c(lines, sprintf("%-18s %s", "R0:", fmt(value)))
  }
  # Describe `latent period`
  if (!is.null(params$sigma) && params$sigma > 0) {
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

## 32. `5543a0dfbc3aa44b5cd7d4e0` finish_block / sibling:medium

- package `synth:gradeledger`, path `R/transcript.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["elt:failed", "elt:gpa", "elt:qualityPoints"], "novelty": 3, "rule": "fb_statement", "seed": "b44-analytics/gradeledger/R/transcript.R", "signature_region": false, "structural_support": 0.515, "target_lines": 11, "test_file": false, "unit": "termSummary", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 115, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 405, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -27,3 +27,14 @@
   value <- tapply(records$credits * passed, f, sum, default = 0)
   out$earned <- as.numeric(value)
   # Field `failed`
+  value <- tapply(!passed, f, sum, default = 0L)
+  out$failed <- as.integer(value)
+  # Field `qualityPoints`
+  value <- tapply(records$credits * records$points, f, sum, default = 0)
+  out$qualityPoints <- as.numeric(value)
+  # Field `gpa`
+  value <- out$qualityPoints / out$attempted
+  out$gpa <- round(value, 2)
+
+  return(out)
+}
```

**Target (gold region)**

```r
  value <- tapply(!passed, f, sum, default = 0L)
  out$failed <- as.integer(value)
  # Field `qualityPoints`
  value <- tapply(records$credits * records$points, f, sum, default = 0)
  out$qualityPoints <- as.numeric(value)
  # Field `gpa`
  value <- out$qualityPoints / out$attempted
  out$gpa <- round(value, 2)

  return(out)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/transcript.R
# ---- diagnostics (static) --------------------
# typed statements: 10 of 17
# args not yet referenced: none
# unclosed braces in typed code: 1
# ----------------------------------------------
#' Summarise each term
#'
#' Courses with zero grade points count as attempted but not earned.
#'
#' @inheritParams validateRecords
#' @export
#' @return A data frame with columns `term`, `courses`, `attempted`, `earned`,
#'   `failed`, `qualityPoints` and `gpa`, one row per term.
termSummary <- function(records) {
  terms <- sort(unique(records$term))
  f <- factor(records$term, levels = terms)
  passed <- records$points > 0
  out <- data.frame(term = terms)

  # Field `courses`
  value <- tapply(records$course, f, length, default = 0L)
  out$courses <- as.integer(value)
  # Field `attempted`
  value <- tapply(records$credits, f, sum, default = 0)
  out$attempted <- as.numeric(value)
  # Field `earned`
  value <- tapply(records$credits * passed, f, sum, default = 0)
  out$earned <- as.numeric(value)
  # Field `failed`
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

## 33. `08b241d80f9ee1ffec3799e7` finish_block / sibling:medium

- package `synth:tablemill`, path `R/plain.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b26-reporting/tablemill/R/plain.R", "signature_region": false, "structural_support": 0.69, "target_lines": 9, "test_file": false, "unit": "truncateText", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 53, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 436, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -36,3 +36,12 @@
     middle = {
       head <- ceiling(keep / 2)
       tail <- keep - head
+      paste0(
+        substr(y, 1L, head),
+        ellipsis,
+        substr(y, nchar(y) - tail + 1L, nchar(y))
+      )
+    }
+  )
+  x
+}
```

**Target (gold region)**

```r
      paste0(
        substr(y, 1L, head),
        ellipsis,
        substr(y, nchar(y) - tail + 1L, nchar(y))
      )
    }
  )
  x
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/plain.R
#' Shorten text to a maximum width
#'
#' @description
#' Text longer than `width` loses characters on the chosen side, which is
#' replaced by `ellipsis`.
#'
#' @param x A character vector.
#' @param width Maximum width in characters, including the ellipsis.
#' @param side One of `"right"`, `"left"` or `"middle"`.
#' @param ellipsis String marking the removed characters.
#' @return A character vector the same length as `x`.
#' @export
truncateText <- function(
  x,
  width,
  side = c("right", "left", "middle"),
  ellipsis = "..."
) {
  side <- match.arg(side)
  x <- as.character(x)
  long <- !is.na(x) & nchar(x) > width
  keep <- width - nchar(ellipsis)
  if (any(long) && keep < 1L) {
    stop("`width` must leave room for at least one character.")
  }

  y <- x[long]
  x[long] <- switch(
    side,
    right = {
      paste0(substr(y, 1L, keep), ellipsis)
    },
    left = {
      paste0(ellipsis, substr(y, nchar(y) - keep + 1L, nchar(y)))
    },
    middle = {
      head <- ceiling(keep / 2)
      tail <- keep - head
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

## 34. `10179c9b50cbf621603eb6d8` finish_block / sibling:medium

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

## 35. `7e75792b8fab1a7f3b7d6e23` finish_block / sibling:medium

- package `synth:plotsmith`, path `R/scales.R`
- generator: `{"behaviour": "file_checked_not_entered", "bucket": "medium", "determinacy": "sibling", "full_body": true, "novel_tokens": ["default:aesthetics"], "novelty": 1, "rule": "fb_statement", "seed": "b07-plotting/plotsmith/R/scales.R", "signature_region": true, "structural_support": 1.0, "target_lines": 17, "test_file": false, "unit": "scale_fill_plotsmith", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 122, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 607, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -58,3 +58,20 @@
 #' @rdname scale_colour_plotsmith
 #' @export
 scale_fill_plotsmith <- function(reverse = FALSE, ...) {
+  pal <- plotsmith_colours()
+  if (reverse) {
+    pal <- rev(pal)
+  }
+
+  # Recycling colours would silently merge groups, so refuse instead
+  ggplot2::discrete_scale(
+    aesthetics = "fill",
+    palette = function(n) {
+      if (n > length(pal)) {
+        stop("The plotsmith palette has only ", length(pal), " colours.")
+      }
+      pal[seq_len(n)]
+    },
+    ...
+  )
+}
```

**Target (gold region)**

```r
scale_fill_plotsmith <- function(reverse = FALSE, ...) {
  pal <- plotsmith_colours()
  if (reverse) {
    pal <- rev(pal)
  }

  # Recycling colours would silently merge groups, so refuse instead
  ggplot2::discrete_scale(
    aesthetics = "fill",
    palette = function(n) {
      if (n > length(pal)) {
        stop("The plotsmith palette has only ", length(pal), " colours.")
      }
      pal[seq_len(n)]
    },
    ...
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/scales.R
#' @param digits Number of significant digits to keep.
#' @param ... Passed to [ggplot2::scale_x_continuous()] or
#'   [ggplot2::scale_y_continuous()].
#' @return A ggplot2 scale.
#' @export
scale_x_compact <- function(digits = 3, ...) {
  ggplot2::scale_x_continuous(
    labels = function(x) label_compact(x, digits = digits),
    ...
  )
}

#' @rdname scale_x_compact
#' @export
scale_y_compact <- function(digits = 3, ...) {
  ggplot2::scale_y_continuous(
    labels = function(x) label_compact(x, digits = digits),
    ...
  )
}

#' House qualitative colours
#'
#' @return A character vector of hex colours.
#' @noRd
plotsmith_colours <- function() {
  c("#3b6e8f", "#d1793a", "#5b9a68", "#b8484f", "#8a6fae", "#a38b5a", "#5f6b75")
}

#' Discrete house colour and fill scales
#'
#' Maps a discrete variable to the plotsmith qualitative palette, failing
#' loudly when there are more levels than palette colours.
#'
#' @param reverse Whether to use the palette in reverse order.
#' @param ... Passed to [ggplot2::discrete_scale()].
#' @return A ggplot2 scale.
#' @export
scale_colour_plotsmith <- function(reverse = FALSE, ...) {
  pal <- plotsmith_colours()
  if (reverse) {
    pal <- rev(pal)
  }

  # Recycling colours would silently merge groups, so refuse instead
  ggplot2::discrete_scale(
    aesthetics = "colour",
    palette = function(n) {
      if (n > length(pal)) {
        stop("The plotsmith palette has only ", length(pal), " colours.")
      }
      pal[seq_len(n)]
    },
    ...
  )
}

#' @rdname scale_colour_plotsmith
#' @export
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
scale_fill_plotsmith <- function(reverse = FALSE, ...) {<|user_cursor|>
=======
<[fim-middle]>
````

</details>

## 36. `1dd36171059aca3e817f6146` finish_block / sibling:medium

- package `synth:parcelgrid`, path `R/extent.R`
- generator: `{"behaviour": "file_checked_not_entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["ymax", "ymin"], "novelty": 2, "rule": "fb_statement", "seed": "b29-s4/parcelgrid/R/extent.R", "signature_region": false, "structural_support": 0.643, "target_lines": 14, "test_file": false, "unit": "extent_buffer", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 120, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 319, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -18,3 +18,17 @@
     x@xmax <- mid
   } else {
     x@xmin <- x@xmin - dx
+    x@xmax <- x@xmax + dx
+  }
+
+  # Along y
+  if (x@ymax - x@ymin + 2 * dy < 0) {
+    mid <- (x@ymin + x@ymax) / 2
+    x@ymin <- mid
+    x@ymax <- mid
+  } else {
+    x@ymin <- x@ymin - dy
+    x@ymax <- x@ymax + dy
+  }
+  return(x)
+}
```

**Target (gold region)**

```r
    x@xmax <- x@xmax + dx
  }

  # Along y
  if (x@ymax - x@ymin + 2 * dy < 0) {
    mid <- (x@ymin + x@ymax) / 2
    x@ymin <- mid
    x@ymax <- mid
  } else {
    x@ymin <- x@ymin - dy
    x@ymax <- x@ymax + dy
  }
  return(x)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/extent.R
#' Grow or shrink an extent
#'
#' Moves every bound outwards by `dx` along x and `dy` along y. Negative
#' values shrink the extent; shrinking past the centre collapses that axis
#' onto its midpoint.
#'
#' @param x An `Extent` object.
#' @param dx Distance added on both sides along x.
#' @param dy Distance added on both sides along y. Defaults to `dx`.
#' @return The buffered `Extent` object.
#' @family extent
#' @export
extent_buffer <- function(x, dx, dy = dx) {
  # Along x
  if (x@xmax - x@xmin + 2 * dx < 0) {
    mid <- (x@xmin + x@xmax) / 2
    x@xmin <- mid
    x@xmax <- mid
  } else {
    x@xmin <- x@xmin - dx
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

## 37. `6ef918735c1e3cf7f6361adc` finish_block / sibling:medium

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

## 38. `367c90af90a05bc399157d44` finish_block / sibling:medium

- package `synth:scenariokit`, path `R/scenarios.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["elt:paired.se", "elt:ratio", "paired.se"], "novelty": 3, "rule": "fb_statement", "seed": "b38-simulation/scenariokit/R/scenarios.R", "signature_region": false, "structural_support": 0.556, "target_lines": 10, "test_file": false, "unit": "scenario.contrast", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 73, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 435, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -29,3 +29,13 @@
   # Column `difference`
   out$difference <- out$mean - mean(ref)
   # Column `ratio`
+  out$ratio <- out$mean / mean(ref)
+  # Column `paired.se`
+  out$paired.se <- vapply(
+    values,
+    function(v) stats::sd(v - ref) / sqrt(length(v)),
+    numeric(1),
+    USE.NAMES = FALSE
+  )
+  out
+}
```

**Target (gold region)**

```r
  out$ratio <- out$mean / mean(ref)
  # Column `paired.se`
  out$paired.se <- vapply(
    values,
    function(v) stats::sd(v - ref) / sqrt(length(v)),
    numeric(1),
    USE.NAMES = FALSE
  )
  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/scenarios.R
#' Contrast scenarios against a reference
#'
#' @param results A named list of replicate lists, as returned by
#'   [run.policy.experiment()].
#' @param metric A function of one trajectory returning a single number.
#' @param reference Name of the reference scenario.
#' @return A data frame with columns `scenario`, `mean`, `se` (standard
#'   error of the mean), `difference` (to the reference mean), `ratio` and
#'   `paired.se` (standard error of the replicate-wise differences, valid
#'   under common random numbers).
#' @export
scenario.contrast <- function(results, metric, reference = "baseline") {
  if (!reference %in% names(results)) {
    stop(sprintf("Reference scenario `%s` not found.", reference))
  }
  values <- lapply(results, function(runs) vapply(runs, metric, numeric(1)))
  ref <- values[[reference]]
  out <- data.frame(scenario = names(values))

  # Column `mean`
  out$mean <- vapply(values, mean, numeric(1), USE.NAMES = FALSE)
  # Column `se`
  out$se <- vapply(
    values,
    function(v) stats::sd(v) / sqrt(length(v)),
    numeric(1),
    USE.NAMES = FALSE
  )
  # Column `difference`
  out$difference <- out$mean - mean(ref)
  # Column `ratio`
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

## 39. `0eaa559199e1ebb33dded73c` finish_block / sibling:medium

- package `synth:scenariokit`, path `R/summaries.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": ["elt:time", "which.max"], "novelty": 2, "rule": "fb_nested_close", "seed": "b38-simulation/scenariokit/R/summaries.R", "signature_region": false, "structural_support": 0.718, "target_lines": 17, "test_file": false, "unit": "metric.peak.time", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 111, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 439, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -35,3 +35,20 @@
 metric.peak.time <- function(runs, column = "I") {
   if (!is.list(runs) || length(runs) == 0L) {
     stop("`runs` must be a non-empty list of trajectories.")
+  }
+  values <- vapply(
+    runs,
+    function(run) {
+      if (!column %in% names(run)) {
+        stop(sprintf("Column `%s` not found in a run.", column))
+      }
+      if (anyNA(run[[column]])) {
+        return(NA_real_)
+      }
+      run$time[[which.max(run[[column]])]]
+    },
+    numeric(1)
+  )
+  names(values) <- names(runs)
+  values
+}
```

**Target (gold region)**

```r
  }
  values <- vapply(
    runs,
    function(run) {
      if (!column %in% names(run)) {
        stop(sprintf("Column `%s` not found in a run.", column))
      }
      if (anyNA(run[[column]])) {
        return(NA_real_)
      }
      run$time[[which.max(run[[column]])]]
    },
    numeric(1)
  )
  names(values) <- names(runs)
  values
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/summaries.R
#' Per-replicate outcome metrics
#'
#' Each metric maps a list of trajectories to one value per trajectory;
#' trajectories with missing values in `column` give `NA`.
#'
#' @param runs A list of trajectories, each a data frame with a `time`
#'   column, as returned by [toy.epidemic()].
#' @param column Column the metric is computed from.
#' @return A numeric vector with one value per element of `runs`.
#' @rdname outcome-metrics
#' @export
metric.peak.size <- function(runs, column = "I") {
  if (!is.list(runs) || length(runs) == 0L) {
    stop("`runs` must be a non-empty list of trajectories.")
  }
  values <- vapply(
    runs,
    function(run) {
      if (!column %in% names(run)) {
        stop(sprintf("Column `%s` not found in a run.", column))
      }
      if (anyNA(run[[column]])) {
        return(NA_real_)
      }
      max(run[[column]])
    },
    numeric(1)
  )
  names(values) <- names(runs)
  values
}

#' @rdname outcome-metrics
#' @export
metric.peak.time <- function(runs, column = "I") {
  if (!is.list(runs) || length(runs) == 0L) {
    stop("`runs` must be a non-empty list of trajectories.")
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

## 40. `f9b12d904291dd1f9e390c21` finish_block / sibling:medium

- package `synth:geoprism`, path `R/bbox.R`
- generator: `{"behaviour": "entered", "bucket": "medium", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b42-geo/geoprism/R/bbox.R", "signature_region": false, "structural_support": 0.514, "target_lines": 13, "test_file": false, "unit": "bbox_combine", "variant": "outline"}`
- verification: `{"air_idempotent": true, "completion_tokens": 170, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 548, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -32,3 +32,16 @@
   if (op == "union") {
     xmin <- min(a[["xmin"]], b[["xmin"]])
     ymin <- min(a[["ymin"]], b[["ymin"]])
+    xmax <- max(a[["xmax"]], b[["xmax"]])
+    ymax <- max(a[["ymax"]], b[["ymax"]])
+  } else {
+    xmin <- max(a[["xmin"]], b[["xmin"]])
+    ymin <- max(a[["ymin"]], b[["ymin"]])
+    xmax <- min(a[["xmax"]], b[["xmax"]])
+    ymax <- min(a[["ymax"]], b[["ymax"]])
+    if (xmin > xmax || ymin > ymax) {
+      return(NULL)
+    }
+  }
+  c(xmin = xmin, ymin = ymin, xmax = xmax, ymax = ymax)
+}
```

**Target (gold region)**

```r
    xmax <- max(a[["xmax"]], b[["xmax"]])
    ymax <- max(a[["ymax"]], b[["ymax"]])
  } else {
    xmin <- max(a[["xmin"]], b[["xmin"]])
    ymin <- max(a[["ymin"]], b[["ymin"]])
    xmax <- min(a[["xmax"]], b[["xmax"]])
    ymax <- min(a[["ymax"]], b[["ymax"]])
    if (xmin > xmax || ymin > ymax) {
      return(NULL)
    }
  }
  c(xmin = xmin, ymin = ymin, xmax = xmax, ymax = ymax)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/bbox.R
# ---- outline --------------------------------
# fn: bbox_combine(a, b, op)
# plan: 8 top-level statements; 6 typed above the cursor
# docs: yes
# ----------------------------------------------
#' Union or intersection of two bounding boxes
#'
#' @param a,b Named numeric vectors `c(xmin, ymin, xmax, ymax)`.
#' @param op `"union"` or `"intersection"`.
#' @return A bounding box, or `NULL` when an intersection is empty.
#' @export
bbox_combine <- function(a, b, op = c("union", "intersection")) {
  op <- match.arg(op)
  sides <- c("xmin", "ymin", "xmax", "ymax")

  # Validate `a`
  if (!is.numeric(a) || !all(sides %in% names(a))) {
    stop(sprintf("`%s` must be a bounding box.", "a"), call. = FALSE)
  }
  if (a[["xmin"]] > a[["xmax"]] || a[["ymin"]] > a[["ymax"]]) {
    stop(sprintf("`%s` has min greater than max.", "a"), call. = FALSE)
  }

  # Validate `b`
  if (!is.numeric(b) || !all(sides %in% names(b))) {
    stop(sprintf("`%s` must be a bounding box.", "b"), call. = FALSE)
  }
  if (b[["xmin"]] > b[["xmax"]] || b[["ymin"]] > b[["ymax"]]) {
    stop(sprintf("`%s` has min greater than max.", "b"), call. = FALSE)
  }

  if (op == "union") {
    xmin <- min(a[["xmin"]], b[["xmin"]])
    ymin <- min(a[["ymin"]], b[["ymin"]])
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

## 41. `21e44a36996dbc062562d1c1` finish_block / sibling:long

- package `synth:matwright`, path `R/decompose.R`
- generator: `{"behaviour": "entered", "bucket": "long", "determinacy": "sibling", "full_body": false, "novel_tokens": ["is.unsorted", "rev"], "novelty": 2, "rule": "fb_statement", "seed": "b33-numeric/matwright/R/decompose.R", "signature_region": false, "structural_support": 0.536, "target_lines": 21, "test_file": false, "unit": "decompCheck", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 166, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 666, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -58,3 +58,24 @@
       )
     },
     cholesky = {
+      c(
+        lower = all(abs(f$L[upper.tri(f$L)]) <= tol),
+        positiveDiag = all(diag(f$L) > 0)
+      )
+    },
+    eigen = {
+      c(
+        orthogonal = max(abs(crossprod(f$vectors) - I)) <= tol,
+        sorted = !is.unsorted(rev(f$values))
+      )
+    },
+    svd = {
+      c(
+        orthogonal = max(abs(crossprod(f$u) - I)) <= tol &&
+          max(abs(crossprod(f$v) - I)) <= tol,
+        nonNegative = all(f$d >= 0),
+        sorted = !is.unsorted(rev(f$d))
+      )
+    }
+  )
+}
```

**Target (gold region)**

```r
      c(
        lower = all(abs(f$L[upper.tri(f$L)]) <= tol),
        positiveDiag = all(diag(f$L) > 0)
      )
    },
    eigen = {
      c(
        orthogonal = max(abs(crossprod(f$vectors) - I)) <= tol,
        sorted = !is.unsorted(rev(f$values))
      )
    },
    svd = {
      c(
        orthogonal = max(abs(crossprod(f$u) - I)) <= tol &&
          max(abs(crossprod(f$v) - I)) <= tol,
        nonNegative = all(f$d >= 0),
        sorted = !is.unsorted(rev(f$d))
      )
    }
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/decompose.R
#' @return The solution vector `x` of `A x = b`.
#' @export
solveWith <- function(dec, b) {
  stopifnot(inherits(dec, "mw_decomp"))
  f <- dec$factors
  switch(
    dec$method,
    lu = {
      y <- forwardSub(f$L, b[f$perm])
      backSub(f$U, y)
    },
    qr = {
      y <- backSub(f$R, drop(crossprod(f$Q, b)))
      y[order(f$pivot)]
    },
    cholesky = {
      y <- forwardSub(f$L, b)
      backSub(t(f$L), y)
    },
    eigen = {
      y <- drop(crossprod(f$vectors, b)) / f$values
      drop(f$vectors %*% y)
    },
    svd = {
      y <- drop(crossprod(f$u, b)) / f$d
      drop(f$v %*% y)
    }
  )
}

#' Check the structure of a decomposition
#'
#' @param dec An object of class `mw_decomp` from [matDecompose()].
#' @param tol Absolute tolerance used in every comparison.
#' @family decompositions
#' @return A named logical vector. Its elements depend on the method: `lu`
#'   gives `unitLower` and `upper`; `qr` gives `orthogonal` and `upper`;
#'   `cholesky` gives `lower` and `positiveDiag`; `eigen` gives `orthogonal`
#'   and `sorted`; `svd` gives `orthogonal`, `nonNegative` and `sorted`.
#' @export
decompCheck <- function(dec, tol = 1e-10) {
  stopifnot(inherits(dec, "mw_decomp"))
  f <- dec$factors
  I <- diag(dec$dim[1])
  switch(
    dec$method,
    lu = {
      c(
        unitLower = all(abs(f$L[upper.tri(f$L)]) <= tol) &&
          all(abs(diag(f$L) - 1) <= tol),
        upper = all(abs(f$U[lower.tri(f$U)]) <= tol)
      )
    },
    qr = {
      c(
        orthogonal = max(abs(crossprod(f$Q) - I)) <= tol,
        upper = all(abs(f$R[lower.tri(f$R)]) <= tol)
      )
    },
    cholesky = {
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

## 42. `8635933ef30f4901e90938c9` finish_block / sibling:long

- package `synth:plotloom`, path `R/axes.R`
- generator: `{"behaviour": "entered", "bucket": "long", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b37-plotting/plotloom/R/axes.R", "signature_region": false, "structural_support": 1.0, "target_lines": 21, "test_file": false, "unit": "draw_axis_y", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 273, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 731, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -58,3 +58,24 @@
 ) {
   if (!side %in% c(2L, 4L)) {
     stop("`side` must be 2 or 4 for the y axis.")
+  }
+  if (!is.numeric(lim) || length(lim) != 2L || anyNA(lim)) {
+    stop("`lim` must be two non-missing numbers.")
+  }
+  breaks <- pretty(lim, n = n)
+  breaks <- breaks[breaks >= min(lim) & breaks <= max(lim)]
+  labels <- axis_label_format(breaks, style = style)
+  axis(side, at = breaks, labels = labels, col = col, col.axis = col, las = 1)
+
+  minor_at <- numeric()
+  if (minor && length(breaks) > 1L) {
+    step <- (breaks[[2]] - breaks[[1]]) / 2
+    minor_at <- c(breaks[[1]] - step, breaks + step)
+    minor_at <- minor_at[minor_at >= min(lim) & minor_at <= max(lim)]
+    axis(side, at = minor_at, labels = FALSE, tcl = -0.25, col = col)
+  }
+  if (!is.null(title)) {
+    mtext(title, side = side, line = 2.5, col = col)
+  }
+  invisible(list(breaks = breaks, labels = labels, minor = minor_at))
+}
```

**Target (gold region)**

```r
  }
  if (!is.numeric(lim) || length(lim) != 2L || anyNA(lim)) {
    stop("`lim` must be two non-missing numbers.")
  }
  breaks <- pretty(lim, n = n)
  breaks <- breaks[breaks >= min(lim) & breaks <= max(lim)]
  labels <- axis_label_format(breaks, style = style)
  axis(side, at = breaks, labels = labels, col = col, col.axis = col, las = 1)

  minor_at <- numeric()
  if (minor && length(breaks) > 1L) {
    step <- (breaks[[2]] - breaks[[1]]) / 2
    minor_at <- c(breaks[[1]] - step, breaks + step)
    minor_at <- minor_at[minor_at >= min(lim) & minor_at <= max(lim)]
    axis(side, at = minor_at, labels = FALSE, tcl = -0.25, col = col)
  }
  if (!is.null(title)) {
    mtext(title, side = side, line = 2.5, col = col)
  }
  invisible(list(breaks = breaks, labels = labels, minor = minor_at))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/axes.R
#' @return Invisibly, a list with elements `breaks`, `labels` and `minor`.
#' @rdname draw_axis
#' @export
draw_axis_x <- function(
  lim,
  n = 5L,
  style = "plain",
  side = 1L,
  col = "grey30",
  minor = TRUE,
  title = NULL
) {
  if (!side %in% c(1L, 3L)) {
    stop("`side` must be 1 or 3 for the x axis.")
  }
  if (!is.numeric(lim) || length(lim) != 2L || anyNA(lim)) {
    stop("`lim` must be two non-missing numbers.")
  }
  breaks <- pretty(lim, n = n)
  breaks <- breaks[breaks >= min(lim) & breaks <= max(lim)]
  labels <- axis_label_format(breaks, style = style)
  axis(side, at = breaks, labels = labels, col = col, col.axis = col, las = 1)

  minor_at <- numeric()
  if (minor && length(breaks) > 1L) {
    step <- (breaks[[2]] - breaks[[1]]) / 2
    minor_at <- c(breaks[[1]] - step, breaks + step)
    minor_at <- minor_at[minor_at >= min(lim) & minor_at <= max(lim)]
    axis(side, at = minor_at, labels = FALSE, tcl = -0.25, col = col)
  }
  if (!is.null(title)) {
    mtext(title, side = side, line = 2.5, col = col)
  }
  invisible(list(breaks = breaks, labels = labels, minor = minor_at))
}

#' Draw the vertical axis
#'
#' @param lim Axis limits as a length-two numeric vector.
#' @param n Approximate number of ticks.
#' @param style Label style passed to [axis_label_format()].
#' @param side Side of the plot, `2` (left) or `4` (right).
#' @param col Colour of the axis line, ticks and title.
#' @param minor Whether to add unlabelled ticks halfway between the labelled
#'   ones.
#' @param title Optional axis title drawn in the margin.
#' @return Invisibly, a list with elements `breaks`, `labels` and `minor`.
#' @rdname draw_axis
#' @export
draw_axis_y <- function(
  lim,
  n = 5L,
  style = "plain",
  side = 2L,
  col = "grey30",
  minor = TRUE,
  title = NULL
) {
  if (!side %in% c(2L, 4L)) {
    stop("`side` must be 2 or 4 for the y axis.")
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

## 43. `f21d0195ad69335db92ec7e5` finish_block / sibling:long

- package `synth:dashkit`, path `R/validate_inputs.R`
- generator: `{"behaviour": "entered", "bucket": "long", "determinacy": "sibling", "full_body": false, "novel_tokens": ["elt:product", "elt:segment"], "novelty": 2, "rule": "fb_comment_lead", "seed": "b36-shiny/dashkit/R/validate_inputs.R", "signature_region": false, "structural_support": 0.984, "target_lines": 23, "test_file": false, "unit": "check_filter_inputs", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 223, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 489, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -38,3 +38,26 @@
     )
   }
 
+  # Validate `segment`
+  if (length(segment) == 0L) {
+    problems[["segment"]] <- sprintf("Select at least one `%s`.", "segment")
+  } else if (length(unknown <- setdiff(segment, choices$segment)) > 0L) {
+    problems[["segment"]] <- sprintf(
+      "Unknown `%s`: %s.",
+      "segment",
+      paste(unknown, collapse = ", ")
+    )
+  }
+
+  # Validate `product`
+  if (length(product) == 0L) {
+    problems[["product"]] <- sprintf("Select at least one `%s`.", "product")
+  } else if (length(unknown <- setdiff(product, choices$product)) > 0L) {
+    problems[["product"]] <- sprintf(
+      "Unknown `%s`: %s.",
+      "product",
+      paste(unknown, collapse = ", ")
+    )
+  }
+  problems
+}
```

**Target (gold region)**

```r
  # Validate `segment`
  if (length(segment) == 0L) {
    problems[["segment"]] <- sprintf("Select at least one `%s`.", "segment")
  } else if (length(unknown <- setdiff(segment, choices$segment)) > 0L) {
    problems[["segment"]] <- sprintf(
      "Unknown `%s`: %s.",
      "segment",
      paste(unknown, collapse = ", ")
    )
  }

  # Validate `product`
  if (length(product) == 0L) {
    problems[["product"]] <- sprintf("Select at least one `%s`.", "product")
  } else if (length(unknown <- setdiff(product, choices$product)) > 0L) {
    problems[["product"]] <- sprintf(
      "Unknown `%s`: %s.",
      "product",
      paste(unknown, collapse = ", ")
    )
  }
  problems
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/validate_inputs.R
#' Check the categorical filter inputs
#'
#' Each filter must have at least one selection, and every selection must be
#' one of the offered choices.
#'
#' @param region Selected regions.
#' @param channel Selected sales channels.
#' @param segment Selected customer segments.
#' @param product Selected product lines.
#' @param choices A named list of allowed values with elements `region`,
#'   `channel`, `segment` and `product`.
#' @return A named character vector with one message per failing filter;
#'   empty when every filter is valid.
#' @family input checks
#' @export
check_filter_inputs <- function(region, channel, segment, product, choices) {
  problems <- character(0)

  # Validate `region`
  if (length(region) == 0L) {
    problems[["region"]] <- sprintf("Select at least one `%s`.", "region")
  } else if (length(unknown <- setdiff(region, choices$region)) > 0L) {
    problems[["region"]] <- sprintf(
      "Unknown `%s`: %s.",
      "region",
      paste(unknown, collapse = ", ")
    )
  }

  # Validate `channel`
  if (length(channel) == 0L) {
    problems[["channel"]] <- sprintf("Select at least one `%s`.", "channel")
  } else if (length(unknown <- setdiff(channel, choices$channel)) > 0L) {
    problems[["channel"]] <- sprintf(
      "Unknown `%s`: %s.",
      "channel",
      paste(unknown, collapse = ", ")
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

## 44. `82b8caa9795d9fc1d0e53d74` finish_block / sibling:long

- package `synth:meridiana`, path `R/units.R`
- generator: `{"behaviour": "entered", "bucket": "long", "determinacy": "sibling", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b42-geo/meridiana/R/units.R", "signature_region": false, "structural_support": 0.973, "target_lines": 22, "test_file": false, "unit": "angle.parts", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 120, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 326, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -30,3 +30,25 @@
       a <- a %% 24
       whole <- floor(a)
       minutes <- floor((a - whole) * 60)
+      seconds <- ((a - whole) * 60 - minutes) * 60
+      data.frame(
+        sign = sign,
+        whole = whole,
+        minutes = minutes,
+        seconds = seconds
+      )
+    },
+    hm = {
+      a <- a %% 24
+      whole <- floor(a)
+      minutes <- (a - whole) * 60
+      seconds <- 0
+      data.frame(
+        sign = sign,
+        whole = whole,
+        minutes = minutes,
+        seconds = seconds
+      )
+    }
+  )
+}
```

**Target (gold region)**

```r
      seconds <- ((a - whole) * 60 - minutes) * 60
      data.frame(
        sign = sign,
        whole = whole,
        minutes = minutes,
        seconds = seconds
      )
    },
    hm = {
      a <- a %% 24
      whole <- floor(a)
      minutes <- (a - whole) * 60
      seconds <- 0
      data.frame(
        sign = sign,
        whole = whole,
        minutes = minutes,
        seconds = seconds
      )
    }
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/units.R
angle.parts <- function(x, unit = c("dms", "dm", "hms", "hm")) {
  unit <- match.arg(unit)
  sign <- ifelse(x < 0, -1, 1)
  a <- abs(x)
  switch(
    unit,
    dms = {
      whole <- floor(a)
      minutes <- floor((a - whole) * 60)
      seconds <- ((a - whole) * 60 - minutes) * 60
      data.frame(
        sign = sign,
        whole = whole,
        minutes = minutes,
        seconds = seconds
      )
    },
    dm = {
      whole <- floor(a)
      minutes <- (a - whole) * 60
      seconds <- 0
      data.frame(
        sign = sign,
        whole = whole,
        minutes = minutes,
        seconds = seconds
      )
    },
    hms = {
      a <- a %% 24
      whole <- floor(a)
      minutes <- floor((a - whole) * 60)
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

## 45. `2021ed9b263a8673c9099903` finish_block / constrained_short

- package `synth:terraslab`, path `R/extract.R`
- generator: `{"behaviour": "no_check", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b13-geo/terraslab/R/extract.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "zonalByRaster", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 10, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 172, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -2,3 +2,5 @@
   terra::compareGeom(r, zones, stopOnError = TRUE)
   out <- terra::zonal(r, zones, fun = fun, na.rm = TRUE)
   names(out)[1] <- "zone"
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
<filename>R/extract.R
zonalByRaster <- function(r, zones, fun = "mean") {
  terra::compareGeom(r, zones, stopOnError = TRUE)
  out <- terra::zonal(r, zones, fun = fun, na.rm = TRUE)
  names(out)[1] <- "zone"
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

## 46. `b89a5fe2410d5bc29a6fd9b4` finish_block / constrained_short

- package `synth:rangeweft`, path `R/coverage.R`
- generator: `{"behaviour": "file_checked_not_entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b17-bioc/rangeweft/R/coverage.R", "signature_region": false, "structural_support": 0.0, "target_lines": 2, "test_file": false, "unit": "spanCoverage", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 11, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 445, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -34,3 +34,5 @@
       weight[idx]
     )
   }
+  return(out)
+}
```

**Target (gold region)**

```r
  return(out)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/coverage.R
# ---- diagnostics (static) --------------------
# typed statements: 6 of 7
# args not yet referenced: none
# unclosed braces in typed code: 1
# ----------------------------------------------
#' Per-base coverage of an interval set
#'
#' Computes run-length encoded coverage for every sequence in `seqLengths`.
#'
#' @export
#' @param x A `SpanSet` object.
#' @param seqLengths Named integer vector of sequence lengths.
#' @param weight Either a single number, a numeric vector with one weight per
#'   interval, or the name of a metadata column holding weights.
#' @return A named list of `rle` objects, one per sequence.
spanCoverage <- function(x, seqLengths, weight = 1) {
  n <- length(x@start)
  if (is.character(weight)) {
    if (!weight %in% names(x@mcols)) {
      stop(sprintf("no metadata column named '%s'", weight))
    }
    weight <- x@mcols[[weight]]
  }
  weight <- rep_len(as.numeric(weight), n)

  out <- vector("list", length(seqLengths))
  names(out) <- names(seqLengths)
  for (s in names(seqLengths)) {
    idx <- which(x@seqnames == s)
    out[[s]] <- .coverageOnSeq(
      x@start[idx],
      x@end[idx],
      as.integer(seqLengths[[s]]),
      weight[idx]
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

## 47. `c9ff2f154a7d1eb9c7ececba` finish_block / constrained_short

- package `synth:diffledger`, path `R/normalise.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b17-bioc/diffledger/R/normalise.R", "signature_region": false, "structural_support": 0.0, "target_lines": 4, "test_file": false, "unit": "check_counts", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 28, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 306, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -15,3 +15,7 @@
     stop("`counts` must be non-negative")
   }
   if (!is.null(rownames(counts)) && anyDuplicated(rownames(counts)) > 0L) {
+    stop("feature names in `counts` must be unique")
+  }
+  invisible(counts)
+}
```

**Target (gold region)**

```r
    stop("feature names in `counts` must be unique")
  }
  invisible(counts)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/normalise.R
#' @title Validate a count matrix
#' @description Checks that counts are a non-negative numeric matrix without
#'   missing values and with unique feature names.
#' @param counts A numeric matrix of raw counts, features in rows.
#' @return `counts`, invisibly, when valid; otherwise an error is raised.
#' @keywords internal
check_counts <- function(counts) {
  if (!is.matrix(counts) || !is.numeric(counts)) {
    stop("`counts` must be a numeric matrix")
  }
  if (anyNA(counts)) {
    stop("`counts` must not contain missing values")
  }
  if (any(counts < 0)) {
    stop("`counts` must be non-negative")
  }
  if (!is.null(rownames(counts)) && anyDuplicated(rownames(counts)) > 0L) {
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

## 48. `10b9cb4b91cf5cbafc122a5f` finish_block / constrained_short

- package `synth:gaugewise`, path `R/units.R`
- generator: `{"behaviour": "entered", "bucket": "short", "determinacy": "constrained_short", "full_body": false, "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b20-environment/gaugewise/R/units.R", "signature_region": false, "structural_support": 0.0, "target_lines": 3, "test_file": false, "unit": "convert_columns", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 20, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 416, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -34,3 +34,6 @@
     units[[col]] <- rule[[3]]
   }
 
+  attr(data, "units") <- units
+  data
+}
```

**Target (gold region)**

```r
  attr(data, "units") <- units
  data
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/units.R
#' @title Convert several columns of a station table at once
#'
#' @description Applies a conversion specification to a data frame, converting each named
#'   column and recording the new unit in a `units` attribute.
#'
#' @param data A data frame of station observations.
#' @param spec A named list; each element is a character vector
#'   `c(quantity, from, to)` for the column of the same name.
#' @return `data` with converted columns and an updated `units` attribute.
#' @export
convert_columns <- function(data, spec) {
  units <- attr(data, "units")
  if (is.null(units)) {
    units <- list()
  }

  for (col in names(spec)) {
    if (!col %in% names(data)) {
      warning("Column `", col, "` not found; skipped.")
      next
    }

    rule <- spec[[col]]
    converter <- switch(
      rule[[1]],
      speed = convert_speed,
      pressure = convert_pressure,
      precipitation = convert_precipitation,
      discharge = convert_discharge,
      temperature = convert_temperature,
      stop("Unknown quantity for `", col, "`: ", rule[[1]])
    )
    data[[col]] <- converter(data[[col]], from = rule[[2]], to = rule[[3]])
    units[[col]] <- rule[[3]]
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

## 49. `25dae79cbe78212a6fd16365` roxygen_drafting / roxy_full

- package `synth:glyphgauge`, path `R/distance.R`
- generator: `{"behaviour": "entered", "determined_share": 0.667, "rule": "roxy_full", "seed": "b10-text/glyphgauge/R/distance.R", "tag_order": "title description param return export", "target_lines": 9, "unit": "lcsDist"}`
- verification: `{"air_idempotent": true, "completion_tokens": 95, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 522, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

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
    stop("Hamming distance needs strings of equal length.")
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

## 50. `85945ad90c59b8f9f36f36aa` roxygen_drafting / roxy_full

- package `synth:terraslab`, path `R/load.R`
- generator: `{"behaviour": "no_check", "determined_share": 0.8, "rule": "roxy_full", "seed": "b13-geo/terraslab/R/load.R", "tag_order": "param noRd", "target_lines": 5, "unit": "asRasterVector"}`
- verification: `{"air_idempotent": true, "completion_tokens": 67, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 460, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

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
    stop("Area of interest does not overlap any raster cells.")
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

## 51. `ceb4fc4b97c7bdf9a336d924` roxygen_drafting / roxy_full

- package `synth:fieldpanel`, path `R/checkInputs.R`
- generator: `{"behaviour": "entered", "determined_share": 0.692, "rule": "roxy_full", "seed": "b36-shiny/fieldpanel/R/checkInputs.R", "tag_order": "description param return family export", "target_lines": 13, "unit": "parseRateInput"}`
- verification: `{"air_idempotent": true, "completion_tokens": 139, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 610, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,6 +28,19 @@
   return(list(value = as.integer(value), error = NULL))
 }
 
+#' Parse a rate typed into a text input
+#'
+#' @description
+#' Trims the raw text, converts it to a proportion and checks its range.
+#' A trailing `%` divides the number by 100.
+#'
+#' @param x Raw text from the input.
+#' @param label Label of the input, used in messages.
+#' @param min,max Smallest and largest accepted value.
+#' @return A list with elements `value` (`NULL` on failure) and `error`
+#'   (`NULL` on success).
+#' @family input parsers
+#' @export
 parseRateInput <- function(x, label, min = 0, max = 1) {
   x <- trimws(as.character(x))
   if (length(x) != 1L || is.na(x) || !nzchar(x)) {
```

**Target (gold region)**

```r
#' Parse a rate typed into a text input
#'
#' @description
#' Trims the raw text, converts it to a proportion and checks its range.
#' A trailing `%` divides the number by 100.
#'
#' @param x Raw text from the input.
#' @param label Label of the input, used in messages.
#' @param min,max Smallest and largest accepted value.
#' @return A list with elements `value` (`NULL` on failure) and `error`
#'   (`NULL` on success).
#' @family input parsers
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/checkInputs.R
#' Trims the raw text, converts it to an integer and checks its range.
#'
#' @param x Raw text from the input.
#' @param label Label of the input, used in messages.
#' @param min,max Smallest and largest accepted value.
#' @return A list with elements `value` (`NULL` on failure) and `error`
#'   (`NULL` on success).
#' @family input parsers
#' @export
parseCountInput <- function(x, label, min = 0, max = Inf) {
  x <- trimws(as.character(x))
  if (length(x) != 1L || is.na(x) || !nzchar(x)) {
    return(list(value = NULL, error = sprintf("%s is required.", label)))
  }
  value <- suppressWarnings(as.numeric(x))
  if (is.na(value) || value != round(value)) {
    return(list(
      value = NULL,
      error = sprintf("%s must be a whole number.", label)
    ))
  }
  if (value < min || value > max) {
    return(list(
      value = NULL,
      error = sprintf("%s must be between %s and %s.", label, min, max)
    ))
  }
  return(list(value = as.integer(value), error = NULL))
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
parseRateInput <- function(x, label, min = 0, max = 1) {
  x <- trimws(as.character(x))
  if (length(x) != 1L || is.na(x) || !nzchar(x)) {
    return(list(value = NULL, error = sprintf("%s is required.", label)))
  }
  isPercent <- endsWith(x, "%")
  value <- suppressWarnings(as.numeric(sub("%$", "", x)))
  if (is.na(value)) {
    return(list(value = NULL, error = sprintf("%s must be a number.", label)))
  }
  if (isPercent) {
    value <- value / 100
  }
  if (value < min || value > max) {
    return(list(
      value = NULL,
      error = sprintf("%s must be between %s and %s.", label, min, max)
    ))
  }
  return(list(value = value, error = NULL))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 52. `4a31c3496831c7b54c3b9080` roxygen_drafting / roxy_full

- package `synth:inkplate`, path `R/escape.R`
- generator: `{"behaviour": "entered", "determined_share": 0.714, "rule": "roxy_full", "seed": "b10-text/inkplate/R/escape.R", "tag_order": "param return export", "target_lines": 7, "unit": "escape.for"}`
- verification: `{"air_idempotent": true, "completion_tokens": 68, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 448, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,6 +28,13 @@
   out
 }
 
+#' Escape text for a given output format
+#'
+#' @param x A character vector to escape.
+#' @param format Target output format.
+#' @param na How to render missing values.
+#' @return A character vector the same length as `x`.
+#' @export
 escape.for <- function(
   x,
   format = c("html", "attr", "markdown", "latex"),
```

**Target (gold region)**

```r
#' Escape text for a given output format
#'
#' @param x A character vector to escape.
#' @param format Target output format.
#' @param na How to render missing values.
#' @return A character vector the same length as `x`.
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/escape.R
#' Escape text for LaTeX
#'
#' @param x A character vector to escape.
#' @param na How to render missing values.
#' @return A character vector the same length as `x`.
#' @export
escape.latex <- function(x, na = "") {
  x <- as.character(x)
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

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
escape.for <- function(
  x,
  format = c("html", "attr", "markdown", "latex"),
  na = ""
) {
  format <- match.arg(format)
  switch(
    format,
    html = escape.html(x, na = na),
    attr = escape.attr(x, na = na),
    markdown = escape.markdown(x, na = na),
    latex = escape.latex(x, na = na)
  )
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 53. `234173c423438cbb26b1a4ca` roxygen_drafting / roxy_full

- package `synth:surveyscrub`, path `R/validate.R`
- generator: `{"behaviour": "no_check", "determined_share": 0.714, "rule": "roxy_full", "seed": "b01-tidyverse/surveyscrub/R/validate.R", "tag_order": "param return export", "target_lines": 7, "unit": "check_allowed"}`
- verification: `{"air_idempotent": true, "completion_tokens": 96, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 442, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -18,6 +18,13 @@
   )
 }
 
+#' Check that a column only contains allowed values
+#'
+#' @param data A data frame of survey responses.
+#' @param col <[`data-masking`][dplyr::dplyr_data_masking]> The column to check.
+#' @param allowed Vector of permitted values; `NA` is always permitted.
+#' @return A one-row tibble describing the check result.
+#' @export
 check_allowed <- function(data, col, allowed) {
   values <- dplyr::pull(data, {{ col }})
   bad <- !is.na(values) & !(values %in% allowed)
```

**Target (gold region)**

```r
#' Check that a column only contains allowed values
#'
#' @param data A data frame of survey responses.
#' @param col <[`data-masking`][dplyr::dplyr_data_masking]> The column to check.
#' @param allowed Vector of permitted values; `NA` is always permitted.
#' @return A one-row tibble describing the check result.
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/validate.R
#' Check that a numeric column lies within a range
#'
#' @param data A data frame of survey responses.
#' @param col <[`data-masking`][dplyr::dplyr_data_masking]> The column to check.
#' @param lower,upper Inclusive bounds of the allowed range.
#' @return A one-row tibble describing the check result.
#' @export
check_range <- function(data, col, lower, upper) {
  values <- dplyr::pull(data, {{ col }})

  # Missing answers are the business of check_missing_rate(), not this one.
  bad <- !is.na(values) & (values < lower | values > upper)
  new_check_result(
    check = paste0("range_", rlang::as_label(rlang::enquo(col))),
    passed = !any(bad),
    n_failed = sum(bad),
    detail = paste(utils::head(unique(values[bad]), 5), collapse = ", ")
  )
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
check_allowed <- function(data, col, allowed) {
  values <- dplyr::pull(data, {{ col }})
  bad <- !is.na(values) & !(values %in% allowed)
  new_check_result(
    check = paste0("allowed_", rlang::as_label(rlang::enquo(col))),
    passed = !any(bad),
    n_failed = sum(bad),
    detail = paste(utils::head(unique(values[bad]), 5), collapse = ", ")
  )
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 54. `84067fe155894f2d22a66805` roxygen_drafting / roxy_full

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

## 55. `0bb578066d2dc3bfe9955a72` roxygen_drafting / roxy_full

- package `synth:foliokit`, path `R/money.R`
- generator: `{"behaviour": "entered", "determined_share": 0.667, "rule": "roxy_full", "seed": "b05-oop/foliokit/R/money.R", "tag_order": "param return export", "target_lines": 6, "unit": "vec_ptype_abbr.foliokit_money"}`
- verification: `{"air_idempotent": true, "completion_tokens": 62, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 342, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,6 +28,12 @@
   out
 }
 
+#' Abbreviated type name for money vectors
+#'
+#' @param x A `foliokit_money` vector.
+#' @param ... Ignored.
+#' @return A single string used in tibble column headers.
+#' @export
 vec_ptype_abbr.foliokit_money <- function(x, ...) {
   tolower(attr(x, "currency"))
 }
```

**Target (gold region)**

```r
#' Abbreviated type name for money vectors
#'
#' @param x A `foliokit_money` vector.
#' @param ... Ignored.
#' @return A single string used in tibble column headers.
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/money.R
#' @export
currency <- function(x) {
  if (!is_money(x)) {
    stop("`x` must be a money vector.")
  }
  attr(x, "currency")
}

#' Format a money vector
#'
#' @param x A `foliokit_money` vector.
#' @param ... Ignored.
#' @return A character vector of formatted amounts.
#' @export
format.foliokit_money <- function(x, ...) {
  cur <- attr(x, "currency")
  values <- vctrs::vec_data(x)
  out <- formatC(
    values,
    format = "f",
    digits = minor_units(cur),
    big.mark = ","
  )

  # paste() would turn missing amounts into "USD NA", so blank them last
  out <- paste(cur, out)
  out[is.na(values)] <- NA_character_
  out
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
vec_ptype_abbr.foliokit_money <- function(x, ...) {
  tolower(attr(x, "currency"))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 56. `b6d14180104b1d713920c5b1` roxygen_drafting / roxy_full

- package `synth:itemwise`, path `R/scoring.R`
- generator: `{"behaviour": "entered", "determined_share": 0.667, "rule": "roxy_full", "seed": "b15-psych-trials/itemwise/R/scoring.R", "tag_order": "param noRd", "target_lines": 9, "unit": "check_item_range"}`
- verification: `{"air_idempotent": true, "completion_tokens": 106, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 426, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -21,6 +21,15 @@
   return(data)
 }
 
+#' Check that item responses fall inside the response scale
+#'
+#' Flags any non-missing response outside `[min, max]` or not a whole number.
+#'
+#' @param data A data frame of item responses, one row per respondent.
+#' @param items A character vector of item names making up the scale.
+#' @param min The lowest admissible response category.
+#' @param max The highest admissible response category.
+#' @noRd
 check_item_range <- function(data, items, min, max) {
   values <- unlist(data[items], use.names = FALSE)
   values <- values[!is.na(values)]
```

**Target (gold region)**

```r
#' Check that item responses fall inside the response scale
#'
#' Flags any non-missing response outside `[min, max]` or not a whole number.
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items A character vector of item names making up the scale.
#' @param min The lowest admissible response category.
#' @param max The highest admissible response category.
#' @noRd
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/scoring.R
#' Reverse-key selected columns of a response data frame
#'
#' Applies [reverse_key()] to every column named in `reverse`, leaving the
#' remaining columns untouched.
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param reverse A character vector of item names to reverse-key.
#' @param min The lowest admissible response category.
#' @param max The highest admissible response category.
#' @return `data` with the reverse-keyed columns replaced.
#' @export
reverse_key_items <- function(data, reverse, min, max) {
  missing_cols <- setdiff(reverse, names(data))
  if (length(missing_cols) > 0) {
    stop("Unknown items: ", paste(missing_cols, collapse = ", "))
  }

  for (item in reverse) {
    data[[item]] <- reverse_key(data[[item]], min = min, max = max)
  }
  return(data)
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
check_item_range <- function(data, items, min, max) {
  values <- unlist(data[items], use.names = FALSE)
  values <- values[!is.na(values)]

  bad <- values < min | values > max | values != round(values)
  if (any(bad)) {
    stop(
      sum(bad),
      " response(s) outside the scale ",
      min,
      "-",
      max,
      "."
    )
  }
  return(invisible(data))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 57. `0c426535b8e49c2ed2fd2103` roxygen_drafting / roxy_full

- package `synth:foldwise`, path `R/metrics.R`
- generator: `{"behaviour": "entered", "determined_share": 0.7, "rule": "roxy_full", "seed": "b11-ml/foldwise/R/metrics.R", "tag_order": "param return export", "target_lines": 10, "unit": "brier_vec"}`
- verification: `{"air_idempotent": true, "completion_tokens": 93, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 445, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -20,6 +20,16 @@
   -mean(p$truth * log(prob) + (1 - p$truth) * log(1 - prob))
 }
 
+#' Brier score
+#'
+#' Computes the mean squared difference between 0/1 outcomes and the given
+#' event probabilities.
+#'
+#' @param truth A numeric vector of observed values.
+#' @param estimate A numeric vector of predicted event probabilities.
+#' @param na_rm Logical; drop incomplete pairs before computing.
+#' @return A single numeric value.
+#' @export
 brier_vec <- function(truth, estimate, na_rm = TRUE) {
   p <- align_pairs(as.numeric(truth), estimate, na_rm)
   if (!all(p$truth %in% c(0, 1))) {
```

**Target (gold region)**

```r
#' Brier score
#'
#' Computes the mean squared difference between 0/1 outcomes and the given
#' event probabilities.
#'
#' @param truth A numeric vector of observed values.
#' @param estimate A numeric vector of predicted event probabilities.
#' @param na_rm Logical; drop incomplete pairs before computing.
#' @return A single numeric value.
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/metrics.R
#' Binary log loss
#'
#' Computes the mean negative log-likelihood of 0/1 outcomes under the given
#' event probabilities, clipping probabilities away from 0 and 1.
#'
#' @param truth A numeric vector of observed values.
#' @param estimate A numeric vector of predicted event probabilities.
#' @param eps Clipping tolerance applied to `estimate`.
#' @param na_rm Logical; drop incomplete pairs before computing.
#' @return A single numeric value.
#' @export
log_loss_vec <- function(truth, estimate, eps = 1e-15, na_rm = TRUE) {
  p <- align_pairs(as.numeric(truth), estimate, na_rm)
  if (!all(p$truth %in% c(0, 1))) {
    stop("`truth` must be coded as 0/1.")
  }

  # log(0) would make a single confident miss dominate the mean
  prob <- pmin(pmax(p$estimate, eps), 1 - eps)
  -mean(p$truth * log(prob) + (1 - p$truth) * log(1 - prob))
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
brier_vec <- function(truth, estimate, na_rm = TRUE) {
  p <- align_pairs(as.numeric(truth), estimate, na_rm)
  if (!all(p$truth %in% c(0, 1))) {
    stop("`truth` must be coded as 0/1.")
  }
  mean((p$truth - p$estimate)^2)
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 58. `340899cf41375965271c26f4` roxygen_drafting / roxy_full

- package `synth:relaydesk`, path `R/webhooks.R`
- generator: `{"behaviour": "entered", "determined_share": 0.667, "rule": "roxy_full", "seed": "b18-web/relaydesk/R/webhooks.R", "tag_order": "inheritParams return keywords", "target_lines": 9, "unit": "normalizeAlertEvent"}`
- verification: `{"air_idempotent": true, "completion_tokens": 106, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 509, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -27,6 +27,15 @@
   )
 }
 
+#' Normalize a monitoring alert event
+#'
+#' Maps a firing alert from the monitoring service to the common event
+#' shape used by the ticket store. Resolved alerts are ignored.
+#'
+#' @inheritParams normalizeGitEvent
+#' @return A list with `source`, `kind`, `externalId`, `title`, `actor` and
+#'   `url`, or `NULL` for events that are ignored.
+#' @keywords internal
 normalizeAlertEvent <- function(payload, eventType) {
   if (!identical(eventType, "alert") || !identical(payload$state, "firing")) {
     return(NULL)
```

**Target (gold region)**

```r
#' Normalize a monitoring alert event
#'
#' Maps a firing alert from the monitoring service to the common event
#' shape used by the ticket store. Resolved alerts are ignored.
#'
#' @inheritParams normalizeGitEvent
#' @return A list with `source`, `kind`, `externalId`, `title`, `actor` and
#'   `url`, or `NULL` for events that are ignored.
#' @keywords internal
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/webhooks.R
#' Normalize a contact form event
#'
#' Maps a submission from the hosted contact form to the common event
#' shape used by the ticket store.
#'
#' @inheritParams normalizeGitEvent
#' @return A list with `source`, `kind`, `externalId`, `title`, `actor` and
#'   `url`, or `NULL` for events that are ignored.
#' @keywords internal
normalizeFormEvent <- function(payload, eventType) {
  if (!identical(eventType, "submission.created")) {
    return(NULL)
  }

  fields <- payload$data$fields
  if (is.null(fields$subject) || is.null(fields$email)) {
    return(NULL)
  }

  list(
    source = "form",
    kind = "request",
    externalId = as.character(payload$data$submission_id),
    title = fields$subject,
    actor = tolower(fields$email),
    url = payload$data$admin_url
  )
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
normalizeAlertEvent <- function(payload, eventType) {
  if (!identical(eventType, "alert") || !identical(payload$state, "firing")) {
    return(NULL)
  }

  severity <- payload$labels$severity %||% "warning"
  list(
    source = "alert",
    kind = if (severity == "critical") "incident" else "alert",
    externalId = as.character(payload$fingerprint),
    title = sprintf(
      "[%s] %s",
      toupper(severity),
      payload$annotations$summary %||% payload$alertname %||% "alert"
    ),
    actor = payload$receiver %||% "monitoring",
    url = payload$generatorURL
  )
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 59. `0f3cd7ce75df6c0707a57a2f` roxygen_drafting / roxy_full

- package `synth:taskrelay`, path `R/logging.R`
- generator: `{"behaviour": "entered", "determined_share": 0.714, "rule": "roxy_full", "seed": "b14-pipelines/taskrelay/R/logging.R", "tag_order": "export inheritParams rdname", "target_lines": 7, "unit": "log_warn"}`
- verification: `{"air_idempotent": true, "completion_tokens": 57, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 238, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -10,6 +10,13 @@
   return(log_at(logger, "INFO", fmt, ...))
 }
 
+#' Log a warning message
+#'
+#' Shorthand for `log_at(logger, "WARN", ...)`.
+#'
+#' @export
+#' @inheritParams log_at
+#' @rdname log_shortcuts
 log_warn <- function(logger, fmt, ...) {
   return(log_at(logger, "WARN", fmt, ...))
 }
```

**Target (gold region)**

```r
#' Log a warning message
#'
#' Shorthand for `log_at(logger, "WARN", ...)`.
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

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
log_warn <- function(logger, fmt, ...) {
  return(log_at(logger, "WARN", fmt, ...))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 60. `2c76df14dd873a1a26b41083` roxygen_drafting / roxy_full

- package `synth:rendershed`, path `R/schedule.R`
- generator: `{"behaviour": "entered", "determined_share": 0.8, "rule": "roxy_full", "seed": "b14-pipelines/rendershed/R/schedule.R", "tag_order": "title param rdname keywords", "target_lines": 5, "unit": "nextRunWeekly"}`
- verification: `{"air_idempotent": true, "completion_tokens": 66, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 398, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -13,6 +13,11 @@
   candidate
 }
 
+#' @title Next run time for a weekly cadence
+#' @param cadence A cadence created by [parseCadence()].
+#' @param from The reference time; the result is strictly after it.
+#' @rdname nextRun-helpers
+#' @keywords internal
 nextRunWeekly <- function(cadence, from = Sys.time()) {
   today <- as.Date(from, tz = cadence$tz)
 
```

**Target (gold region)**

```r
#' @title Next run time for a weekly cadence
#' @param cadence A cadence created by [parseCadence()].
#' @param from The reference time; the result is strictly after it.
#' @rdname nextRun-helpers
#' @keywords internal
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/schedule.R
#' @title Next run time for a daily cadence
#' @param cadence A cadence created by [parseCadence()].
#' @param from The reference time; the result is strictly after it.
#' @return A `POSIXct` time stamp.
#' @rdname nextRun-helpers
#' @keywords internal
nextRunDaily <- function(cadence, from = Sys.time()) {
  today <- as.Date(from, tz = cadence$tz)
  candidate <- atTimeOfDay(today, cadence)
  if (candidate <= from) {
    candidate <- atTimeOfDay(today + 1L, cadence)
  }
  candidate
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
nextRunWeekly <- function(cadence, from = Sys.time()) {
  today <- as.Date(from, tz = cadence$tz)

  # %u numbers weekdays 1 (Monday) to 7, the same order as weekdayCodes
  target <- match(cadence$anchor, weekdayCodes)
  current <- as.integer(format(today, "%u"))
  offset <- (target - current) %% 7L

  candidate <- atTimeOfDay(today + offset, cadence)
  if (candidate <= from) {
    candidate <- atTimeOfDay(today + offset + 7L, cadence)
  }
  candidate
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>
