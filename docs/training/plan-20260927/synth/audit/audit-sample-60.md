# SYNTH-v1 audit sample

Stratified random sample (seed 3407) of 60 of 5370 cases, drawn from `/mnt/e/sepalith/campaign-20260915/data-work/SYNTH-v1`. Strata: {"finish_block": {"short": 18, "medium": 10, "long": 8}, "roxygen_drafting": {"roxy_full": 6, "roxy_tags": 6}, "format_propagation": {"fmt_one_line": 6, "fmt_aligned": 6}}.

Each case shows the edited window as a diff (prefix + region + suffix, before and after the gold edit),
the gold target, the verification record and the exact rendered prompt of the drop-in TRAIN row.

## 1. `10da597f03915ad01afc8e49` finish_block / short

- package `synth:shiftplan`, path `R/packing.R`
- generator: `{"bucket": "short", "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b19-graphs-or/shiftplan/R/packing.R", "signature_region": false, "style": "dense", "target_lines": 2, "unit": "binpack.summary", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 10, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 234, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -10,3 +10,5 @@
   lb <- binpack.lower.bound(size, capacity)
   attr(out, "lower.bound") <- lb
   attr(out, "gap") <- nrow(out) - lb
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
<filename>R/packing.R
binpack.summary <- function(size, bin, capacity) {
  load <- tapply(size, bin, sum)
  n.items <- tapply(size, bin, length)
  out <- data.frame(
    bin = as.integer(names(load)),
    n.items = as.integer(n.items),
    load = as.numeric(load),
    fill = as.numeric(load) / capacity
  )
  lb <- binpack.lower.bound(size, capacity)
  attr(out, "lower.bound") <- lb
  attr(out, "gap") <- nrow(out) - lb
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

## 2. `8325ac399d2be83d627441e6` finish_block / short

- package `synth:markovine`, path `R/seir.R`
- generator: `{"bucket": "short", "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b23-simulation/markovine/R/seir.R", "signature_region": false, "style": "sectioned", "target_lines": 2, "unit": "seir.initial", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 32, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 314, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -17,3 +17,5 @@
   if (S0 < 0) {
     stop("Initial compartments exceed the population size.", call. = FALSE)
   }
+  return(c(S = S0, E = E0, I = I0, R = R0.immune))
+}
```

**Target (gold region)**

```r
  return(c(S = S0, E = E0, I = I0, R = R0.immune))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/seir.R
# ---- diagnostics (static) --------------------
# typed statements: 2 of 3
# args not yet referenced: none
# unclosed braces in typed code: 1
# ----------------------------------------------
#' Initial state for an SEIR model
#'
#' @param N Total population size.
#' @param I0 Initial number of infectious individuals.
#' @param E0 Initial number of exposed individuals.
#' @param R0.immune Initial number of immune individuals.
#' @return A named numeric vector with elements `S`, `E`, `I` and `R`.
#' @family seir models
#' @export
seir.initial <- function(N, I0 = 1, E0 = 0, R0.immune = 0) {
  S0 <- N - I0 - E0 - R0.immune
  if (S0 < 0) {
    stop("Initial compartments exceed the population size.", call. = FALSE)
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

## 3. `24e0041eb94d10678359d144` finish_block / short

- package `synth:pagesieve`, path `tests/testthat/test-extract.R`
- generator: `{"bucket": "short", "novel_tokens": ["\"@data-id\"", "\"Invalid field spec\"", "\"a@b@c\"", "expect_error"], "novelty": 4, "rule": "fb_statement", "seed": "b18-web/pagesieve/tests/testthat/test-extract.R", "signature_region": false, "style": "sectioned", "target_lines": 3, "unit": "test_that", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 48, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 776, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -57,3 +57,6 @@
     parse_field_spec("a.title@href"),
     list(css = "a.title", attr = "href")
   )
+  expect_equal(parse_field_spec("@data-id"), list(css = NULL, attr = "data-id"))
+  expect_error(parse_field_spec("a@b@c"), "Invalid field spec")
+})
```

**Target (gold region)**

```r
  expect_equal(parse_field_spec("@data-id"), list(css = NULL, attr = "data-id"))
  expect_error(parse_field_spec("a@b@c"), "Invalid field spec")
})
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>tests/testthat/test-extract.R
test_that("extract_text collapses whitespace and handles no matches", {
  html <- listing_html()
  expect_equal(extract_text(html, "a.title"), c("First post", "Second post"))
  expect_identical(extract_text(html, "h1"), NA_character_)
  expect_identical(extract_text(html, "h1", missing = ""), "")
})

test_that("extract_attr returns attribute values per element", {
  html <- listing_html()
  expect_equal(extract_attr(html, "a.title", "href"), c("/p/1", "/p/2#top"))
  expect_identical(
    extract_attr(html, "a.title", "data-id"),
    c(NA_character_, NA_character_)
  )
  expect_identical(extract_attr(html, "h1", "href"), NA_character_)
})

test_that("extract_number strips currency and separators", {
  html <- listing_html()
  expect_equal(extract_number(html, ".price"), c(1200.5, NA, 15))
  expect_identical(extract_number(html, "h1"), NA_real_)
  euro <- rvest::minimal_html('<p class="v">1.234,5 EUR</p>')
  expect_equal(extract_number(euro, ".v", decimal_mark = ","), 1234.5)
})

test_that("extract_links resolves, deduplicates and filters links", {
  html <- listing_html()
  links <- extract_links(html, "https://blog.example.org/index.html")
  expect_true("https://blog.example.org/p/1" %in% links)
  expect_true("https://blog.example.org/p/2" %in% links)
  expect_false(any(grepl("^mailto:", links)))
  local <- extract_links(html, "https://blog.example.org/", same_host = TRUE)
  expect_false("https://other.example.net/x" %in% local)
  expect_length(local, 2)
})

test_that("extract_meta collects title and repeated meta tags", {
  html <- rvest::minimal_html(
    '
    <meta name="Description" content="A test page">
    <meta property="og:image" content="a.png">
    <meta property="og:image" content="b.png">
    <meta charset="utf-8">
  ',
    title = "Demo"
  )
  meta <- extract_meta(html)
  expect_equal(meta$title, "Demo")
  expect_equal(meta$description, "A test page")
  expect_equal(meta[["og:image"]], c("a.png", "b.png"))
  expect_null(meta$charset)
})

test_that("parse_field_spec splits selector and attribute", {
  expect_equal(parse_field_spec("a.title"), list(css = "a.title", attr = NULL))
  expect_equal(
    parse_field_spec("a.title@href"),
    list(css = "a.title", attr = "href")
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

## 4. `63c4faebf981743cd4b43f66` finish_block / short

- package `synth:cellarlink`, path `tests/testthat/test-sql_build.R`
- generator: `{"bucket": "short", "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b12-db/cellarlink/tests/testthat/test-sql_build.R", "signature_region": false, "style": "dense", "target_lines": 2, "unit": "test_that", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 28, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1015, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -85,3 +85,5 @@
     "sku"
   )
   out <- DBI::dbGetQuery(db, "SELECT sku, qty FROM stock ORDER BY sku")
+  expect_equal(out$qty, c(1L, 20L, 30L))
+})
```

**Target (gold region)**

```r
  expect_equal(out$qty, c(1L, 20L, 30L))
})
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>tests/testthat/test-sql_build.R
test_that("quote_cols quotes and joins column names", {
  sql <- quote_cols(con, c("id", "order date"))
  expect_s4_class(sql, "SQL")
  expect_equal(as.character(sql), '"id", "order date"')
  expect_error(quote_cols(con, character()), "at least one column")
})

test_that("quote_table splits schema-qualified names", {
  expect_equal(
    as.character(quote_table(con, "sales.orders")),
    '"sales"."orders"'
  )
  expect_equal(as.character(quote_table(con, "orders")), '"orders"')
})

test_that("build_select handles columns and filters", {
  sql <- build_select(con, "orders", cols = c("id", "total"))
  expect_equal(as.character(sql), 'SELECT "id", "total" FROM "orders"')
  sql <- build_select(con, "orders", where = list(status = "open"))
  expect_equal(
    as.character(sql),
    "SELECT * FROM \"orders\" WHERE \"status\" = 'open'"
  )
})

test_that("build_delete refuses to run without a filter", {
  expect_error(build_delete(con, "orders", list()), "without a WHERE")
  sql <- build_delete(con, "orders", list(id = 3L))
  expect_equal(as.character(sql), 'DELETE FROM "orders" WHERE "id" = 3')
})

test_that("build_update refuses to run without a filter", {
  expect_error(
    build_update(con, "orders", list(total = 10), list()),
    "without a WHERE"
  )
  sql <- build_update(con, "orders", list(total = 10), list(id = 3L))
  expect_equal(
    as.character(sql),
    'UPDATE "orders" SET "total" = 10 WHERE "id" = 3'
  )
})

test_that("where_clause covers scalars, vectors and missing values", {
  sql <- where_clause(con, list(a = 1L, b = c("x", "y"), c = NA))
  expect_equal(
    as.character(sql),
    "\"a\" = 1 AND \"b\" IN ('x', 'y') AND \"c\" IS NULL"
  )
  expect_error(where_clause(con, list(1, 2)), "must be named")
})

test_that("date_range_sql leaves open bounds out", {
  expect_equal(as.character(date_range_sql(con, "day")), "1 = 1")
  expect_equal(
    as.character(date_range_sql(con, "day", from = "2024-01-01")),
    "\"day\" >= '2024-01-01'"
  )
  expect_equal(
    as.character(date_range_sql(
      con,
      "day",
      from = "2024-01-01",
      to = "2024-02-01"
    )),
    "\"day\" >= '2024-01-01' AND \"day\" < '2024-02-01'"
  )
})

test_that("upsert_on_conflict updates existing rows in SQLite", {
  skip_if_not_installed("RSQLite")
  db <- connect_sqlite()
  on.exit(DBI::dbDisconnect(db), add = TRUE)
  DBI::dbExecute(db, "CREATE TABLE stock (sku TEXT PRIMARY KEY, qty INTEGER)")
  DBI::dbWriteTable(
    db,
    "stock",
    data.frame(sku = c("a", "b"), qty = c(1L, 2L)),
    append = TRUE
  )
  upsert_on_conflict(
    db,
    "stock",
    data.frame(sku = c("b", "c"), qty = c(20L, 30L)),
    "sku"
  )
  out <- DBI::dbGetQuery(db, "SELECT sku, qty FROM stock ORDER BY sku")
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

## 5. `3f386c5367a8a16e99b99660` finish_block / short

- package `synth:pagesieve`, path `R/extract.R`
- generator: `{"bucket": "short", "novel_tokens": ["html_attr", "nzchar", "values"], "novelty": 3, "rule": "fb_nested_close", "seed": "b18-web/pagesieve/R/extract.R", "signature_region": false, "style": "dense", "target_lines": 5, "unit": "extract_attr", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 44, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 239, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -12,3 +12,8 @@
   nodes <- rvest::html_elements(html, css)
   if (length(nodes) == 0) {
     return(missing)
+  }
+  values <- rvest::html_attr(nodes, attr)
+  values[is.na(values) | !nzchar(values)] <- missing
+  values
+}
```

**Target (gold region)**

```r
  }
  values <- rvest::html_attr(nodes, attr)
  values[is.na(values) | !nzchar(values)] <- missing
  values
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/extract.R
#' Extract an attribute from matching elements
#'
#' Selects elements with a CSS selector and returns the value of one of
#' their attributes.
#'
#' @inheritParams extract_text
#' @param attr Name of the attribute to read.
#' @return A character vector with one entry per matching element.
#' @family extractors
#' @export
extract_attr <- function(html, css, attr, missing = NA_character_) {
  nodes <- rvest::html_elements(html, css)
  if (length(nodes) == 0) {
    return(missing)
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

## 6. `7dbf8fbe7fd1725ad82def07` finish_block / short

- package `synth:itemwise`, path `R/norms.R`
- generator: `{"bucket": "short", "novel_tokens": ["0.5", "below"], "novelty": 2, "rule": "fb_statement", "seed": "b15-psych-trials/itemwise/R/norms.R", "signature_region": false, "style": "sectioned", "target_lines": 7, "unit": "percentile_rank", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 50, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 295, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -20,3 +20,10 @@
       if (is.na(value)) {
         return(NA_real_)
       }
+      below <- sum(reference < value)
+      equal <- sum(reference == value)
+      100 * (below + 0.5 * equal) / n
+    },
+    numeric(1)
+  )
+}
```

**Target (gold region)**

```r
      below <- sum(reference < value)
      equal <- sum(reference == value)
      100 * (below + 0.5 * equal) / n
    },
    numeric(1)
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/norms.R
#' Percentile ranks against a reference sample
#'
#' Uses the mid-point definition: the percentage of reference scores below a
#' value plus half the percentage equal to it.
#'
#' @param x A numeric vector of raw scale scores.
#' @param reference A numeric vector of raw scores from the norm sample.
#' @return A numeric vector of percentile ranks between 0 and 100.
#' @family norm conversions
#' @export
percentile_rank <- function(x, reference) {
  reference <- reference[!is.na(reference)]
  if (length(reference) == 0) {
    stop("`reference` has no non-missing values.", call. = FALSE)
  }
  n <- length(reference)
  vapply(
    x,
    function(value) {
      if (is.na(value)) {
        return(NA_real_)
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

## 7. `5a8437334bfd8b06451918bc` finish_block / short

- package `synth:cellwright`, path `R/walk.R`
- generator: `{"bucket": "short", "novel_tokens": ["runif"], "novelty": 1, "rule": "fb_nested_close", "seed": "b16-classic/cellwright/R/walk.R", "signature_region": false, "style": "dense", "target_lines": 6, "unit": "walk.first.passage", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 47, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 206, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -7,3 +7,9 @@
   while (pos != level) {
     if (n >= max.steps) {
       return(NA_integer_)
+    }
+    pos <- pos + if (runif(1) < p.up) 1L else -1L
+    n <- n + 1L
+  }
+  n
+}
```

**Target (gold region)**

```r
    }
    pos <- pos + if (runif(1) < p.up) 1L else -1L
    n <- n + 1L
  }
  n
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/walk.R
walk.first.passage <- function(level, p.up = 0.5, start = 0L, max.steps = 1e5) {
  if (level == start) {
    stop("'level' must differ from 'start'", call. = FALSE)
  }
  pos <- start
  n <- 0L
  while (pos != level) {
    if (n >= max.steps) {
      return(NA_integer_)
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

## 8. `bd1e259e42341c5bafadebb6` finish_block / short

- package `synth:millrace_catchment`, path `analysis/01_clean_stations.R`
- generator: `{"bucket": "short", "novel_tokens": ["9L", "bad"], "novelty": 2, "rule": "fb_nested_close", "seed": "b20-environment/millrace_catchment/analysis/01_clean_stations.R", "signature_region": false, "style": "dense", "target_lines": 8, "unit": "mask_flagged", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 59, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 164, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -3,3 +3,11 @@
     flag_col <- paste0(v, "_qc")
     if (!flag_col %in% names(df)) {
       next
+    }
+    bad <- !is.na(df[[flag_col]]) &
+      df[[flag_col]] > worst_ok &
+      df[[flag_col]] != 9L
+    df[[v]][bad] <- NA_real_
+  }
+  df
+}
```

**Target (gold region)**

```r
    }
    bad <- !is.na(df[[flag_col]]) &
      df[[flag_col]] > worst_ok &
      df[[flag_col]] != 9L
    df[[v]][bad] <- NA_real_
  }
  df
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>analysis/01_clean_stations.R
mask_flagged <- function(df, vars, worst_ok = 1L) {
  for (v in vars) {
    flag_col <- paste0(v, "_qc")
    if (!flag_col %in% names(df)) {
      next
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

## 9. `6d7d7f56f5081b58fb6020e1` finish_block / short

- package `synth:orbitnet`, path `R/centrality.R`
- generator: `{"bucket": "short", "novel_tokens": ["centrality_one", "igraph", "name", "tibble"], "novelty": 4, "rule": "fb_statement", "seed": "b19-graphs-or/orbitnet/R/centrality.R", "signature_region": false, "style": "dense", "target_lines": 6, "unit": "centrality_table", "variant": "outline"}`
- verification: `{"air_idempotent": true, "completion_tokens": 54, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 273, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -17,3 +17,9 @@
   measures = c("degree", "betweenness", "pagerank"),
   weighted = TRUE
 ) {
+  out <- tibble::tibble(name = igraph::V(g)$name)
+  for (m in measures) {
+    out[[m]] <- centrality_one(g, m, weighted = weighted)
+  }
+  out
+}
```

**Target (gold region)**

```r
  out <- tibble::tibble(name = igraph::V(g)$name)
  for (m in measures) {
    out[[m]] <- centrality_one(g, m, weighted = weighted)
  }
  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/centrality.R
# ---- outline --------------------------------
# fn: centrality_table(g, measures, weighted)
# plan: 3 top-level statements; 0 typed above the cursor
# docs: yes
# ----------------------------------------------
#' Centrality table for several measures
#'
#' @param g An `igraph` object.
#' @param measures Character vector of measure names accepted by
#'   [centrality_one()].
#' @param weighted Logical; use the `weight` edge attribute when present.
#' @return A tibble with a `name` column and one column per measure.
#' @export
#' @family centrality
centrality_table <- function(
  g,
  measures = c("degree", "betweenness", "pagerank"),
  weighted = TRUE
) {
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

## 10. `47477743212a2999a325ff6b` finish_block / short

- package `synth:allocadder`, path `R/interim.R`
- generator: `{"bucket": "short", "novel_tokens": ["exp", "log"], "novelty": 2, "rule": "fb_statement", "seed": "b15-psych-trials/allocadder/R/interim.R", "signature_region": false, "style": "dense", "target_lines": 2, "unit": "spendPocock", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 26, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 258, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -14,3 +14,5 @@
 #' @export
 spendPocock <- function(t, alpha = 0.05) {
   stopifnot(all(t > 0 & t <= 1))
+  alpha * log(1 + (exp(1) - 1) * t)
+}
```

**Target (gold region)**

```r
  alpha * log(1 + (exp(1) - 1) * t)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/interim.R
# ---- diagnostics (static) --------------------
# typed statements: 1 of 2
# args not yet referenced: alpha
# unclosed braces in typed code: 1
# ----------------------------------------------
#' Pocock type alpha spending
#'
#' Lan-DeMets spending function that spends alpha roughly evenly over looks.
#'
#' @param t Information fractions in (0, 1].
#' @param alpha Overall two-sided type I error rate.
#' @return Cumulative alpha spent at each information fraction.
#' @family spending functions
#' @export
spendPocock <- function(t, alpha = 0.05) {
  stopifnot(all(t > 0 & t <= 1))
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

## 11. `958bdf00b88f0534a0ea57c7` finish_block / short

- package `synth:gaugewise`, path `R/flow_duration.R`
- generator: `{"bucket": "short", "novel_tokens": ["50", "fdc_quantile", "percent", "unname"], "novelty": 4, "rule": "fb_statement", "seed": "b20-environment/gaugewise/R/flow_duration.R", "signature_region": false, "style": "dense", "target_lines": 2, "unit": "fdc_q50", "variant": "outline"}`
- verification: `{"air_idempotent": true, "completion_tokens": 22, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 228, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -12,3 +12,5 @@
 #' @family flow duration
 #' @export
 fdc_q50 <- function(flow) {
+  unname(fdc_quantile(flow, percent = 50))
+}
```

**Target (gold region)**

```r
  unname(fdc_quantile(flow, percent = 50))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/flow_duration.R
# ---- outline --------------------------------
# fn: fdc_q50(flow)
# plan: 1 top-level statements; 0 typed above the cursor
# docs: yes
# ----------------------------------------------
#' Flow at the 50 percent exceedance level
#'
#' Convenience wrapper for the median flow index Q50.
#'
#' @param flow A numeric vector of daily discharge values in cubic metres per second.
#' @return A single numeric discharge value.
#' @family flow duration
#' @export
fdc_q50 <- function(flow) {
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

## 12. `e9945cce90a19f42211ffab3` finish_block / short

- package `synth:shiftplan`, path `tests/testthat/test-packing.R`
- generator: `{"bucket": "short", "novel_tokens": ["binpack.best.fit"], "novelty": 1, "rule": "fb_statement", "seed": "b19-graphs-or/shiftplan/tests/testthat/test-packing.R", "signature_region": false, "style": "dense", "target_lines": 2, "unit": "test_that", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 31, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 148, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -1,2 +1,4 @@
 test_that("bin packers reject oversized items", {
   expect_error(binpack.first.fit(c(3, 12), capacity = 10), "larger than")
+  expect_error(binpack.best.fit(c(3, 12), capacity = 10), "larger than")
+})
```

**Target (gold region)**

```r
  expect_error(binpack.best.fit(c(3, 12), capacity = 10), "larger than")
})
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>tests/testthat/test-packing.R
test_that("bin packers reject oversized items", {
  expect_error(binpack.first.fit(c(3, 12), capacity = 10), "larger than")
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

## 13. `a39429cd49e39400564a2a0f` finish_block / short

- package `synth:pkgsmith`, path `R/templates.R`
- generator: `{"bucket": "short", "novel_tokens": ["\"'.\"", "\"Wrote '\""], "novelty": 2, "rule": "fb_statement", "seed": "b22-devtools/pkgsmith/R/templates.R", "signature_region": false, "style": "dense", "target_lines": 3, "unit": "use_template", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 24, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 396, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -34,3 +34,6 @@
     data
   )
   writeLines(lines, dest)
+  message("Wrote '", save_as, "'.")
+  invisible(dest)
+}
```

**Target (gold region)**

```r
  message("Wrote '", save_as, "'.")
  invisible(dest)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/templates.R
#' Write a rendered template into a package
#'
#' @param template File name of the template inside `inst/templates`.
#' @param save_as Destination path relative to the package root.
#' @param data A named list of values to substitute.
#' @param overwrite Whether an existing file may be replaced.
#' @param path Path to the package root.
#' @return The destination path, invisibly.
#' @export
use_template <- function(
  template,
  save_as,
  data = list(),
  overwrite = FALSE,
  path = "."
) {
  dest <- file.path(path, save_as)
  if (file.exists(dest) && !overwrite) {
    message("'", save_as, "' already exists; leaving it untouched.")
    return(invisible(dest))
  }
  dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
  pkg <- desc_get_field(
    "Package",
    default = basename(normalizePath(path)),
    path = path
  )
  data <- utils::modifyList(
    list(package = pkg, year = format(Sys.Date(), "%Y")),
    data
  )
  lines <- render_template(
    readLines(template_path(template), warn = FALSE),
    data
  )
  writeLines(lines, dest)
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

## 14. `7f4f09f0858343c0b1ccacef` finish_block / short

- package `synth:terraslab`, path `tests/testthat/test-tiles.R`
- generator: `{"bucket": "short", "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b13-geo/terraslab/tests/testthat/test-tiles.R", "signature_region": false, "style": "sectioned", "target_lines": 2, "unit": "test_that", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 30, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 378, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -18,3 +18,5 @@
 
 test_that("tileScheme() rejects non-positive tile sizes", {
   expect_error(tileScheme(makeRaster(), tileRows = 0), "must be positive")
+  expect_error(tileScheme(makeRaster(), tileCols = -1), "must be positive")
+})
```

**Target (gold region)**

```r
  expect_error(tileScheme(makeRaster(), tileCols = -1), "must be positive")
})
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>tests/testthat/test-tiles.R
test_that("tileScheme() covers every row and column exactly once", {
  r <- makeRaster()
  scheme <- tileScheme(r, tileRows = 4, tileCols = 5)
  expect_equal(nrow(scheme), 3 * 3)
  rows <- unique(scheme[, c("rowStart", "rowEnd")])
  cols <- unique(scheme[, c("colStart", "colEnd")])
  expect_equal(sum(rows$rowEnd - rows$rowStart + 1), 10)
  expect_equal(sum(cols$colEnd - cols$colStart + 1), 12)
  expect_false(anyDuplicated(scheme$tileId) > 0)
})

test_that("tileScheme() returns a single tile for small rasters", {
  scheme <- tileScheme(makeRaster(), tileRows = 100, tileCols = 100)
  expect_equal(nrow(scheme), 1)
  expect_equal(scheme$rowEnd, 10)
  expect_equal(scheme$colEnd, 12)
})

test_that("tileScheme() rejects non-positive tile sizes", {
  expect_error(tileScheme(makeRaster(), tileRows = 0), "must be positive")
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

## 15. `dd33845074a37daa618c9f0e` finish_block / short

- package `synth:tallyhook`, path `R/cache.R`
- generator: `{"bucket": "short", "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b06-infra-tests/tallyhook/R/cache.R", "signature_region": false, "style": "dense", "target_lines": 3, "unit": "cache_get", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 25, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 514, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -43,3 +43,6 @@
 cache_get <- function(key, default = NULL) {
   if (!cache_has(key)) {
     return(default)
+  }
+  get(key, envir = the_cache, inherits = FALSE)$value
+}
```

**Target (gold region)**

```r
  }
  get(key, envir = the_cache, inherits = FALSE)$value
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/cache.R
#' Build a cache key from arbitrary values
#'
#' Hashes the supplied values together with a namespace so that different
#' memoised functions never collide.
#'
#' @param namespace A string identifying the owner of the cache entry.
#' @param ... Values that determine the cached result.
#' @return A string key.
#' @keywords internal
cache_key <- function(namespace, ...) {
  check_string(namespace)
  values <- rlang::list2(...)
  paste0(namespace, "-", rlang::hash(values))
}

#' Check, read and write entries in the session cache
#'
#' The session cache is an environment that lives for the duration of the R
#' session. Entries expire after `ttl` seconds when a time-to-live is set.
#'
#' @param key A cache key, usually created by [cache_key()].
#' @return `cache_has()` returns `TRUE` or `FALSE`.
#' @rdname session-cache
#' @keywords internal
cache_has <- function(key) {
  if (!exists(key, envir = the_cache, inherits = FALSE)) {
    return(FALSE)
  }
  entry <- get(key, envir = the_cache, inherits = FALSE)
  if (is.null(entry$expires)) {
    return(TRUE)
  }
  expired <- Sys.time() > entry$expires
  if (expired) {
    rm(list = key, envir = the_cache)
  }
  !expired
}

#' @rdname session-cache
#' @param default The value returned when `key` is missing or expired.
#' @return `cache_get()` returns the cached value or `default`.
cache_get <- function(key, default = NULL) {
  if (!cache_has(key)) {
    return(default)
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

## 16. `4c4066b7b84b3626fa94346b` finish_block / short

- package `synth:aerocal`, path `R/aqi_breakpoints.R`
- generator: `{"bucket": "short", "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b20-environment/aerocal/R/aqi_breakpoints.R", "signature_region": false, "style": "dense", "target_lines": 4, "unit": "reviseBreakpoints", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 21, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 383, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -21,3 +21,7 @@
     if (tbl$concLo[i + 1] >= tbl$concHi[i + 1]) {
       stop("Revision would collapse the band above.", call. = FALSE)
     }
+  }
+  validateBreakpoints(tbl)
+  return(tbl)
+}
```

**Target (gold region)**

```r
  }
  validateBreakpoints(tbl)
  return(tbl)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/aqi_breakpoints.R
reviseBreakpoints <- function(tbl, category, concLo, concHi) {
  validateBreakpoints(tbl)
  i <- match(category, tbl$category)
  if (is.na(i)) {
    stop("Unknown category: ", category, call. = FALSE)
  }
  if (concHi <= concLo) {
    stop("`concHi` must exceed `concLo`.", call. = FALSE)
  }
  step <- 10^(-attr(tbl, "digits"))
  tbl$concLo[i] <- concLo
  tbl$concHi[i] <- concHi
  if (i > 1) {
    tbl$concHi[i - 1] <- concLo - step
    if (tbl$concHi[i - 1] <= tbl$concLo[i - 1]) {
      stop("Revision would collapse the band below.", call. = FALSE)
    }
  }
  if (i < nrow(tbl)) {
    tbl$concLo[i + 1] <- concHi + step
    if (tbl$concLo[i + 1] >= tbl$concHi[i + 1]) {
      stop("Revision would collapse the band above.", call. = FALSE)
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

## 17. `4f681527ca25b0a6f4432847` finish_block / short

- package `synth:foldwise`, path `R/splitters.R`
- generator: `{"bucket": "short", "novel_tokens": ["\"Need at least two groups for leave-one-group-out.\"", "\"logo\"", "call."], "novelty": 3, "rule": "fb_statement", "seed": "b11-ml/foldwise/R/splitters.R", "signature_region": false, "style": "dense", "target_lines": 8, "unit": "logo_splits", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 97, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1309, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -118,3 +118,11 @@
 #' @family splitters
 #' @export
 logo_splits <- function(groups) {
+  n <- length(groups)
+  unique_groups <- unique(groups)
+  if (length(unique_groups) < 2L) {
+    stop("Need at least two groups for leave-one-group-out.", call. = FALSE)
+  }
+  holdout <- lapply(unique_groups, function(g) which(groups == g))
+  return(new_splits(holdout, n, method = "logo"))
+}
```

**Target (gold region)**

```r
  n <- length(groups)
  unique_groups <- unique(groups)
  if (length(unique_groups) < 2L) {
    stop("Need at least two groups for leave-one-group-out.", call. = FALSE)
  }
  holdout <- lapply(unique_groups, function(g) which(groups == g))
  return(new_splits(holdout, n, method = "logo"))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/splitters.R
#' Build a split object from assessment indices
#'
#' Turns a list of held-out index vectors into a `fw_splits` object, where each
#' element stores both the analysis and assessment rows.
#'
#' @param holdout A list of integer vectors, one per fold, of assessment rows.
#' @param n Number of rows to split.
#' @param method A short label describing how the folds were made.
#' @return An object of class `fw_splits`.
#' @keywords internal
new_splits <- function(holdout, n, method) {
  all_rows <- seq_len(n)
  folds <- vector("list", length(holdout))
  for (i in seq_along(holdout)) {
    assess <- sort(holdout[[i]])
    folds[[i]] <- list(
      id = sprintf("Fold%02d", i),
      analysis = setdiff(all_rows, assess),
      assessment = assess
    )
  }
  return(structure(folds, n = n, method = method, class = "fw_splits"))
}

#' Plain k-fold cross-validation splits
#'
#' Randomly assigns each row to one of `k` folds of nearly equal size.
#'
#' @param n Number of rows to split.
#' @param k Number of folds.
#' @param seed Optional integer seed for reproducible assignment.
#' @return An object of class `fw_splits`.
#' @family splitters
#' @export
#' @examples
#' kfold_splits(20, k = 4, seed = 1)
kfold_splits <- function(n, k = 5, seed = NULL) {
  n <- check_split_size(n, k)
  if (!is.null(seed)) {
    set.seed(seed)
  }
  fold_id <- sample(rep_len(seq_len(k), n))
  holdout <- split(seq_len(n), fold_id)
  return(new_splits(unname(holdout), n, method = "kfold"))
}

#' Stratified k-fold cross-validation splits
#'
#' Assigns rows to folds separately within each level of `strata`, so that
#' class proportions are roughly preserved in every fold.
#'
#' @param strata A vector of class labels, one per row.
#' @param k Number of folds.
#' @param seed Optional integer seed for reproducible assignment.
#' @return An object of class `fw_splits`.
#' @family splitters
#' @export
stratified_splits <- function(strata, k = 5, seed = NULL) {
  n <- check_split_size(length(strata), k)
  if (anyNA(strata)) {
    stop("`strata` must not contain missing values.", call. = FALSE)
  }
  if (!is.null(seed)) {
    set.seed(seed)
  }
  fold_id <- integer(n)
  offset <- 0L
  for (level in unique(strata)) {
    rows <- which(strata == level)
    # rotate the starting fold so small classes do not all land in fold 1
    ids <- ((seq_along(rows) + offset - 1L) %% k) + 1L
    fold_id[rows] <- sample(ids)
    offset <- offset + length(rows)
  }
  holdout <- split(seq_len(n), fold_id)
  return(new_splits(unname(holdout), n, method = "stratified"))
}

#' Grouped k-fold cross-validation splits
#'
#' Keeps every row of a group in the same fold, so that no group appears in
#' both the analysis and assessment sets of a split.
#'
#' @param groups A vector of group identifiers, one per row.
#' @param k Number of folds.
#' @param seed Optional integer seed for reproducible assignment.
#' @return An object of class `fw_splits`.
#' @family splitters
#' @export
grouped_splits <- function(groups, k = 5, seed = NULL) {
  n <- length(groups)
  unique_groups <- unique(groups)
  check_split_size(length(unique_groups), k)
  if (!is.null(seed)) {
    set.seed(seed)
  }
  # greedy balancing: biggest groups first, each into the lightest fold
  sizes <- table(groups)[as.character(unique_groups)]
  order_idx <- order(-as.integer(sizes), sample(length(sizes)))
  load <- integer(k)
  group_fold <- integer(length(unique_groups))
  for (j in order_idx) {
    target <- which.min(load)
    group_fold[j] <- target
    load[target] <- load[target] + sizes[[j]]
  }
  fold_id <- group_fold[match(groups, unique_groups)]
  holdout <- split(seq_len(n), factor(fold_id, levels = seq_len(k)))
  return(new_splits(unname(holdout), n, method = "grouped"))
}

#' Leave-one-group-out splits
#'
#' Creates one split per distinct group, holding that group out entirely.
#'
#' @param groups A vector of group identifiers, one per row.
#' @return An object of class `fw_splits`.
#' @family splitters
#' @export
logo_splits <- function(groups) {
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

## 18. `4143714753e4cb92b254784e` finish_block / short

- package `synth:rendershed`, path `R/render.R`
- generator: `{"bucket": "short", "novel_tokens": [], "novelty": 0, "rule": "fb_nested_close", "seed": "b14-pipelines/rendershed/R/render.R", "signature_region": false, "style": "dense", "target_lines": 3, "unit": "renderMany", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 12, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 383, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -26,3 +26,6 @@
       paste(basename(out$input[failed]), collapse = ", "),
       call. = FALSE
     )
+  }
+  out
+}
```

**Target (gold region)**

```r
  }
  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/render.R
#' Render several documents in one go
#'
#' Renders each input in turn with [renderDocument()] and binds the
#' per-document rows together. Failures do not stop the remaining renders.
#'
#' @param inputs A character vector of source document paths.
#' @param outputDir Directory that receives the rendered file.
#' @param params A named list of document parameters, shared by all inputs.
#' @param ... Passed to [renderDocument()].
#' @return A data frame with one row per input.
#' @family renderers
#' @export
renderMany <- function(inputs, outputDir, params = list(), ...) {
  rows <- vector("list", length(inputs))
  for (i in seq_along(inputs)) {
    rows[[i]] <- renderDocument(inputs[[i]], outputDir, params = params, ...)
  }
  out <- do.call(rbind, rows)
  failed <- out$status != "ok"
  if (any(failed)) {
    warning(
      sum(failed),
      " of ",
      nrow(out),
      " document(s) failed to render: ",
      paste(basename(out$input[failed]), collapse = ", "),
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

## 19. `1fe887cb06a03465fb6b1ac4` finish_block / medium

- package `synth:rendershed`, path `R/render.R`
- generator: `{"bucket": "medium", "novel_tokens": ["\"Would remove: \"", "unlink"], "novelty": 2, "rule": "fb_statement", "seed": "b14-pipelines/rendershed/R/render.R", "signature_region": false, "style": "dense", "target_lines": 10, "unit": "cleanIntermediates", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 76, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1312, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -143,3 +143,13 @@
     `|`,
     lapply(patterns, grepl, x = basename(candidates))
   )]
+  if (length(hits) == 0L) {
+    return(invisible(character()))
+  }
+  if (dryRun) {
+    message("Would remove: ", paste(basename(hits), collapse = ", "))
+  } else {
+    unlink(hits, recursive = TRUE)
+  }
+  invisible(hits)
+}
```

**Target (gold region)**

```r
  if (length(hits) == 0L) {
    return(invisible(character()))
  }
  if (dryRun) {
    message("Would remove: ", paste(basename(hits), collapse = ", "))
  } else {
    unlink(hits, recursive = TRUE)
  }
  invisible(hits)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/render.R
#' Render a Quarto document
#'
#' Renders with the quarto CLI and moves the result into `outputDir`, since
#' quarto writes next to the source file by default.
#'
#' @param input Path to the source document.
#' @param outputDir Directory that receives the rendered file.
#' @param params A named list of document parameters.
#' @param format The output format, one of `"html"`, `"pdf"` or `"docx"`.
#' @param quiet Whether to suppress rendering progress output.
#' @return The path to the rendered file, invisibly.
#' @family renderers
#' @export
renderQuarto <- function(
  input,
  outputDir,
  params = list(),
  format = "html",
  quiet = TRUE
) {
  outputFile <- outputPathFor(input, outputDir, format, label = params$runLabel)
  quarto::quarto_render(
    input,
    output_format = format,
    output_file = basename(outputFile),
    execute_params = params,
    quiet = quiet
  )
  produced <- file.path(dirname(input), basename(outputFile))
  if (!file.rename(produced, outputFile)) {
    file.copy(produced, outputFile, overwrite = TRUE)
    unlink(produced)
  }
  invisible(outputFile)
}

#' Render a document with the matching engine
#'
#' Dispatches to [renderRmd()] or [renderQuarto()] based on the file
#' extension and records how long the render took.
#'
#' @param input Path to the source document.
#' @param outputDir Directory that receives the rendered file.
#' @param params A named list of document parameters.
#' @param format The output format, one of `"html"`, `"pdf"` or `"docx"`.
#' @param quiet Whether to suppress rendering progress output.
#' @return A one-row data frame describing the render.
#' @family renderers
#' @export
renderDocument <- function(
  input,
  outputDir,
  params = list(),
  format = "html",
  quiet = TRUE
) {
  if (!file.exists(input)) {
    stop("Input document not found: ", input, call. = FALSE)
  }
  dir.create(outputDir, recursive = TRUE, showWarnings = FALSE)
  engine <- detectEngine(input)
  started <- Sys.time()
  result <- tryCatch(
    {
      path <- switch(
        engine,
        quarto = renderQuarto(input, outputDir, params, format, quiet),
        rmarkdown = renderRmd(input, outputDir, params, format, quiet)
      )
      list(output = path, status = "ok", message = NA_character_)
    },
    error = function(e) {
      list(
        output = NA_character_,
        status = "error",
        message = conditionMessage(e)
      )
    }
  )
  data.frame(
    input = input,
    engine = engine,
    output = result$output,
    status = result$status,
    message = result$message,
    seconds = round(
      as.numeric(difftime(Sys.time(), started, units = "secs")),
      2
    ),
    stringsAsFactors = FALSE
  )
}

#' Render several documents in one go
#'
#' Renders each input in turn with [renderDocument()] and binds the
#' per-document rows together. Failures do not stop the remaining renders.
#'
#' @param inputs A character vector of source document paths.
#' @param outputDir Directory that receives the rendered file.
#' @param params A named list of document parameters, shared by all inputs.
#' @param ... Passed to [renderDocument()].
#' @return A data frame with one row per input.
#' @family renderers
#' @export
renderMany <- function(inputs, outputDir, params = list(), ...) {
  rows <- vector("list", length(inputs))
  for (i in seq_along(inputs)) {
    rows[[i]] <- renderDocument(inputs[[i]], outputDir, params = params, ...)
  }
  out <- do.call(rbind, rows)
  failed <- out$status != "ok"
  if (any(failed)) {
    warning(
      sum(failed),
      " of ",
      nrow(out),
      " document(s) failed to render: ",
      paste(basename(out$input[failed]), collapse = ", "),
      call. = FALSE
    )
  }
  out
}

#' Remove rendering leftovers
#'
#' Deletes knitr cache folders, `_files` directories and stray `.log` and
#' `.tex` files that renders leave next to the sources.
#'
#' @param dir Directory to clean.
#' @param patterns Regular expressions for files and folders to delete.
#' @param dryRun If `TRUE`, only report what would be removed.
#' @return The paths that were (or would be) removed, invisibly.
#' @export
cleanIntermediates <- function(
  dir,
  patterns = c("_cache$", "_files$", "\\.log$", "\\.tex$"),
  dryRun = FALSE
) {
  candidates <- list.files(dir, full.names = TRUE, include.dirs = TRUE)
  hits <- candidates[Reduce(
    `|`,
    lapply(patterns, grepl, x = basename(candidates))
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

## 20. `554f98f49937eb06d8283dbc` finish_block / medium

- package `synth:cellwright`, path `R/walk.R`
- generator: `{"bucket": "medium", "novel_tokens": ["p.win", "runif"], "novelty": 2, "rule": "fb_statement", "seed": "b16-classic/cellwright/R/walk.R", "signature_region": false, "style": "dense", "target_lines": 14, "unit": "gambler.ruin", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 98, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 352, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -20,3 +20,17 @@
     stop("'target' must exceed 'bankroll'", call. = FALSE)
   }
   rounds <- 0L
+  repeat {
+    bankroll <- bankroll + if (runif(1) < p.win) 1 else -1
+    rounds <- rounds + 1L
+    if (bankroll == 0) {
+      result <- "ruin"
+      break
+    }
+    if (bankroll == target) {
+      result <- "target"
+      break
+    }
+  }
+  list(result = result, rounds = rounds)
+}
```

**Target (gold region)**

```r
  repeat {
    bankroll <- bankroll + if (runif(1) < p.win) 1 else -1
    rounds <- rounds + 1L
    if (bankroll == 0) {
      result <- "ruin"
      break
    }
    if (bankroll == target) {
      result <- "target"
      break
    }
  }
  list(result = result, rounds = rounds)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/walk.R
# ---- diagnostics (static) --------------------
# typed statements: 4 of 6
# args not yet referenced: none
# unclosed braces in typed code: 1
# ----------------------------------------------
#' Play out the gambler's ruin
#'
#' A gambler bets one unit per round until the bankroll hits zero or the
#' target. Uses `repeat` so that the loop condition is checked after a bet.
#'
#' @param bankroll Starting number of units.
#' @param target Bankroll at which the gambler walks away.
#' @param p.win Probability of winning a single bet.
#' @return A list with `result` ("ruin" or "target") and `rounds`.
#' @export
gambler.ruin <- function(bankroll = 10, target = 20, p.win = 0.5) {
  check.count(bankroll, "bankroll")
  check.probability(p.win, "p.win")
  if (target <= bankroll) {
    stop("'target' must exceed 'bankroll'", call. = FALSE)
  }
  rounds <- 0L
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

## 21. `37727de7df9500380fdf4245` finish_block / medium

- package `synth:pagesieve`, path `R/tables.R`
- generator: `{"bucket": "medium", "novel_tokens": ["clean_names", "cols", "tbl"], "novelty": 3, "rule": "fb_statement", "seed": "b18-web/pagesieve/R/tables.R", "signature_region": false, "style": "dense", "target_lines": 10, "unit": "pick_table", "variant": "outline"}`
- verification: `{"air_idempotent": true, "completion_tokens": 72, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 256, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -13,3 +13,13 @@
 #' @return A data frame, or `NULL` when no table qualifies.
 #' @export
 pick_table <- function(tables, must_have) {
+  must_have <- clean_names(must_have)
+  for (tbl in tables) {
+    cols <- clean_names(names(tbl))
+    if (all(must_have %in% cols)) {
+      names(tbl) <- cols
+      return(tbl)
+    }
+  }
+  NULL
+}
```

**Target (gold region)**

```r
  must_have <- clean_names(must_have)
  for (tbl in tables) {
    cols <- clean_names(names(tbl))
    if (all(must_have %in% cols)) {
      names(tbl) <- cols
      return(tbl)
    }
  }
  NULL
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/tables.R
# ---- outline --------------------------------
# fn: pick_table(tables, must_have)
# plan: 3 top-level statements; 0 typed above the cursor
# docs: yes
# ----------------------------------------------
#' Pick the table that contains the expected columns
#'
#' Returns the first table whose cleaned column names include every name in
#' `must_have`.
#'
#' @param tables A list of data frames, as returned by [scrape_tables()].
#' @param must_have Character vector of required column names.
#' @return A data frame, or `NULL` when no table qualifies.
#' @export
pick_table <- function(tables, must_have) {
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

## 22. `5b053134d83e6ab95f2c0c58` finish_block / medium

- package `synth:herdwalk`, path `R/interventions.R`
- generator: `{"bucket": "medium", "novel_tokens": ["day", "outbreak_duration", "size"], "novelty": 3, "rule": "fb_statement", "seed": "b23-simulation/herdwalk/R/interventions.R", "signature_region": false, "style": "dense", "target_lines": 9, "unit": "compare_scenarios", "variant": "outline"}`
- verification: `{"air_idempotent": true, "completion_tokens": 79, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 560, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -46,3 +46,12 @@
       peak <- peak_infections(run$counts)
       row <- row + 1L
       result$scenario[row] <- name
+      result$rep[row] <- r
+      result$peak_day[row] <- peak$day
+      result$peak_size[row] <- peak$size
+      result$attack_rate[row] <- attack_rate(run$counts)
+      result$duration[row] <- outbreak_duration(run$counts)
+    }
+  }
+  return(result)
+}
```

**Target (gold region)**

```r
      result$rep[row] <- r
      result$peak_day[row] <- peak$day
      result$peak_size[row] <- peak$size
      result$attack_rate[row] <- attack_rate(run$counts)
      result$duration[row] <- outbreak_duration(run$counts)
    }
  }
  return(result)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/interventions.R
# ---- outline --------------------------------
# fn: compare_scenarios(pop, scenarios, reps, days, params)
# plan: 6 top-level statements; 4 typed above the cursor
# docs: yes
# ----------------------------------------------
#' Compare intervention scenarios
#'
#' Runs the same starting population under several intervention scenarios
#' and collects the outbreak metrics for each replicate.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param scenarios A named list of scenarios; each element is a list with
#'   `type` and further arguments for [apply_intervention()].
#' @param reps Number of replicate runs per scenario.
#' @param days Maximum number of days to simulate.
#' @param params A named list of model parameters, see [run_herd()].
#' @return A data frame with one row per scenario and replicate.
#' @family interventions
#' @export
compare_scenarios <- function(
  pop,
  scenarios,
  reps = 10,
  days = 150,
  params = list()
) {
  if (is.null(names(scenarios)) || any(names(scenarios) == "")) {
    stop("`scenarios` must be a fully named list.", call. = FALSE)
  }
  n_rows <- length(scenarios) * reps
  result <- data.frame(
    scenario = character(n_rows),
    rep = integer(n_rows),
    peak_day = integer(n_rows),
    peak_size = integer(n_rows),
    attack_rate = numeric(n_rows),
    duration = integer(n_rows),
    stringsAsFactors = FALSE
  )
  row <- 0L
  for (name in names(scenarios)) {
    spec <- scenarios[[name]]
    for (r in seq_len(reps)) {
      start <- do.call(apply_intervention, c(list(pop), spec))
      run <- run_herd(start, days = days, params = params)
      peak <- peak_infections(run$counts)
      row <- row + 1L
      result$scenario[row] <- name
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

## 23. `a99634dfd8314da74388d7a4` finish_block / medium

- package `synth:rangeweft`, path `R/overlaps.R`
- generator: `{"bucket": "medium", "novel_tokens": ["\"queryHits\"", "\"subjectHits\"", "cbind", "dimnames", "queryHits", "rbind", "subjectHits"], "novelty": 7, "rule": "fb_nested_close", "seed": "b17-bioc/rangeweft/R/overlaps.R", "signature_region": false, "style": "dense", "target_lines": 16, "unit": ".overlapsOnSeq", "variant": "diag"}`
- verification: `{"air_idempotent": true, "completion_tokens": 112, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 520, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -46,3 +46,19 @@
         .pairOverlaps(qs, qe, subject@start[s], subject@end[s], type, maxgap)
       ) {
         found <- c(found, s)
+      }
+    }
+    if (length(found) > 0L) {
+      hits[[k]] <- cbind(queryHits = q, subjectHits = sort(found))
+    }
+  }
+  out <- do.call(rbind, hits)
+  if (is.null(out)) {
+    out <- matrix(
+      integer(),
+      ncol = 2L,
+      dimnames = list(NULL, c("queryHits", "subjectHits"))
+    )
+  }
+  return(out)
+}
```

**Target (gold region)**

```r
      }
    }
    if (length(found) > 0L) {
      hits[[k]] <- cbind(queryHits = q, subjectHits = sort(found))
    }
  }
  out <- do.call(rbind, hits)
  if (is.null(out)) {
    out <- matrix(
      integer(),
      ncol = 2L,
      dimnames = list(NULL, c("queryHits", "subjectHits"))
    )
  }
  return(out)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/overlaps.R
# ---- diagnostics (static) --------------------
# typed statements: 3 of 7
# args not yet referenced: none
# unclosed braces in typed code: 4
# ----------------------------------------------
#' Find overlaps between two interval sets on one sequence
#'
#' Sweeps over subject intervals sorted by start to collect hits for every
#' query interval on a single sequence.
#'
#' @param qIdx,sIdx Integer indices of query and subject intervals on the
#'   sequence.
#' @param query,subject `SpanSet` objects.
#' @param type Overlap rule.
#' @param maxgap Largest gap between intervals still counted as overlap.
#' @param ignoreStrand Should strand be ignored?
#' @return A two-column integer matrix of query and subject indices.
#' @noRd
.overlapsOnSeq <- function(
  qIdx,
  sIdx,
  query,
  subject,
  type,
  maxgap,
  ignoreStrand
) {
  sIdx <- sIdx[order(subject@start[sIdx])]
  sStart <- subject@start[sIdx]
  hits <- vector("list", length(qIdx))
  for (k in seq_along(qIdx)) {
    q <- qIdx[k]
    qs <- query@start[q]
    qe <- query@end[q]
    # subject intervals starting after the query end cannot overlap
    upper <- findInterval(qe + maxgap + 1L, sStart)
    found <- integer()
    for (m in seq_len(upper)) {
      s <- sIdx[m]
      if (
        !ignoreStrand && !.strandCompatible(query@strand[q], subject@strand[s])
      ) {
        next
      }
      if (
        .pairOverlaps(qs, qe, subject@start[s], subject@end[s], type, maxgap)
      ) {
        found <- c(found, s)
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

## 24. `e0ca901126f83e4a35aecba6` finish_block / medium

- package `synth:surveyscrub`, path `R/validate.R`
- generator: `{"bucket": "medium", "novel_tokens": ["\", \"", "\"Failed checks: \"", "bind_rows", "call.", "collapse", "dplyr", "report"], "novelty": 7, "rule": "fb_nested_close", "seed": "b01-tidyverse/surveyscrub/R/validate.R", "signature_region": false, "style": "sectioned", "target_lines": 10, "unit": "run_checks", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 78, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 358, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,3 +28,13 @@
           n_failed = NA_integer_,
           detail = conditionMessage(w)
         )
+      }
+    )
+  })
+  report <- dplyr::bind_rows(results)
+  if (stop_on_fail && !all(report$passed)) {
+    failed <- report$check[!report$passed]
+    stop("Failed checks: ", paste(failed, collapse = ", "), call. = FALSE)
+  }
+  report
+}
```

**Target (gold region)**

```r
      }
    )
  })
  report <- dplyr::bind_rows(results)
  if (stop_on_fail && !all(report$passed)) {
    failed <- report$check[!report$passed]
    stop("Failed checks: ", paste(failed, collapse = ", "), call. = FALSE)
  }
  report
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/validate.R
#' Run a list of checks and combine the results
#'
#' Each element of `checks` is a function taking the data and returning a
#' check result; errors inside a check are captured as failed rows rather
#' than aborting the whole run.
#'
#' @param data A data frame of survey responses.
#' @param checks A named list of functions, each returning a check result.
#' @param stop_on_fail If `TRUE`, raise an error when any check fails.
#' @return A tibble with one row per check.
#' @export
run_checks <- function(data, checks, stop_on_fail = FALSE) {
  results <- purrr::imap(checks, function(fn, nm) {
    tryCatch(
      fn(data),
      error = function(e) {
        new_check_result(
          nm,
          passed = FALSE,
          n_failed = NA_integer_,
          detail = conditionMessage(e)
        )
      },
      warning = function(w) {
        new_check_result(
          nm,
          passed = FALSE,
          n_failed = NA_integer_,
          detail = conditionMessage(w)
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

## 25. `2149e53afce05ba9d73634d7` finish_block / medium

- package `synth:numquill`, path `R/survival.R`
- generator: `{"bucket": "medium", "novel_tokens": ["\"nq.km\"", "events", "table", "upper"], "novelty": 4, "rule": "fb_statement", "seed": "b02-base-stats/numquill/R/survival.R", "signature_region": false, "style": "dense", "target_lines": 11, "unit": "kaplan.meier", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 67, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 633, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -35,3 +35,14 @@
   z <- stats::qnorm(1 - (1 - conf.level) / 2)
   se.ll <- sqrt(tab$var.log) / abs(log(tab$surv))
   tab$lower <- tab$surv^exp(z * se.ll)
+  tab$upper <- tab$surv^exp(-z * se.ll)
+  structure(
+    list(
+      table = tab,
+      n = length(time),
+      events = sum(status),
+      conf.level = conf.level
+    ),
+    class = "nq.km"
+  )
+}
```

**Target (gold region)**

```r
  tab$upper <- tab$surv^exp(-z * se.ll)
  structure(
    list(
      table = tab,
      n = length(time),
      events = sum(status),
      conf.level = conf.level
    ),
    class = "nq.km"
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/survival.R
#' Tabulate events and numbers at risk
#'
#' @param time Numeric vector of follow-up times.
#' @param status Event indicator, `1` for an event and `0` for censoring.
#' @return A data frame with columns `time`, `n.risk`, `n.event` and
#'   `n.censor`, one row per distinct time.
#' @keywords internal
risk.table <- function(time, status) {
  if (length(time) != length(status)) {
    stop("`time` and `status` must have the same length.", call. = FALSE)
  }
  if (!all(status %in% c(0, 1))) {
    stop("`status` must be coded 0/1.", call. = FALSE)
  }
  ut <- sort(unique(time))
  n.event <- vapply(ut, function(t) sum(time == t & status == 1), numeric(1))
  n.censor <- vapply(ut, function(t) sum(time == t & status == 0), numeric(1))
  n.risk <- vapply(ut, function(t) sum(time >= t), numeric(1))
  data.frame(time = ut, n.risk = n.risk, n.event = n.event, n.censor = n.censor)
}

#' Kaplan-Meier estimate of the survival function
#'
#' @param time Numeric vector of follow-up times.
#' @param status Event indicator, `1` for an event and `0` for censoring.
#' @param conf.level Confidence level for the log-log pointwise interval.
#' @return An object of class `nq.km`.
#' @family survival estimators
#' @export
kaplan.meier <- function(time, status, conf.level = 0.95) {
  tab <- risk.table(time, status)
  tab <- tab[tab$n.event > 0, , drop = FALSE]
  tab$surv <- cumprod(1 - tab$n.event / tab$n.risk)
  tab$var.log <- cumsum(tab$n.event / (tab$n.risk * (tab$n.risk - tab$n.event)))
  z <- stats::qnorm(1 - (1 - conf.level) / 2)
  se.ll <- sqrt(tab$var.log) / abs(log(tab$surv))
  tab$lower <- tab$surv^exp(z * se.ll)
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

## 26. `2246c868f9f498b7c7388c92` finish_block / medium

- package `synth:geodrift`, path `R/coords.R`
- generator: `{"bucket": "medium", "novel_tokens": ["\"`lat` must lie between -90 and 90.\"", "\"`lon` and `lat` must be numeric.\"", "\"`lon` and `lat` must have the same length.\"", "\"`lon` must lie between -180 and 180.\"", "180", "90", "call.", "na.rm"], "novelty": 8, "rule": "fb_statement", "seed": "b08-rcpp-science/geodrift/R/coords.R", "signature_region": false, "style": "dense", "target_lines": 14, "unit": "validateLonLat", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 171, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 119, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -1 +1,15 @@
 validateLonLat <- function(lon, lat) {
+  if (!is.numeric(lon) || !is.numeric(lat)) {
+    stop("`lon` and `lat` must be numeric.", call. = FALSE)
+  }
+  if (length(lon) != length(lat)) {
+    stop("`lon` and `lat` must have the same length.", call. = FALSE)
+  }
+  if (any(abs(lat) > 90, na.rm = TRUE)) {
+    stop("`lat` must lie between -90 and 90.", call. = FALSE)
+  }
+  if (any(abs(lon) > 180, na.rm = TRUE)) {
+    stop("`lon` must lie between -180 and 180.", call. = FALSE)
+  }
+  invisible(TRUE)
+}
```

**Target (gold region)**

```r
  if (!is.numeric(lon) || !is.numeric(lat)) {
    stop("`lon` and `lat` must be numeric.", call. = FALSE)
  }
  if (length(lon) != length(lat)) {
    stop("`lon` and `lat` must have the same length.", call. = FALSE)
  }
  if (any(abs(lat) > 90, na.rm = TRUE)) {
    stop("`lat` must lie between -90 and 90.", call. = FALSE)
  }
  if (any(abs(lon) > 180, na.rm = TRUE)) {
    stop("`lon` must lie between -180 and 180.", call. = FALSE)
  }
  invisible(TRUE)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/coords.R
validateLonLat <- function(lon, lat) {
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

## 27. `ae6beceda74a347162321849` finish_block / medium

- package `synth:apiclerk`, path `R/config.R`
- generator: `{"bucket": "medium", "novel_tokens": ["\"* \"", "\"Invalid configuration:\\n\"", "\"\\n\"", "call.", "collapse"], "novelty": 5, "rule": "fb_statement", "seed": "b03-datatable-io/apiclerk/R/config.R", "signature_region": false, "style": "dense", "target_lines": 9, "unit": "validate_config", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 56, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 347, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -19,3 +19,12 @@
   if (is.na(config$cache_ttl) || config$cache_ttl < 0) {
     problems <- c(problems, "`cache_ttl` must be zero or positive")
   }
+  if (length(problems) > 0L) {
+    stop(
+      "Invalid configuration:\n",
+      paste0("* ", problems, collapse = "\n"),
+      call. = FALSE
+    )
+  }
+  config
+}
```

**Target (gold region)**

```r
  if (length(problems) > 0L) {
    stop(
      "Invalid configuration:\n",
      paste0("* ", problems, collapse = "\n"),
      call. = FALSE
    )
  }
  config
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/config.R
#' Validate a configuration list
#'
#' Collects every problem before failing so users can fix them in one go.
#'
#' @param config A configuration list.
#' @return `config`, unchanged, if it is valid.
#' @keywords internal
validate_config <- function(config) {
  problems <- character()
  if (!grepl("^https?://", config$base_url)) {
    problems <- c(problems, "`base_url` must start with http:// or https://")
  }
  if (is.na(config$timeout) || config$timeout <= 0) {
    problems <- c(problems, "`timeout` must be a positive number")
  }
  if (is.na(config$max_tries) || config$max_tries < 1L) {
    problems <- c(problems, "`max_tries` must be at least 1")
  }
  if (is.na(config$cache_ttl) || config$cache_ttl < 0) {
    problems <- c(problems, "`cache_ttl` must be zero or positive")
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

## 28. `c136b094797b32e046563674` finish_block / medium

- package `synth:assayshelf`, path `R/show-methods.R`
- generator: `{"bucket": "medium", "novel_tokens": ["\"colInfo\"", "\"meta(%d): %s\\n\"", "meta"], "novelty": 3, "rule": "fb_statement", "seed": "b17-bioc/assayshelf/R/show-methods.R", "signature_region": false, "style": "dense", "target_lines": 11, "unit": "setMethod", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 100, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 811, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -71,3 +71,14 @@
   ))
   cat(.format_row_line(object), "\n", sep = "")
   cat(.format_info_line(object@rowInfo, "rowInfo"), "\n", sep = "")
+  cat(.format_col_line(object), "\n", sep = "")
+  cat(.format_info_line(object@colInfo, "colInfo"), "\n", sep = "")
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
  cat(.format_col_line(object), "\n", sep = "")
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
#' Abbreviate a vector of names for display
#'
#' Shows the first and last few names separated by an ellipsis when the
#' vector is longer than `n_head + n_tail`.
#'
#' @param ids A character vector of names.
#' @param n_head Number of leading names to show.
#' @param n_tail Number of trailing names to show.
#' @return A single string.
#' @noRd
.abbrev_names <- function(ids, n_head = 2L, n_tail = 2L) {
  if (is.null(ids) || length(ids) == 0L) {
    return("(none)")
  }
  if (length(ids) <= n_head + n_tail) {
    return(paste(ids, collapse = " "))
  }
  paste(
    c(utils::head(ids, n_head), "...", utils::tail(ids, n_tail)),
    collapse = " "
  )
}

#' Format the feature line of the show output
#'
#' @param object An `AssayShelf` object.
#' @return A single string.
#' @noRd
.format_row_line <- function(object) {
  ids <- rownames(object@rowInfo)
  sprintf("rownames(%d): %s", length(ids), .abbrev_names(ids))
}

#' Format the sample line of the show output
#'
#' @param object An `AssayShelf` object.
#' @return A single string.
#' @noRd
.format_col_line <- function(object) {
  ids <- rownames(object@colInfo)
  sprintf("colnames(%d): %s", length(ids), .abbrev_names(ids))
}

#' Format an annotation-column line of the show output
#'
#' @param info A data frame of annotations.
#' @param label Prefix for the line, e.g. `"rowInfo"`.
#' @return A single string.
#' @noRd
.format_info_line <- function(info, label) {
  cols <- names(info)
  sprintf("%s names(%d): %s", label, length(cols), .abbrev_names(cols, 3L, 1L))
}

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
  cat(.format_row_line(object), "\n", sep = "")
  cat(.format_info_line(object@rowInfo, "rowInfo"), "\n", sep = "")
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

## 29. `3b31ff56fd9b1aa3d0ac28ad` finish_block / long

- package `synth:gaugewise`, path `R/gap_fill.R`
- generator: `{"bucket": "long", "novel_tokens": ["\"observed\"", "anyNA", "climatology", "fill_climatology", "fill_linear", "fill_neighbour", "ifelse", "linear", "several.ok", "switch"], "novelty": 10, "rule": "fb_statement", "seed": "b20-environment/gaugewise/R/gap_fill.R", "signature_region": false, "style": "dense", "target_lines": 30, "unit": "fill_cascade", "variant": "plain"}`
- verification: `{"air_idempotent": true, "completion_tokens": 193, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 347, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -20,3 +20,33 @@
   methods = c("linear", "neighbour", "climatology"),
   max_gap = 3L
 ) {
+  methods <- match.arg(methods, several.ok = TRUE)
+  value <- x
+  source <- ifelse(is.na(x), NA_character_, "observed")
+  for (m in methods) {
+    if (!anyNA(value)) {
+      break
+    }
+    step <- switch(
+      m,
+      linear = fill_linear(value, max_gap = max_gap),
+      neighbour = if (is.null(neighbour)) {
+        NULL
+      } else {
+        fill_neighbour(value, neighbour)
+      },
+      climatology = fill_climatology(value, date)
+    )
+    if (is.null(step)) {
+      next
+    }
+    value <- step$value
+    source[step$filled] <- m
+  }
+  return(data.frame(
+    date = date,
+    value = value,
+    original = x,
+    source = source
+  ))
+}
```

**Target (gold region)**

```r
  methods <- match.arg(methods, several.ok = TRUE)
  value <- x
  source <- ifelse(is.na(x), NA_character_, "observed")
  for (m in methods) {
    if (!anyNA(value)) {
      break
    }
    step <- switch(
      m,
      linear = fill_linear(value, max_gap = max_gap),
      neighbour = if (is.null(neighbour)) {
        NULL
      } else {
        fill_neighbour(value, neighbour)
      },
      climatology = fill_climatology(value, date)
    )
    if (is.null(step)) {
      next
    }
    value <- step$value
    source[step$filled] <- m
  }
  return(data.frame(
    date = date,
    value = value,
    original = x,
    source = source
  ))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/gap_fill.R
#' Fill gaps with a cascade of methods
#'
#' Applies the requested fillers in order, each one only touching values that
#' are still missing after the previous step, and records which method filled
#' each value.
#'
#' @param x A numeric vector of observations at a regular time step.
#' @param date A vector of class `Date` matching `x`.
#' @param neighbour Optional numeric vector from a nearby station, aligned
#'   with `x`.
#' @param methods Character vector of fillers to apply, in order; any of
#'   `"linear"`, `"neighbour"` and `"climatology"`.
#' @param max_gap Longest gap, in time steps, that will be filled.
#' @return A data frame with columns `date`, `value`, `original` and `source`.
#' @export
fill_cascade <- function(
  x,
  date,
  neighbour = NULL,
  methods = c("linear", "neighbour", "climatology"),
  max_gap = 3L
) {
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

## 30. `5837164a22fdb17550ab2b94` finish_block / long

- package `synth:graphwalk`, path `R/shortest_path.R`
- generator: `{"bucket": "long", "novel_tokens": ["\"Negative cycle detected; distances are not well defined.\"", "\"Weight matrix must be square.\"", "better", "diag", "pmin", "via"], "novelty": 6, "rule": "fb_statement", "seed": "b19-graphs-or/graphwalk/R/shortest_path.R", "signature_region": false, "style": "dense", "target_lines": 25, "unit": "floyd_warshall", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 190, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1139, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -122,3 +122,28 @@
 #' @export
 #' @family shortest-paths
 floyd_warshall <- function(m) {
+  if (nrow(m) != ncol(m)) {
+    stop("Weight matrix must be square.", call. = FALSE)
+  }
+  d <- m
+  d[is.na(d)] <- Inf
+  diag(d) <- pmin(diag(d), 0)
+  n <- nrow(d)
+  for (k in seq_len(n)) {
+    for (i in seq_len(n)) {
+      if (is.infinite(d[i, k])) {
+        next
+      }
+      via <- d[i, k] + d[k, ]
+      better <- via < d[i, ]
+      d[i, better] <- via[better]
+    }
+  }
+  if (any(diag(d) < 0)) {
+    warning(
+      "Negative cycle detected; distances are not well defined.",
+      call. = FALSE
+    )
+  }
+  return(d)
+}
```

**Target (gold region)**

```r
  if (nrow(m) != ncol(m)) {
    stop("Weight matrix must be square.", call. = FALSE)
  }
  d <- m
  d[is.na(d)] <- Inf
  diag(d) <- pmin(diag(d), 0)
  n <- nrow(d)
  for (k in seq_len(n)) {
    for (i in seq_len(n)) {
      if (is.infinite(d[i, k])) {
        next
      }
      via <- d[i, k] + d[k, ]
      better <- via < d[i, ]
      d[i, better] <- via[better]
    }
  }
  if (any(diag(d) < 0)) {
    warning(
      "Negative cycle detected; distances are not well defined.",
      call. = FALSE
    )
  }
  return(d)
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/shortest_path.R
#' Single-source shortest paths with Dijkstra's algorithm
#'
#' Uses a simple linear scan instead of a heap, which is fine for the small
#' graphs this package targets (a few thousand vertices).
#'
#' @param wadj A weighted adjacency list as returned by
#'   [edges_to_weighted_adjlist()].
#' @param source Name of the starting vertex.
#' @return A list with named numeric `dist` and named character `prev`.
#' @export
#' @family shortest-paths
dijkstra <- function(wadj, source) {
  if (!source %in% names(wadj)) {
    stop("Unknown source vertex: ", source, call. = FALSE)
  }
  v <- names(wadj)
  dist <- setNames(rep(Inf, length(v)), v)
  prev <- setNames(rep(NA_character_, length(v)), v)
  done <- setNames(rep(FALSE, length(v)), v)
  dist[source] <- 0
  repeat {
    open <- v[!done]
    if (length(open) == 0) {
      break
    }
    u <- open[which.min(dist[open])]
    if (is.infinite(dist[u])) {
      break
    }
    done[u] <- TRUE
    nb <- wadj[[u]]
    for (w in names(nb)) {
      alt <- dist[u] + nb[[w]]
      if (alt < dist[w]) {
        dist[w] <- alt
        prev[w] <- u
      }
    }
  }
  return(list(dist = dist, prev = prev))
}

#' Single-source shortest paths with Bellman-Ford
#'
#' Handles negative edge weights and reports negative cycles reachable from
#' the source.
#'
#' @param wadj A weighted adjacency list as returned by
#'   [edges_to_weighted_adjlist()].
#' @param source Name of the starting vertex.
#' @return A list with named numeric `dist` and named character `prev`.
#' @export
#' @family shortest-paths
bellman_ford <- function(wadj, source) {
  if (!source %in% names(wadj)) {
    stop("Unknown source vertex: ", source, call. = FALSE)
  }
  v <- names(wadj)
  dist <- setNames(rep(Inf, length(v)), v)
  prev <- setNames(rep(NA_character_, length(v)), v)
  dist[source] <- 0
  for (iter in seq_len(length(v) - 1L)) {
    changed <- FALSE
    for (u in v) {
      if (is.infinite(dist[u])) {
        next
      }
      nb <- wadj[[u]]
      for (w in names(nb)) {
        alt <- dist[u] + nb[[w]]
        if (alt < dist[w]) {
          dist[w] <- alt
          prev[w] <- u
          changed <- TRUE
        }
      }
    }
    if (!changed) {
      break
    }
  }
  for (u in v[is.finite(dist)]) {
    nb <- wadj[[u]]
    for (w in names(nb)) {
      if (dist[u] + nb[[w]] < dist[w]) {
        stop(
          "Graph contains a negative cycle reachable from ",
          source,
          call. = FALSE
        )
      }
    }
  }
  return(list(dist = dist, prev = prev))
}

#' Extract a shortest path from a predecessor vector
#'
#' @param sp A list as returned by [dijkstra()] or [bellman_ford()].
#' @param target Name of the destination vertex.
#' @return A character vector of vertices from source to `target`, or
#'   `character(0)` if `target` is unreachable.
#' @export
#' @family shortest-paths
shortest_path_to <- function(sp, target) {
  if (is.infinite(sp$dist[[target]])) {
    return(character(0))
  }
  path <- target
  while (!is.na(sp$prev[[path[1]]])) {
    path <- c(sp$prev[[path[1]]], path)
  }
  return(path)
}

#' All-pairs shortest distances with Floyd-Warshall
#'
#' @param m A square numeric matrix of edge weights, with `Inf` (or `NA`) for
#'   missing edges.
#' @return A square numeric matrix of shortest distances with the same
#'   dimnames as `m`.
#' @export
#' @family shortest-paths
floyd_warshall <- function(m) {
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

## 31. `0dbe9a49124ae7ddea65ae01` finish_block / long

- package `synth:aerocal`, path `R/degree_days.R`
- generator: `{"bucket": "long", "novel_tokens": [], "novelty": 0, "rule": "fb_statement", "seed": "b20-environment/aerocal/R/degree_days.R", "signature_region": false, "style": "dense", "target_lines": 21, "unit": "ddTriangle", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 179, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 708, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -61,3 +61,24 @@
   checkTemps(tmin, tmax)
   out <- numeric(length(tmin))
   for (i in seq_along(tmin)) {
+    lo <- tmin[i]
+    hi <- tmax[i]
+    if (is.na(lo) || is.na(hi)) {
+      out[i] <- NA_real_
+      next
+    }
+    if (lo >= upper) {
+      out[i] <- upper - base
+    } else if (hi <= base) {
+      out[i] <- 0
+    } else if (lo >= base) {
+      out[i] <- (lo + hi) / 2 - base
+    } else {
+      out[i] <- (hi - base)^2 / (2 * (hi - lo))
+    }
+    if (hi > upper && lo < upper) {
+      out[i] <- out[i] - (hi - upper)^2 / (2 * (hi - lo))
+    }
+  }
+  out
+}
```

**Target (gold region)**

```r
    lo <- tmin[i]
    hi <- tmax[i]
    if (is.na(lo) || is.na(hi)) {
      out[i] <- NA_real_
      next
    }
    if (lo >= upper) {
      out[i] <- upper - base
    } else if (hi <= base) {
      out[i] <- 0
    } else if (lo >= base) {
      out[i] <- (lo + hi) / 2 - base
    } else {
      out[i] <- (hi - base)^2 / (2 * (hi - lo))
    }
    if (hi > upper && lo < upper) {
      out[i] <- out[i] - (hi - upper)^2 / (2 * (hi - lo))
    }
  }
  out
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/degree_days.R
#' Check daily temperature inputs
#'
#' @param tmin,tmax Numeric vectors of daily minimum and maximum air
#'   temperature in degrees Celsius.
#' @return `NULL`, invisibly; called for its error side effects.
#' @keywords internal
checkTemps <- function(tmin, tmax) {
  if (length(tmin) != length(tmax)) {
    stop("`tmin` and `tmax` must have the same length.", call. = FALSE)
  }
  swapped <- !is.na(tmin) & !is.na(tmax) & tmin > tmax
  if (any(swapped)) {
    stop(
      sprintf("%d day(s) have `tmin` above `tmax`.", sum(swapped)),
      call. = FALSE
    )
  }
  invisible(NULL)
}

#' Daily degree-days by the averaging method
#'
#' Subtracts the base temperature from the daily mean, after capping the
#' maximum at the upper threshold.
#'
#' @param tmin,tmax Numeric vectors of daily minimum and maximum air
#'   temperature in degrees Celsius.
#' @param base Lower developmental threshold in degrees Celsius.
#' @param upper Upper developmental threshold in degrees Celsius.
#' @return A numeric vector of daily degree-days.
#' @family degree-day methods
#' @export
ddAverage <- function(tmin, tmax, base = 10, upper = 30) {
  checkTemps(tmin, tmax)
  out <- numeric(length(tmin))
  for (i in seq_along(tmin)) {
    if (is.na(tmin[i]) || is.na(tmax[i])) {
      out[i] <- NA_real_
      next
    }
    hi <- min(tmax[i], upper)
    lo <- min(tmin[i], upper)
    out[i] <- max(0, (hi + lo) / 2 - base)
  }
  out
}

#' Daily degree-days by the single-triangle method
#'
#' Approximates the daily temperature curve by a triangle and integrates the
#' area between the base and upper thresholds.
#'
#' @param tmin,tmax Numeric vectors of daily minimum and maximum air
#'   temperature in degrees Celsius.
#' @param base Lower developmental threshold in degrees Celsius.
#' @param upper Upper developmental threshold in degrees Celsius.
#' @return A numeric vector of daily degree-days.
#' @family degree-day methods
#' @export
ddTriangle <- function(tmin, tmax, base = 10, upper = 30) {
  checkTemps(tmin, tmax)
  out <- numeric(length(tmin))
  for (i in seq_along(tmin)) {
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

## 32. `1b7bac28926dc1ce1d5ef729` finish_block / long

- package `synth:markovine`, path `R/branching.R`
- generator: `{"bucket": "long", "novel_tokens": ["p.hat", "run", "sqrt", "totals"], "novelty": 4, "rule": "fb_statement", "seed": "b23-simulation/markovine/R/branching.R", "signature_region": false, "style": "dense", "target_lines": 21, "unit": "bp.check.extinction", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 159, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1424, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -163,3 +163,24 @@
   founders = 1,
   generations = 100
 ) {
+  extinct <- logical(reps)
+  totals <- rep(NA_real_, reps)
+  for (r in seq_len(reps)) {
+    run <- bp.simulate(
+      offspring,
+      generations = generations,
+      founders = founders
+    )
+    extinct[r] <- run$extinct
+    if (run$extinct) {
+      totals[r] <- run$total
+    }
+  }
+  p.hat <- mean(extinct)
+  return(list(
+    simulated = p.hat,
+    theoretical = bp.extinction.prob(offspring, founders = founders),
+    se = sqrt(p.hat * (1 - p.hat) / reps),
+    totals = totals[!is.na(totals)]
+  ))
+}
```

**Target (gold region)**

```r
  extinct <- logical(reps)
  totals <- rep(NA_real_, reps)
  for (r in seq_len(reps)) {
    run <- bp.simulate(
      offspring,
      generations = generations,
      founders = founders
    )
    extinct[r] <- run$extinct
    if (run$extinct) {
      totals[r] <- run$total
    }
  }
  p.hat <- mean(extinct)
  return(list(
    simulated = p.hat,
    theoretical = bp.extinction.prob(offspring, founders = founders),
    se = sqrt(p.hat * (1 - p.hat) / reps),
    totals = totals[!is.na(totals)]
  ))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/branching.R
#' @return A list of class `offspring` with a sampler `r` and a probability
#'   generating function `pgf`.
#' @family offspring distributions
#' @export
offspring.negbin <- function(mean, size) {
  stopifnot(mean >= 0, size > 0)
  return(structure(
    list(
      name = "negbin",
      mean = mean,
      r = function(n) stats::rnbinom(n, size = size, mu = mean),
      pgf = function(s) (1 + mean / size * (1 - s))^(-size)
    ),
    class = "offspring"
  ))
}

#' Geometric offspring distribution
#'
#' Describes a branching process with geometric offspring counts on
#' `0, 1, 2, ...`.
#'
#' @param mean Mean number of offspring per individual.
#' @return A list of class `offspring` with a sampler `r` and a probability
#'   generating function `pgf`.
#' @family offspring distributions
#' @export
offspring.geometric <- function(mean) {
  stopifnot(mean >= 0)
  p <- 1 / (1 + mean)
  return(structure(
    list(
      name = "geometric",
      mean = mean,
      r = function(n) stats::rgeom(n, p),
      pgf = function(s) p / (1 - (1 - p) * s)
    ),
    class = "offspring"
  ))
}

#' Print an offspring distribution
#'
#' @param x An `offspring` object.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.offspring <- function(x, ...) {
  regime <- if (x$mean < 1) {
    "subcritical"
  } else if (x$mean > 1) {
    "supercritical"
  } else {
    "critical"
  }
  cat(
    "<offspring> ",
    x$name,
    " with mean ",
    format(x$mean),
    " (",
    regime,
    ")\n",
    sep = ""
  )
  return(invisible(x))
}

#' Simulate generation sizes of a Galton-Watson process
#'
#' Starts from a number of founders and draws offspring for every
#' individual in every generation. Stops at extinction or once the
#' population exceeds a cap, which is treated as a major outbreak.
#'
#' @param offspring An `offspring` object, e.g. from [offspring.poisson()].
#' @param generations Maximum number of generations.
#' @param founders Number of individuals in generation zero.
#' @param cap Population size above which simulation stops.
#' @return A list with elements `sizes` (generation sizes), `extinct`,
#'   `capped` and `total`.
#' @family branching processes
#' @export
bp.simulate <- function(offspring, generations = 50, founders = 1, cap = 1e5) {
  stopifnot(inherits(offspring, "offspring"))
  sizes <- rep(NA_real_, generations + 1)
  sizes[1] <- founders
  extinct <- FALSE
  capped <- FALSE
  for (g in seq_len(generations)) {
    current <- sizes[g]
    if (current == 0) {
      extinct <- TRUE
      break
    }
    if (current > cap) {
      capped <- TRUE
      break
    }
    sizes[g + 1] <- sum(offspring$r(current))
  }
  sizes <- sizes[!is.na(sizes)]
  if (!extinct && !capped && utils::tail(sizes, 1) == 0) {
    extinct <- TRUE
  }
  return(list(
    sizes = sizes,
    extinct = extinct,
    capped = capped,
    total = sum(sizes)
  ))
}

#' Extinction probability of a branching process
#'
#' Finds the smallest fixed point of the probability generating function on
#' `[0, 1]` by iterating `q <- pgf(q)` from zero.
#'
#' @param offspring An `offspring` object, e.g. from [offspring.poisson()].
#' @param founders Number of individuals in generation zero.
#' @param tol Convergence tolerance.
#' @param max.iter Maximum number of iterations.
#' @return The probability that the process eventually dies out.
#' @family branching processes
#' @export
bp.extinction.prob <- function(
  offspring,
  founders = 1,
  tol = 1e-10,
  max.iter = 10000
) {
  stopifnot(inherits(offspring, "offspring"))
  if (offspring$mean <= 1) {
    return(1)
  }
  q <- 0
  for (i in seq_len(max.iter)) {
    q.new <- offspring$pgf(q)
    if (abs(q.new - q) < tol) {
      q <- q.new
      break
    }
    q <- q.new
  }
  return(q^founders)
}

#' Compare simulated and theoretical extinction
#'
#' Runs many branching processes and contrasts the share that went extinct
#' with the extinction probability from [bp.extinction.prob()].
#'
#' @param offspring An `offspring` object, e.g. from [offspring.poisson()].
#' @param reps Number of simulated processes.
#' @param founders Number of individuals in generation zero.
#' @param generations Maximum number of generations.
#' @return A list with elements `simulated`, `theoretical`, `se` and
#'   `totals` (total progeny of the extinct runs).
#' @family branching processes
#' @export
bp.check.extinction <- function(
  offspring,
  reps = 1000,
  founders = 1,
  generations = 100
) {
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

## 33. `49188a3572a0b06936f1893b` finish_block / long

- package `synth:elotide`, path `R/xg.R`
- generator: `{"bucket": "long", "novel_tokens": ["3", "awayWins", "draws", "ga", "gh", "homeWins", "pAway", "pDraw", "pHome", "runif", "set.seed", "xPtsAway", "xPtsHome"], "novelty": 13, "rule": "fb_statement", "seed": "b21-applied-analytics/elotide/R/xg.R", "signature_region": false, "style": "sectioned", "target_lines": 28, "unit": "simulateFromXg", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 254, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1484, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -148,3 +148,31 @@
 #'   probabilities and the expected points of each side.
 #' @export
 simulateFromXg <- function(xgHome, xgAway, nSims = 10000L, seed = NULL) {
+  if (!is.null(seed)) {
+    set.seed(seed)
+  }
+  homeWins <- 0L
+  draws <- 0L
+  awayWins <- 0L
+  for (s in seq_len(nSims)) {
+    gh <- sum(stats::runif(length(xgHome)) < xgHome)
+    ga <- sum(stats::runif(length(xgAway)) < xgAway)
+    if (gh > ga) {
+      homeWins <- homeWins + 1L
+    } else if (gh == ga) {
+      draws <- draws + 1L
+    } else {
+      awayWins <- awayWins + 1L
+    }
+  }
+  pHome <- homeWins / nSims
+  pDraw <- draws / nSims
+  pAway <- awayWins / nSims
+  c(
+    home = pHome,
+    draw = pDraw,
+    away = pAway,
+    xPtsHome = 3 * pHome + pDraw,
+    xPtsAway = 3 * pAway + pDraw
+  )
+}
```

**Target (gold region)**

```r
  if (!is.null(seed)) {
    set.seed(seed)
  }
  homeWins <- 0L
  draws <- 0L
  awayWins <- 0L
  for (s in seq_len(nSims)) {
    gh <- sum(stats::runif(length(xgHome)) < xgHome)
    ga <- sum(stats::runif(length(xgAway)) < xgAway)
    if (gh > ga) {
      homeWins <- homeWins + 1L
    } else if (gh == ga) {
      draws <- draws + 1L
    } else {
      awayWins <- awayWins + 1L
    }
  }
  pHome <- homeWins / nSims
  pDraw <- draws / nSims
  pAway <- awayWins / nSims
  c(
    home = pHome,
    draw = pDraw,
    away = pAway,
    xPtsHome = 3 * pHome + pDraw,
    xPtsAway = 3 * pAway + pDraw
  )
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/xg.R
#' Distance from a shot location to the goal centre
#'
#' Pitch coordinates are in metres with the attacking goal centred at
#' `(pitchLength, pitchWidth / 2)`.
#'
#' @param x,y Numeric shot coordinates.
#' @param pitchLength,pitchWidth Pitch dimensions in metres.
#' @return A numeric vector of distances in metres.
#' @family shot geometry
#' @export
shotDistance <- function(x, y, pitchLength = 105, pitchWidth = 68) {
  dx <- pitchLength - x
  dy <- pitchWidth / 2 - y
  sqrt(dx^2 + dy^2)
}

#' Angle subtended by the goal mouth from a shot location
#'
#' The wider the visible goal, the larger the angle and the better the chance.
#'
#' @param x,y Numeric shot coordinates.
#' @param pitchLength,pitchWidth Pitch dimensions in metres.
#' @param goalWidth Width of the goal mouth in metres.
#' @return A numeric vector of angles in radians.
#' @family shot geometry
#' @export
shotAngle <- function(
  x,
  y,
  pitchLength = 105,
  pitchWidth = 68,
  goalWidth = 7.32
) {
  dx <- pitchLength - x
  dy <- pitchWidth / 2 - y
  half <- goalWidth / 2
  angle <- atan2(half - dy, dx) - atan2(-half - dy, dx)
  abs(angle)
}

#' Simple logistic expected-goals model
#'
#' Scores each shot from its distance, angle and body part using fixed
#' coefficients. Penalties receive a flat value.
#'
#' @param shots A data.table of shots with columns `x`, `y`, `bodyPart` and
#'   `situation`.
#' @param coefs Named numeric vector with `intercept`, `distance`, `angle` and
#'   `header` coefficients.
#' @param penaltyXg Value assigned to penalty kicks.
#' @return A numeric vector of shot xG values.
#' @export
shotXg <- function(
  shots,
  coefs = c(intercept = -1.1, distance = -0.11, angle = 1.6, header = -0.9),
  penaltyXg = 0.76
) {
  dist <- shotDistance(shots$x, shots$y)
  ang <- shotAngle(shots$x, shots$y)
  header <- as.numeric(shots$bodyPart == "head")
  eta <- coefs[["intercept"]] +
    coefs[["distance"]] * dist +
    coefs[["angle"]] * ang +
    coefs[["header"]] * header
  xg <- stats::plogis(eta)
  xg[shots$situation == "penalty"] <- penaltyXg
  xg
}

#' Expected goals per team per match
#'
#' Sums shot xG by match and team and attaches actual goals for comparison.
#'
#' @param shots A data.table of shots with columns `matchId`, `team`, `xg` and
#'   `isGoal`.
#' @return A data.table with columns `matchId`, `team`, `shots`, `xg` and
#'   `goals`.
#' @export
xgByMatch <- function(shots) {
  dt <- data.table::as.data.table(shots)
  if (!"xg" %in% names(dt)) {
    dt[, xg := shotXg(dt)]
  }
  dt[,
    .(shots = .N, xg = sum(xg), goals = sum(isGoal)),
    by = .(matchId, team)
  ][]
}

#' Season expected goals for each team
#'
#' Aggregates match-level xG for the team in possession of the shot.
#'
#' @param matchXg A data.table returned by [xgByMatch()].
#' @return A data.table with columns `team`, `matches`, `xgFor`, `goalsFor`
#'   and `xgForPerMatch`.
#' @family season xG
#' @export
xgForByTeam <- function(matchXg) {
  out <- matchXg[,
    .(
      matches = data.table::uniqueN(matchId),
      xgFor = sum(xg),
      goalsFor = sum(goals)
    ),
    by = team
  ]
  out[, xgForPerMatch := xgFor / matches]
  data.table::setorder(out, -xgFor)
  out[]
}

#' Season expected goals conceded for each team
#'
#' Aggregates match-level xG of the opponent in each match.
#'
#' @param matchXg A data.table returned by [xgByMatch()].
#' @return A data.table with columns `team`, `matches`, `xgAgainst`,
#'   `goalsAgainst` and `xgAgainstPerMatch`.
#' @family season xG
#' @export
xgAgainstByTeam <- function(matchXg) {
  opp <- matchXg[matchXg, on = "matchId", allow.cartesian = TRUE][
    team != i.team
  ]
  out <- opp[,
    .(
      matches = data.table::uniqueN(matchId),
      xgAgainst = sum(i.xg),
      goalsAgainst = sum(i.goals)
    ),
    by = team
  ]
  out[, xgAgainstPerMatch := xgAgainst / matches]
  data.table::setorder(out, xgAgainst)
  out[]
}

#' Simulate match outcomes from shot xG
#'
#' Treats every shot as an independent Bernoulli trial and simulates the
#' scoreline many times to obtain win, draw and loss probabilities.
#'
#' @param xgHome,xgAway Numeric vectors of shot xG for each side.
#' @param nSims Number of simulated replays.
#' @param seed Optional random seed for reproducibility.
#' @return A named numeric vector with `home`, `draw` and `away`
#'   probabilities and the expected points of each side.
#' @export
simulateFromXg <- function(xgHome, xgAway, nSims = 10000L, seed = NULL) {
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

## 34. `b005b6a0d9d89d60d6e80b8b` finish_block / long

- package `synth:markovine`, path `R/branching.R`
- generator: `{"bucket": "long", "novel_tokens": ["capped", "current", "tail", "total", "utils"], "novelty": 5, "rule": "fb_statement", "seed": "b23-simulation/markovine/R/branching.R", "signature_region": false, "style": "dense", "target_lines": 28, "unit": "bp.simulate", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 196, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 1021, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -111,3 +111,31 @@
 #' @family branching processes
 #' @export
 bp.simulate <- function(offspring, generations = 50, founders = 1, cap = 1e5) {
+  stopifnot(inherits(offspring, "offspring"))
+  sizes <- rep(NA_real_, generations + 1)
+  sizes[1] <- founders
+  extinct <- FALSE
+  capped <- FALSE
+  for (g in seq_len(generations)) {
+    current <- sizes[g]
+    if (current == 0) {
+      extinct <- TRUE
+      break
+    }
+    if (current > cap) {
+      capped <- TRUE
+      break
+    }
+    sizes[g + 1] <- sum(offspring$r(current))
+  }
+  sizes <- sizes[!is.na(sizes)]
+  if (!extinct && !capped && utils::tail(sizes, 1) == 0) {
+    extinct <- TRUE
+  }
+  return(list(
+    sizes = sizes,
+    extinct = extinct,
+    capped = capped,
+    total = sum(sizes)
+  ))
+}
```

**Target (gold region)**

```r
  stopifnot(inherits(offspring, "offspring"))
  sizes <- rep(NA_real_, generations + 1)
  sizes[1] <- founders
  extinct <- FALSE
  capped <- FALSE
  for (g in seq_len(generations)) {
    current <- sizes[g]
    if (current == 0) {
      extinct <- TRUE
      break
    }
    if (current > cap) {
      capped <- TRUE
      break
    }
    sizes[g + 1] <- sum(offspring$r(current))
  }
  sizes <- sizes[!is.na(sizes)]
  if (!extinct && !capped && utils::tail(sizes, 1) == 0) {
    extinct <- TRUE
  }
  return(list(
    sizes = sizes,
    extinct = extinct,
    capped = capped,
    total = sum(sizes)
  ))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/branching.R
#' Poisson offspring distribution
#'
#' Describes a branching process in which each individual has a Poisson
#' number of offspring.
#'
#' @param mean Mean number of offspring per individual.
#' @return A list of class `offspring` with a sampler `r` and a probability
#'   generating function `pgf`.
#' @family offspring distributions
#' @export
offspring.poisson <- function(mean) {
  stopifnot(mean >= 0)
  return(structure(
    list(
      name = "poisson",
      mean = mean,
      r = function(n) stats::rpois(n, mean),
      pgf = function(s) exp(mean * (s - 1))
    ),
    class = "offspring"
  ))
}

#' Negative binomial offspring distribution
#'
#' Describes a branching process with overdispersed offspring counts, the
#' usual model for superspreading.
#'
#' @param mean Mean number of offspring per individual.
#' @param size Dispersion parameter; smaller values mean more variation.
#' @return A list of class `offspring` with a sampler `r` and a probability
#'   generating function `pgf`.
#' @family offspring distributions
#' @export
offspring.negbin <- function(mean, size) {
  stopifnot(mean >= 0, size > 0)
  return(structure(
    list(
      name = "negbin",
      mean = mean,
      r = function(n) stats::rnbinom(n, size = size, mu = mean),
      pgf = function(s) (1 + mean / size * (1 - s))^(-size)
    ),
    class = "offspring"
  ))
}

#' Geometric offspring distribution
#'
#' Describes a branching process with geometric offspring counts on
#' `0, 1, 2, ...`.
#'
#' @param mean Mean number of offspring per individual.
#' @return A list of class `offspring` with a sampler `r` and a probability
#'   generating function `pgf`.
#' @family offspring distributions
#' @export
offspring.geometric <- function(mean) {
  stopifnot(mean >= 0)
  p <- 1 / (1 + mean)
  return(structure(
    list(
      name = "geometric",
      mean = mean,
      r = function(n) stats::rgeom(n, p),
      pgf = function(s) p / (1 - (1 - p) * s)
    ),
    class = "offspring"
  ))
}

#' Print an offspring distribution
#'
#' @param x An `offspring` object.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.offspring <- function(x, ...) {
  regime <- if (x$mean < 1) {
    "subcritical"
  } else if (x$mean > 1) {
    "supercritical"
  } else {
    "critical"
  }
  cat(
    "<offspring> ",
    x$name,
    " with mean ",
    format(x$mean),
    " (",
    regime,
    ")\n",
    sep = ""
  )
  return(invisible(x))
}

#' Simulate generation sizes of a Galton-Watson process
#'
#' Starts from a number of founders and draws offspring for every
#' individual in every generation. Stops at extinction or once the
#' population exceeds a cap, which is treated as a major outbreak.
#'
#' @param offspring An `offspring` object, e.g. from [offspring.poisson()].
#' @param generations Maximum number of generations.
#' @param founders Number of individuals in generation zero.
#' @param cap Population size above which simulation stops.
#' @return A list with elements `sizes` (generation sizes), `extinct`,
#'   `capped` and `total`.
#' @family branching processes
#' @export
bp.simulate <- function(offspring, generations = 50, founders = 1, cap = 1e5) {
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

## 35. `1124a17e3037900d3de0aff1` finish_block / long

- package `synth:graphwalk`, path `R/shortest_path.R`
- generator: `{"bucket": "long", "novel_tokens": ["alt", "is.infinite", "nb", "open", "which.min"], "novelty": 5, "rule": "fb_statement", "seed": "b19-graphs-or/graphwalk/R/shortest_path.R", "signature_region": false, "style": "dense", "target_lines": 22, "unit": "dijkstra", "variant": "file"}`
- verification: `{"air_idempotent": true, "completion_tokens": 150, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 320, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -17,3 +17,25 @@
   dist <- setNames(rep(Inf, length(v)), v)
   prev <- setNames(rep(NA_character_, length(v)), v)
   done <- setNames(rep(FALSE, length(v)), v)
+  dist[source] <- 0
+  repeat {
+    open <- v[!done]
+    if (length(open) == 0) {
+      break
+    }
+    u <- open[which.min(dist[open])]
+    if (is.infinite(dist[u])) {
+      break
+    }
+    done[u] <- TRUE
+    nb <- wadj[[u]]
+    for (w in names(nb)) {
+      alt <- dist[u] + nb[[w]]
+      if (alt < dist[w]) {
+        dist[w] <- alt
+        prev[w] <- u
+      }
+    }
+  }
+  return(list(dist = dist, prev = prev))
+}
```

**Target (gold region)**

```r
  dist[source] <- 0
  repeat {
    open <- v[!done]
    if (length(open) == 0) {
      break
    }
    u <- open[which.min(dist[open])]
    if (is.infinite(dist[u])) {
      break
    }
    done[u] <- TRUE
    nb <- wadj[[u]]
    for (w in names(nb)) {
      alt <- dist[u] + nb[[w]]
      if (alt < dist[w]) {
        dist[w] <- alt
        prev[w] <- u
      }
    }
  }
  return(list(dist = dist, prev = prev))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/shortest_path.R
#' Single-source shortest paths with Dijkstra's algorithm
#'
#' Uses a simple linear scan instead of a heap, which is fine for the small
#' graphs this package targets (a few thousand vertices).
#'
#' @param wadj A weighted adjacency list as returned by
#'   [edges_to_weighted_adjlist()].
#' @param source Name of the starting vertex.
#' @return A list with named numeric `dist` and named character `prev`.
#' @export
#' @family shortest-paths
dijkstra <- function(wadj, source) {
  if (!source %in% names(wadj)) {
    stop("Unknown source vertex: ", source, call. = FALSE)
  }
  v <- names(wadj)
  dist <- setNames(rep(Inf, length(v)), v)
  prev <- setNames(rep(NA_character_, length(v)), v)
  done <- setNames(rep(FALSE, length(v)), v)
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

## 36. `ef71ba4f992b9399c3cd6473` finish_block / long

- package `synth:aerocal`, path `R/calibration.R`
- generator: `{"bucket": "long", "novel_tokens": ["bias", "cor", "err", "mae", "nrmse", "ok", "r2", "rmse", "sqrt", "stats"], "novelty": 10, "rule": "fb_statement", "seed": "b20-environment/aerocal/R/calibration.R", "signature_region": true, "style": "dense", "target_lines": 24, "unit": "calMetrics", "variant": "strip"}`
- verification: `{"air_idempotent": true, "completion_tokens": 170, "family_problems": [], "parse_after": true, "parse_before_status": "incomplete", "passed": true, "prompt_tokens": 125, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -1 +1,25 @@
 calMetrics <- function(estimate, reference) {
+  ok <- !is.na(estimate) & !is.na(reference)
+  e <- estimate[ok]
+  r <- reference[ok]
+  if (length(e) < 2) {
+    return(c(
+      n = length(e),
+      bias = NA,
+      mae = NA,
+      rmse = NA,
+      nrmse = NA,
+      r2 = NA
+    ))
+  }
+  err <- e - r
+  rmse <- sqrt(mean(err^2))
+  return(c(
+    n = length(e),
+    bias = mean(err),
+    mae = mean(abs(err)),
+    rmse = rmse,
+    nrmse = rmse / mean(r),
+    r2 = stats::cor(e, r)^2
+  ))
+}
```

**Target (gold region)**

```r
calMetrics <- function(estimate, reference) {
  ok <- !is.na(estimate) & !is.na(reference)
  e <- estimate[ok]
  r <- reference[ok]
  if (length(e) < 2) {
    return(c(
      n = length(e),
      bias = NA,
      mae = NA,
      rmse = NA,
      nrmse = NA,
      r2 = NA
    ))
  }
  err <- e - r
  rmse <- sqrt(mean(err^2))
  return(c(
    n = length(e),
    bias = mean(err),
    mae = mean(abs(err)),
    rmse = rmse,
    nrmse = rmse / mean(r),
    r2 = stats::cor(e, r)^2
  ))
}
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/calibration.R
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
<<<<<<< CURRENT
calMetrics <- function(estimate, reference) {<|user_cursor|>
=======
<[fim-middle]>
````

</details>

## 37. `25b1768959cd2fba3f9eeab0` roxygen_drafting / roxy_full

- package `synth:panelwick`, path `R/indices.R`
- generator: `{"determined_share": 0.833, "rule": "roxy_full", "seed": "b09-finance/panelwick/R/indices.R", "style": "dense", "target_lines": 12, "unit": "fisherIndex"}`
- verification: `{"air_idempotent": true, "completion_tokens": 132, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 427, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -20,6 +20,18 @@
   out[order(period)]
 }
 
+#' Fisher ideal price index
+#'
+#' The geometric mean of the Laspeyres and Paasche indices.
+#'
+#' @param prices A data.table with one row per item and period holding
+#'   prices and quantities.
+#' @param base Label of the base period.
+#' @param cols Named character vector giving the `item`, `period`, `price`
+#'   and `qty` column names.
+#' @return A data.table with columns `period` and `index` (base = 100).
+#' @family price indices
+#' @export
 fisherIndex <- function(
   prices,
   base,
```

**Target (gold region)**

```r
#' Fisher ideal price index
#'
#' The geometric mean of the Laspeyres and Paasche indices.
#'
#' @param prices A data.table with one row per item and period holding
#'   prices and quantities.
#' @param base Label of the base period.
#' @param cols Named character vector giving the `item`, `period`, `price`
#'   and `qty` column names.
#' @return A data.table with columns `period` and `index` (base = 100).
#' @family price indices
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/indices.R
#' Paasche price index
#'
#' Weights price relatives by current-period quantities.
#'
#' @param prices A data.table with one row per item and period holding
#'   prices and quantities.
#' @param base Label of the base period.
#' @param cols Named character vector giving the `item`, `period`, `price`
#'   and `qty` column names.
#' @return A data.table with columns `period` and `index` (base = 100).
#' @family price indices
#' @export
paascheIndex <- function(
  prices,
  base,
  cols = c(item = "item", period = "period", price = "price", qty = "qty")
) {
  dt <- alignToBase(prices, base, cols)
  out <- dt[, .(index = 100 * sum(p1 * q1) / sum(p0 * q1)), by = period]
  out[order(period)]
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
fisherIndex <- function(
  prices,
  base,
  cols = c(item = "item", period = "period", price = "price", qty = "qty")
) {
  las <- laspeyresIndex(prices, base, cols)
  paa <- paascheIndex(prices, base, cols)
  out <- las[paa, on = "period", .(period, index = sqrt(index * i.index))]
  out[order(period)]
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 38. `c87f3c70e131a0cfdd70e2f1` roxygen_drafting / roxy_full

- package `synth:foldwise`, path `R/metrics.R`
- generator: `{"determined_share": 0.8, "rule": "roxy_full", "seed": "b11-ml/foldwise/R/metrics.R", "style": "dense", "target_lines": 10, "unit": "rsq_trad_vec"}`
- verification: `{"air_idempotent": true, "completion_tokens": 92, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 491, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -28,6 +28,16 @@
   100 * mean(abs((p$truth[nonzero] - p$estimate[nonzero]) / p$truth[nonzero]))
 }
 
+#' Coefficient of determination
+#'
+#' Computes one minus the ratio of residual to total sum of squares.
+#'
+#' @param truth A numeric vector of observed values.
+#' @param estimate A numeric vector of predicted values.
+#' @param na_rm Logical; drop incomplete pairs before computing.
+#' @return A single numeric value.
+#' @family regression metrics
+#' @export
 rsq_trad_vec <- function(truth, estimate, na_rm = TRUE) {
   p <- align_pairs(truth, estimate, na_rm)
   ss_res <- sum((p$truth - p$estimate)^2)
```

**Target (gold region)**

```r
#' Coefficient of determination
#'
#' Computes one minus the ratio of residual to total sum of squares.
#'
#' @param truth A numeric vector of observed values.
#' @param estimate A numeric vector of predicted values.
#' @param na_rm Logical; drop incomplete pairs before computing.
#' @return A single numeric value.
#' @family regression metrics
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/metrics.R
#' @param estimate A numeric vector of predicted values.
#' @param na_rm Logical; drop incomplete pairs before computing.
#' @return A single numeric value.
#' @family regression metrics
#' @export
mae_vec <- function(truth, estimate, na_rm = TRUE) {
  p <- align_pairs(truth, estimate, na_rm)
  mean(abs(p$truth - p$estimate))
}

#' Mean absolute percentage error
#'
#' Computes the mean absolute error relative to the observed values, as a
#' percentage. Pairs with a zero observed value are skipped.
#'
#' @param truth A numeric vector of observed values.
#' @param estimate A numeric vector of predicted values.
#' @param na_rm Logical; drop incomplete pairs before computing.
#' @return A single numeric value.
#' @family regression metrics
#' @export
mape_vec <- function(truth, estimate, na_rm = TRUE) {
  p <- align_pairs(truth, estimate, na_rm)
  nonzero <- p$truth != 0
  if (!any(nonzero)) {
    return(NA_real_)
  }
  100 * mean(abs((p$truth[nonzero] - p$estimate[nonzero]) / p$truth[nonzero]))
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
rsq_trad_vec <- function(truth, estimate, na_rm = TRUE) {
  p <- align_pairs(truth, estimate, na_rm)
  ss_res <- sum((p$truth - p$estimate)^2)
  ss_tot <- sum((p$truth - mean(p$truth))^2)
  if (ss_tot == 0) {
    return(NA_real_)
  }
  1 - ss_res / ss_tot
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 39. `cea599e7c690871299ff783f` roxygen_drafting / roxy_full

- package `synth:quantia`, path `R/arith.R`
- generator: `{"determined_share": 1.0, "rule": "roxy_full", "seed": "b05-oop/quantia/R/arith.R", "style": "sectioned", "target_lines": 1, "unit": "setMethod"}`
- verification: `{"air_idempotent": true, "completion_tokens": 15, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 504, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -27,6 +27,7 @@
   )
 })
 
+#' @rdname quantity-arith
 setMethod("Arith", signature("numeric", "Quantity"), function(e1, e2) {
   op <- .Generic
   switch(
```

**Target (gold region)**

```r
#' @rdname quantity-arith
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/arith.R
#' @rdname quantity-arith
setMethod("Arith", signature("Quantity", "numeric"), function(e1, e2) {
  op <- .Generic
  switch(
    op,
    "*" = ,
    "/" = new("Quantity", callGeneric(e1@.Data, e2), unit = e1@unit),
    "^" = {
      if (length(e2) != 1L) {
        stop("Exponent must be a single number.", call. = FALSE)
      }
      new("Quantity", e1@.Data^e2, unit = powerUnit(e1@unit, e2))
    },
    {
      if (!isDimensionless(e1@unit)) {
        stop(
          "Can't apply `",
          op,
          "` to ",
          e1@unit@symbol,
          " and a plain number.",
          call. = FALSE
        )
      }
      new("Quantity", callGeneric(e1@.Data, e2), unit = e1@unit)
    }
  )
})

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
setMethod("Arith", signature("numeric", "Quantity"), function(e1, e2) {
  op <- .Generic
  switch(
    op,
    "*" = new("Quantity", e1 * e2@.Data, unit = e2@unit),
    "/" = new("Quantity", e1 / e2@.Data, unit = powerUnit(e2@unit, -1)),
    {
      if (!isDimensionless(e2@unit)) {
        stop(
          "Can't apply `",
          op,
          "` to a plain number and ",
          e2@unit@symbol,
          ".",
          call. = FALSE
        )
      }
      new("Quantity", callGeneric(e1, e2@.Data), unit = e2@unit)
    }
  )
})
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 40. `8986a41e815ab059931bc6d7` roxygen_drafting / roxy_full

- package `synth:visitkit`, path `R/baseline.R`
- generator: `{"determined_share": 0.75, "rule": "roxy_full", "seed": "b01-tidyverse/visitkit/R/baseline.R", "style": "dense", "target_lines": 8, "unit": "derive.pct.change"}`
- verification: `{"air_idempotent": true, "completion_tokens": 116, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 372, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -17,6 +17,14 @@
     )
 }
 
+#' Derive the percent change from baseline
+#'
+#' @param data A long data frame of measurements with a `base` column, as
+#'   returned by [derive.baseline()].
+#' @param value <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the measurement.
+#' @return `data` with an added `pchg` column; `NA` on and before baseline.
+#' @family change derivations
+#' @export
 derive.pct.change <- function(data, value = aval) {
   data %>%
     dplyr::mutate(
```

**Target (gold region)**

```r
#' Derive the percent change from baseline
#'
#' @param data A long data frame of measurements with a `base` column, as
#'   returned by [derive.baseline()].
#' @param value <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the measurement.
#' @return `data` with an added `pchg` column; `NA` on and before baseline.
#' @family change derivations
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/baseline.R
#' Derive the absolute change from baseline
#'
#' @param data A long data frame of measurements with a `base` column, as
#'   returned by [derive.baseline()].
#' @param value <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the measurement.
#' @return `data` with an added `chg` column; `NA` on and before baseline.
#' @family change derivations
#' @export
derive.change <- function(data, value = aval) {
  data %>%
    dplyr::mutate(
      chg = dplyr::if_else(
        .data$study_day > 1L,
        {{ value }} - .data$base,
        NA_real_
      )
    )
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
derive.pct.change <- function(data, value = aval) {
  data %>%
    dplyr::mutate(
      pchg = dplyr::if_else(
        .data$study_day > 1L & .data$base != 0,
        100 * ({{ value }} - .data$base) / .data$base,
        NA_real_
      )
    )
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 41. `3e7d1762eb0e40293f2bbbdc` roxygen_drafting / roxy_full

- package `synth:panelwick`, path `R/indices.R`
- generator: `{"determined_share": 0.667, "rule": "roxy_full", "seed": "b09-finance/panelwick/R/indices.R", "style": "dense", "target_lines": 12, "unit": "laspeyresIndex"}`
- verification: `{"air_idempotent": true, "completion_tokens": 127, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 478, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -21,6 +21,18 @@
   baseRows[dt, on = "item", nomatch = NULL]
 }
 
+#' Laspeyres price index
+#'
+#' Weights price relatives by base-period quantities.
+#'
+#' @param prices A data.table with one row per item and period holding
+#'   prices and quantities.
+#' @param base Label of the base period.
+#' @param cols Named character vector giving the `item`, `period`, `price`
+#'   and `qty` column names.
+#' @return A data.table with columns `period` and `index` (base = 100).
+#' @family price indices
+#' @export
 laspeyresIndex <- function(
   prices,
   base,
```

**Target (gold region)**

```r
#' Laspeyres price index
#'
#' Weights price relatives by base-period quantities.
#'
#' @param prices A data.table with one row per item and period holding
#'   prices and quantities.
#' @param base Label of the base period.
#' @param cols Named character vector giving the `item`, `period`, `price`
#'   and `qty` column names.
#' @return A data.table with columns `period` and `index` (base = 100).
#' @family price indices
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/indices.R
#' Join each period's prices to the base period
#'
#' @param prices A data.table with one row per item and period holding
#'   prices and quantities.
#' @param base Label of the base period.
#' @param cols Named character vector giving the `item`, `period`, `price`
#'   and `qty` column names.
#' @return A data.table with columns `period`, `p1`, `q1`, `p0`, `q0`.
#' @keywords internal
alignToBase <- function(prices, base, cols) {
  checkPriceTable(prices, cols)
  dt <- prices[,
    c(cols[["item"]], cols[["period"]], cols[["price"]], cols[["qty"]]),
    with = FALSE
  ]
  data.table::setnames(dt, c("item", "period", "p1", "q1"))
  baseRows <- dt[period == base, .(item, p0 = p1, q0 = q1)]
  if (nrow(baseRows) == 0) {
    stop("No rows found for base period ", base, ".", call. = FALSE)
  }
  baseRows[dt, on = "item", nomatch = NULL]
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
laspeyresIndex <- function(
  prices,
  base,
  cols = c(item = "item", period = "period", price = "price", qty = "qty")
) {
  dt <- alignToBase(prices, base, cols)
  out <- dt[, .(index = 100 * sum(p1 * q0) / sum(p0 * q0)), by = period]
  out[order(period)]
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 42. `6e29f8ca4954622726d672cb` roxygen_drafting / roxy_full

- package `synth:driftbook`, path `R/returns.R`
- generator: `{"determined_share": 0.833, "rule": "roxy_full", "seed": "b09-finance/driftbook/R/returns.R", "style": "dense", "target_lines": 6, "unit": "log_returns"}`
- verification: `{"air_idempotent": true, "completion_tokens": 63, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 278, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -11,6 +11,12 @@
   c(NA_real_, prices[-1] / prices[-length(prices)] - 1)
 }
 
+#' Log returns from a price series
+#'
+#' @param prices Numeric vector of prices in time order.
+#' @return A numeric vector the same length as `prices`, starting with `NA`.
+#' @family return calculators
+#' @export
 log_returns <- function(prices) {
   if (length(prices) < 2) {
     return(rep(NA_real_, length(prices)))
```

**Target (gold region)**

```r
#' Log returns from a price series
#'
#' @param prices Numeric vector of prices in time order.
#' @return A numeric vector the same length as `prices`, starting with `NA`.
#' @family return calculators
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/returns.R
#' Simple returns from a price series
#'
#' @param prices Numeric vector of prices in time order.
#' @return A numeric vector the same length as `prices`, starting with `NA`.
#' @family return calculators
#' @export
simple_returns <- function(prices) {
  if (length(prices) < 2) {
    return(rep(NA_real_, length(prices)))
  }
  c(NA_real_, prices[-1] / prices[-length(prices)] - 1)
}

<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
log_returns <- function(prices) {
  if (length(prices) < 2) {
    return(rep(NA_real_, length(prices)))
  }
  c(NA_real_, diff(log(prices)))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 43. `be392ce2550cbd622b4a7054` roxygen_drafting / roxy_tags

- package `synth:tillreport`, path `R/periods.R`
- generator: `{"determined_share": 0.75, "rule": "roxy_tags", "seed": "b01-tidyverse/tillreport/R/periods.R", "style": "dense", "target_lines": 4, "unit": "labelMonths"}`
- verification: `{"air_idempotent": true, "completion_tokens": 47, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 460, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -30,6 +30,10 @@
 
 #' Label monthly periods for reports
 #'
+#' @param starts A vector of period start dates.
+#' @return A character vector such as `"2026-02"`.
+#' @family period labels
+#' @export
 labelMonths <- function(starts) {
   starts <- lubridate::as_date(starts)
   sprintf("%d-%02d", lubridate::year(starts), lubridate::month(starts))
```

**Target (gold region)**

```r
#' @param starts A vector of period start dates.
#' @return A character vector such as `"2026-02"`.
#' @family period labels
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/periods.R
#' @export
completePeriods <- function(
  summary,
  periodCol = period,
  unit = c("month", "week", "quarter"),
  fill = list(revenue = 0, units = 0, transactions = 0L)
) {
  unit <- match.arg(unit)
  step <- switch(unit, week = "1 week", month = "1 month", quarter = "3 months")
  periods <- dplyr::pull(summary, {{ periodCol }})
  if (length(periods) == 0) {
    return(summary)
  }
  allPeriods <- seq(min(periods), max(periods), by = step)
  summary |>
    tidyr::complete({{ periodCol }} := allPeriods, fill = fill) |>
    dplyr::arrange({{ periodCol }})
}

#' Label weekly periods for reports
#'
#' @param starts A vector of period start dates.
#' @return A character vector such as `"2026-W07"`.
#' @family period labels
#' @export
labelWeeks <- function(starts) {
  starts <- lubridate::as_date(starts)
  sprintf("%d-W%02d", lubridate::isoyear(starts), lubridate::isoweek(starts))
}

#' Label monthly periods for reports
#'
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
labelMonths <- function(starts) {
  starts <- lubridate::as_date(starts)
  sprintf("%d-%02d", lubridate::year(starts), lubridate::month(starts))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 44. `641835bc7636f05b53c999a3` roxygen_drafting / roxy_tags

- package `synth:rendershed`, path `R/schedule.R`
- generator: `{"determined_share": 0.8, "rule": "roxy_tags", "seed": "b14-pipelines/rendershed/R/schedule.R", "style": "sectioned", "target_lines": 5, "unit": "nextRun"}`
- verification: `{"air_idempotent": true, "completion_tokens": 65, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 430, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -21,6 +21,11 @@
 #'
 #' Dispatches to the daily, weekly or monthly helper.
 #'
+#' @param cadence A cadence created by [parseCadence()], or a cadence string.
+#' @param from The reference time; the result is strictly after it.
+#' @return A `POSIXct` time stamp.
+#' @family scheduling
+#' @export
 nextRun <- function(cadence, from = Sys.time()) {
   if (is.character(cadence)) {
     cadence <- parseCadence(cadence)
```

**Target (gold region)**

```r
#' @param cadence A cadence created by [parseCadence()], or a cadence string.
#' @param from The reference time; the result is strictly after it.
#' @return A `POSIXct` time stamp.
#' @family scheduling
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/schedule.R
#' Next run time for a monthly cadence
#'
#' @param cadence A cadence created by [parseCadence()].
#' @param from The reference time; the result is strictly after it.
#' @return A `POSIXct` time stamp.
#' @family scheduling
#' @keywords internal
nextRunMonthly <- function(cadence, from = Sys.time()) {
  today <- as.Date(from, tz = cadence$tz)
  day <- as.integer(cadence$anchor)
  firstOfMonth <- as.Date(format(today, "%Y-%m-01"))
  candidate <- atTimeOfDay(firstOfMonth + day - 1L, cadence)
  if (candidate <= from) {
    nextMonth <- seq(firstOfMonth, by = "1 month", length.out = 2L)[2L]
    candidate <- atTimeOfDay(nextMonth + day - 1L, cadence)
  }
  candidate
}

#' Next run time for any cadence
#'
#' Dispatches to the daily, weekly or monthly helper.
#'
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
nextRun <- function(cadence, from = Sys.time()) {
  if (is.character(cadence)) {
    cadence <- parseCadence(cadence)
  }
  switch(
    cadence$type,
    daily = nextRunDaily(cadence, from),
    weekly = nextRunWeekly(cadence, from),
    monthly = nextRunMonthly(cadence, from)
  )
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 45. `bc8384a9595a7708ac069d08` roxygen_drafting / roxy_tags

- package `synth:taskrota`, path `R/queue.R`
- generator: `{"determined_share": 1.0, "rule": "roxy_tags", "seed": "b05-oop/taskrota/R/queue.R", "style": "dense", "target_lines": 1, "unit": "PriorityQueue"}`
- verification: `{"air_idempotent": true, "completion_tokens": 11, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 1304, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -3,6 +3,7 @@
 #' `PriorityQueue` keeps jobs ordered by priority, highest first. Jobs with
 #' equal priority are served in the order they were pushed.
 #'
+#' @export
 PriorityQueue <- R6::R6Class(
   "PriorityQueue",
   public = list(
```

**Target (gold region)**

```r
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/queue.R
#' A priority queue of jobs
#'
#' `PriorityQueue` keeps jobs ordered by priority, highest first. Jobs with
#' equal priority are served in the order they were pushed.
#'
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
PriorityQueue <- R6::R6Class(
  "PriorityQueue",
  public = list(
    #' @description
    #' Create an empty queue.
    #' @param capacity Maximum number of jobs the queue may hold.
    initialize = function(capacity = Inf) {
      stopifnot(is.numeric(capacity), length(capacity) == 1L, capacity > 0)
      private$.capacity <- capacity
      private$.jobs <- list()
      private$.seq <- integer()
      invisible(self)
    },
    #' @description
    #' Add a job to the queue.
    #' @param job A [Job] or anything accepted by [as_job()].
    #' @param ... Passed to [as_job()].
    #' @return The queue, invisibly, so calls can be chained.
    push = function(job, ...) {
      if (self$size >= private$.capacity) {
        stop("Queue is full (capacity ", private$.capacity, ").", call. = FALSE)
      }
      job <- as_job(job, ...)
      private$.counter <- private$.counter + 1L
      private$.jobs[[length(private$.jobs) + 1L]] <- job
      private$.seq <- c(private$.seq, private$.counter)
      private$reorder()
      invisible(self)
    },
    #' @description
    #' Remove and return the job with the highest priority.
    #' @return A [Job], or `NULL` when the queue is empty.
    pop = function() {
      if (self$is_empty) {
        return(NULL)
      }
      job <- private$.jobs[[1L]]
      private$.jobs <- private$.jobs[-1L]
      private$.seq <- private$.seq[-1L]
      job
    },
    #' @description
    #' Return the job with the highest priority without removing it.
    #' @return A [Job], or `NULL` when the queue is empty.
    peek = function() {
      if (self$is_empty) {
        return(NULL)
      }
      private$.jobs[[1L]]
    },
    #' @description
    #' Remove a job by identifier.
    #' @param id The identifier of the job to remove.
    #' @return The queue, invisibly.
    remove = function(id) {
      ids <- vapply(private$.jobs, function(j) j$id, character(1))
      keep <- ids != id
      if (all(keep)) {
        warning("No job with id '", id, "' in the queue.", call. = FALSE)
      }
      private$.jobs <- private$.jobs[keep]
      private$.seq <- private$.seq[keep]
      invisible(self)
    },
    #' @description
    #' Drop every job from the queue.
    #' @return The queue, invisibly.
    clear = function() {
      private$.jobs <- list()
      private$.seq <- integer()
      invisible(self)
    },
    #' @description
    #' Print the queue and its first few jobs.
    #' @param n Number of jobs to list.
    #' @param ... Ignored.
    print = function(n = 5L, ...) {
      cat("<PriorityQueue> ", self$size, " job(s)", sep = "")
      if (is.finite(private$.capacity)) {
        cat(" of ", private$.capacity, sep = "")
      }
      cat("\n")
      shown <- utils::head(private$.jobs, n)
      for (job in shown) {
        cat("  - ", job$name, " (priority ", job$priority, ")\n", sep = "")
      }
      if (self$size > n) {
        cat("  ... and ", self$size - n, " more\n", sep = "")
      }
      invisible(self)
    }
  ),
  private = list(
    .jobs = NULL,
    .seq = NULL,
    .capacity = Inf,
    .counter = 0L,
    reorder = function() {
      prios <- vapply(private$.jobs, function(j) j$priority, integer(1))
      ord <- order(-prios, private$.seq)
      private$.jobs <- private$.jobs[ord]
      private$.seq <- private$.seq[ord]
      invisible(NULL)
    }
  ),
  active = list(
    #' @field size Number of jobs currently queued.
    size = function(value) {
      if (!missing(value)) {
        stop("`size` is read-only.", call. = FALSE)
      }
      length(private$.jobs)
    },
    #' @field is_empty `TRUE` if the queue holds no jobs.
    is_empty = function(value) {
      if (!missing(value)) {
        stop("`is_empty` is read-only.", call. = FALSE)
      }
      length(private$.jobs) == 0L
    },
    #' @field jobs A copy of the queued jobs, in serving order.
    jobs = function(value) {
      if (!missing(value)) {
        stop("`jobs` is read-only; use `push()` and `pop()`.", call. = FALSE)
      }
      private$.jobs
    }
  )
)
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 46. `3555d5c6ed5e99f2d6349393` roxygen_drafting / roxy_tags

- package `synth:panelwick`, path `R/lags.R`
- generator: `{"determined_share": 1.0, "rule": "roxy_tags", "seed": "b09-finance/panelwick/R/lags.R", "style": "dense", "target_lines": 9, "unit": "diffBy"}`
- verification: `{"air_idempotent": true, "completion_tokens": 117, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 489, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -26,6 +26,15 @@
 
 #' Difference variables within each panel unit
 #'
+#' @param dt A data.table in long panel format.
+#' @param vars Character vector of columns to shift.
+#' @param id Name of the column identifying panel units.
+#' @param time Name of the column identifying time periods.
+#' @param n Number of periods to shift by.
+#' @return `dt`, keyed by `id` and `time` and modified by reference, invisibly.
+#' @family group shifts
+#' @importFrom data.table := .SD
+#' @export
 diffBy <- function(dt, vars, id, time, n = 1L) {
   checkPanelKeys(dt, id, time)
   data.table::setkeyv(dt, c(id, time))
```

**Target (gold region)**

```r
#' @param dt A data.table in long panel format.
#' @param vars Character vector of columns to shift.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @param n Number of periods to shift by.
#' @return `dt`, keyed by `id` and `time` and modified by reference, invisibly.
#' @family group shifts
#' @importFrom data.table := .SD
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/lags.R
#' Lead variables within each panel unit
#'
#' Rows are ordered by `time` within `id` before shifting, so the input
#' order does not matter.
#'
#' @param dt A data.table in long panel format.
#' @param vars Character vector of columns to shift.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @param n Number of periods to shift by.
#' @return `dt`, keyed by `id` and `time` and modified by reference, invisibly.
#' @family group shifts
#' @importFrom data.table := .SD
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
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
diffBy <- function(dt, vars, id, time, n = 1L) {
  checkPanelKeys(dt, id, time)
  data.table::setkeyv(dt, c(id, time))
  outCols <- paste0(vars, "_diff", n)
  dt[,
    (outCols) := lapply(.SD, function(v) v - data.table::shift(v, n)),
    by = id,
    .SDcols = vars
  ]
  invisible(dt)
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 47. `4b0745c648a9f84f0fbbe325` roxygen_drafting / roxy_tags

- package `synth:shiftplan`, path `R/assignment.R`
- generator: `{"determined_share": 1.0, "rule": "roxy_tags", "seed": "b19-graphs-or/shiftplan/R/assignment.R", "style": "dense", "target_lines": 4, "unit": "assignRegret"}`
- verification: `{"air_idempotent": true, "completion_tokens": 53, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 549, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -26,6 +26,10 @@
 #' At each step assigns the worker whose gap between cheapest and second
 #' cheapest remaining task is largest, so that costly misses happen less.
 #'
+#' @param cost A numeric matrix with workers in rows and tasks in columns.
+#' @return A data frame with columns `worker`, `task` and `cost`.
+#' @export
+#' @family assignment
 assignRegret <- function(cost) {
   cost <- checkCostMatrix(cost)
   n <- nrow(cost)
```

**Target (gold region)**

```r
#' @param cost A numeric matrix with workers in rows and tasks in columns.
#' @return A data frame with columns `worker`, `task` and `cost`.
#' @export
#' @family assignment
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/assignment.R
#' Greedy assignment by globally cheapest pair
#'
#' Repeatedly picks the cheapest remaining worker-task pair. Fast but not
#' optimal; useful as a starting point or a baseline.
#'
#' @param cost A numeric matrix with workers in rows and tasks in columns.
#' @return A data frame with columns `worker`, `task` and `cost`.
#' @export
#' @family assignment
assignGreedy <- function(cost) {
  cost <- checkCostMatrix(cost)
  n <- nrow(cost)
  assigned <- rep(NA_integer_, n)
  work <- cost
  for (step in seq_len(min(dim(cost)))) {
    idx <- which(work == min(work), arr.ind = TRUE)[1, ]
    assigned[idx[1]] <- idx[2]
    work[idx[1], ] <- Inf
    work[, idx[2]] <- Inf
  }
  return(assignmentTable(cost, assigned))
}

#' Greedy assignment by regret
#'
#' At each step assigns the worker whose gap between cheapest and second
#' cheapest remaining task is largest, so that costly misses happen less.
#'
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
assignRegret <- function(cost) {
  cost <- checkCostMatrix(cost)
  n <- nrow(cost)
  assigned <- rep(NA_integer_, n)
  work <- cost
  for (step in seq_len(min(dim(cost)))) {
    open <- which(is.na(assigned))
    regret <- vapply(
      open,
      function(i) {
        row <- sort(work[i, is.finite(work[i, ])])
        if (length(row) < 2) Inf else row[2] - row[1]
      },
      numeric(1)
    )
    i <- open[which.max(regret)]
    j <- which.min(work[i, ])
    assigned[i] <- j
    work[i, ] <- Inf
    work[, j] <- Inf
  }
  return(assignmentTable(cost, assigned))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 48. `b338f19aacf9f6f6d6ebb9ca` roxygen_drafting / roxy_tags

- package `synth:foldwise`, path `R/splitters.R`
- generator: `{"determined_share": 1.0, "rule": "roxy_tags", "seed": "b11-ml/foldwise/R/splitters.R", "style": "dense", "target_lines": 4, "unit": "logo_splits"}`
- verification: `{"air_idempotent": true, "completion_tokens": 47, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 551, "reward": 1.0, "reward_comparison": "air", "reward_exact": true, "reward_flags": []}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -32,6 +32,10 @@
 #'
 #' Creates one split per distinct group, holding that group out entirely.
 #'
+#' @param groups A vector of group identifiers, one per row.
+#' @return An object of class `fw_splits`.
+#' @family splitters
+#' @export
 logo_splits <- function(groups) {
   n <- length(groups)
   unique_groups <- unique(groups)
```

**Target (gold region)**

```r
#' @param groups A vector of group identifiers, one per row.
#' @return An object of class `fw_splits`.
#' @family splitters
#' @export
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/splitters.R
#' both the analysis and assessment sets of a split.
#'
#' @param groups A vector of group identifiers, one per row.
#' @param k Number of folds.
#' @param seed Optional integer seed for reproducible assignment.
#' @return An object of class `fw_splits`.
#' @family splitters
#' @export
grouped_splits <- function(groups, k = 5, seed = NULL) {
  n <- length(groups)
  unique_groups <- unique(groups)
  check_split_size(length(unique_groups), k)
  if (!is.null(seed)) {
    set.seed(seed)
  }
  # greedy balancing: biggest groups first, each into the lightest fold
  sizes <- table(groups)[as.character(unique_groups)]
  order_idx <- order(-as.integer(sizes), sample(length(sizes)))
  load <- integer(k)
  group_fold <- integer(length(unique_groups))
  for (j in order_idx) {
    target <- which.min(load)
    group_fold[j] <- target
    load[target] <- load[target] + sizes[[j]]
  }
  fold_id <- group_fold[match(groups, unique_groups)]
  holdout <- split(seq_len(n), factor(fold_id, levels = seq_len(k)))
  return(new_splits(unname(holdout), n, method = "grouped"))
}

#' Leave-one-group-out splits
#'
#' Creates one split per distinct group, holding that group out entirely.
#'
<filename>selected_references
<filename>edit_history
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
logo_splits <- function(groups) {
  n <- length(groups)
  unique_groups <- unique(groups)
  if (length(unique_groups) < 2L) {
    stop("Need at least two groups for leave-one-group-out.", call. = FALSE)
  }
  holdout <- lapply(unique_groups, function(g) which(groups == g))
  return(new_splits(holdout, n, method = "logo"))
}
<<<<<<< CURRENT
=======
<[fim-middle]>
````

</details>

## 49. `7c79e4be0176e05b85544230` format_propagation / fmt_one_line

- package `synth:visitkit`, path `R/sites.R`
- generator: `{"rule": "fmt_one_line", "seed": "b01-tidyverse/visitkit/R/sites.R", "style": "sectioned", "target_lines": 3}`
- verification: `{"air_idempotent": true, "completion_tokens": 43, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 1392, "reward": 1.0, "reward_comparison": "raw_formatting_target", "reward_exact": true, "reward_flags": ["formatting_only_reference"]}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -42,7 +42,9 @@
   if (length(paths) == 0) {
     stop("No site exports found in '", dir, "'.", call. = FALSE)
   }
-  results <- paths %>% purrr::set_names(basename(paths)) %>% purrr::map(purrr::safely(read.site.file))
+  results <- paths %>%
+    purrr::set_names(basename(paths)) %>%
+    purrr::map(purrr::safely(read.site.file))
   errors <- purrr::map(results, "error") %>% purrr::compact()
   if (length(errors) > 0) {
     warning("Skipped ", length(errors), " file(s): ", paste(names(errors), collapse = ", "), call. = FALSE)
```

**Target (gold region)**

```r
  results <- paths %>%
    purrr::set_names(basename(paths)) %>%
    purrr::map(purrr::safely(read.site.file))
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/sites.R
#' Read one site's visit export
#'
#' Reads the CSV export of a single site, standardises column names and
#' tags every row with the site identifier taken from the file name
#' (`site_<id>_visits.csv`).
#'
#' @param path Path to a site export file.
#' @return A tibble with a leading `site_id` column.
#' @export
read.site.file <- function(path) {
  site_id <- stringr::str_match(basename(path), "^site_([A-Za-z0-9]+)_")[, 2]
  if (is.na(site_id)) {
    stop(
      "Cannot find a site id in file name '",
      basename(path),
      "'.",
      call. = FALSE
    )
  }
  readr::read_csv(
    path,
    col_types = readr::cols(.default = "c"),
    progress = FALSE
  ) %>%
    dplyr::rename_with(
      ~ stringr::str_replace_all(stringr::str_to_lower(.x), "\\s+", "_")
    ) %>%
    dplyr::mutate(site_id = site_id, .before = 1)
}

#' Read and combine visit exports from every site
#'
#' Files that fail to read are skipped with a warning naming them, so one
#' corrupt export does not block the whole data cut.
#'
#' @param dir Directory containing the site export files.
#' @param pattern Regular expression matching export file names.
#' @return A tibble of all sites' visits, with a `"failed_files"` attribute.
#' @export
read.all.sites <- function(dir, pattern = "^site_.*_visits\\.csv$") {
  paths <- list.files(dir, pattern = pattern, full.names = TRUE)
  if (length(paths) == 0) {
    stop("No site exports found in '", dir, "'.", call. = FALSE)
  }
<filename>selected_references
<filename>edit_history
User edited "R/sites.R":

```diff
@@ -13,1 +13,6 @@
-    stop("Cannot find a site id in file name '", basename(path), "'.", call. = FALSE)
+    stop(
+      "Cannot find a site id in file name '",
+      basename(path),
+      "'.",
+      call. = FALSE
+    )
```
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
  errors <- purrr::map(results, "error") %>% purrr::compact()
  if (length(errors) > 0) {
    warning("Skipped ", length(errors), " file(s): ", paste(names(errors), collapse = ", "), call. = FALSE)
  }
  combined <- results %>% purrr::map("result") %>% purrr::compact() %>% purrr::list_rbind()
  attr(combined, "failed_files") <- names(errors)
  combined
}

#' Harmonise column types across site exports
#'
#' Applies a named list of type conversions, e.g.
#' `list(visit_date = parse.visit.date, aval = as.numeric)`, to the columns
#' that exist; missing columns are reported but not created.
#'
#' @param data A data frame of combined site visits.
#' @param types A named list of conversion functions, one per column.
#' @return `data` with converted columns.
#' @export
harmonise.types <- function(data, types) {
  absent <- setdiff(names(types), names(data))
  if (length(absent) > 0) {
    message("Columns not found, skipping: ", paste(absent, collapse = ", "))
  }
  present <- types[intersect(names(types), names(data))]
  purrr::reduce2(
    names(present),
    present,
    function(acc, col, fn) {
      dplyr::mutate(acc, dplyr::across(dplyr::all_of(col), fn))
    },
    .init = data
  )
}

#' Count visits per site
#'
#' @param data A data frame of combined site visits.
#' @param site <[`data-masking`][dplyr::dplyr_data_masking]> Column identifying the site.
#' @return A tibble with one row per site and an `n_visits` column.
#' @family site counts
#' @export
site.visit.counts <- function(data, site = site_id) {
  data %>%
    dplyr::count({{ site }}, name = "n_visits") %>%
    dplyr::arrange(dplyr::desc(.data$n_visits))
}

#' Count subjects per site
#'
#' @param data A data frame of combined site visits.
#' @param site <[`data-masking`][dplyr::dplyr_data_masking]> Column identifying the site.
#' @return A tibble with one row per site and an `n_subjects` column.
#' @family site counts
#' @export
site.subject.counts <- function(data, site = site_id) {
  data %>%
    dplyr::distinct({{ site }}, .data$subject_id) %>%
    dplyr::count({{ site }}, name = "n_subjects") %>%
    dplyr::arrange(dplyr::desc(.data$n_subjects))
}

#' Count out-of-window visits per site
#'
#' @param data A data frame of combined site visits with `early` and `late`
#'   columns from [flag.early.visits()] and [flag.late.visits()].
<<<<<<< CURRENT
  <|user_cursor|>results <- paths %>% purrr::set_names(basename(paths)) %>% purrr::map(purrr::safely(read.site.file))
=======
<[fim-middle]>
````

</details>

## 50. `9feced8a43a1e0b616e294fe` format_propagation / fmt_one_line

- package `synth:quantia`, path `R/accessors.R`
- generator: `{"rule": "fmt_one_line", "seed": "b05-oop/quantia/R/accessors.R", "style": "sectioned", "target_lines": 5}`
- verification: `{"air_idempotent": true, "completion_tokens": 46, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 1315, "reward": 1.0, "reward_comparison": "raw_formatting_target", "reward_exact": true, "reward_flags": ["formatting_only_reference"]}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -87,7 +87,11 @@
   if (length(used) == 0L) {
     return("1")
   }
-  parts <- ifelse(powers[used] == 1, abbrev[used], paste0(abbrev[used], "^", powers[used]))
+  parts <- ifelse(
+    powers[used] == 1,
+    abbrev[used],
+    paste0(abbrev[used], "^", powers[used])
+  )
   paste(parts, collapse = " ")
 }
 
```

**Target (gold region)**

```r
  parts <- ifelse(
    powers[used] == 1,
    abbrev[used],
    paste0(abbrev[used], "^", powers[used])
  )
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/accessors.R
#' Get the magnitudes of a quantity
#'
#' Returns the bare numeric values of a quantity, dropping its unit.
#'
#' @param object A [Quantity-class] object.
#' @param ... Additional arguments for methods.
#' @return A numeric vector.
#' @export
setGeneric("magnitude", function(object, ...) standardGeneric("magnitude"))

#' Set the magnitudes of a quantity
#'
#' @param object A [Quantity-class] object.
#' @param value A numeric vector of the same length as `object`.
#' @return The modified object.
#' @export
setGeneric("magnitude<-", function(object, value) {
  standardGeneric("magnitude<-")
})

#' Get the unit of a quantity
#'
#' @param object A [Quantity-class] object.
#' @param ... Additional arguments for methods.
#' @return A [Unit-class] object.
#' @export
setGeneric("unitOf", function(object, ...) standardGeneric("unitOf"))

#' Set the unit of a quantity without converting
#'
#' Replaces the unit but keeps the magnitudes unchanged. Use [convertTo()]
#' to change units while preserving the physical value.
#'
#' @param object A [Quantity-class] object.
#' @param value A [Unit-class] object or a registered unit symbol.
#' @return The modified object.
#' @export
setGeneric("unitOf<-", function(object, value) standardGeneric("unitOf<-"))

#' @rdname magnitude
setMethod("magnitude", "Quantity", function(object, ...) {
  object@.Data
})

#' @rdname magnitude-set
setReplaceMethod("magnitude", "Quantity", function(object, value) {
  if (!is.numeric(value) || length(value) != length(object)) {
    stop(
      "`value` must be numeric with length ",
      length(object),
      ".",
      call. = FALSE
    )
  }
  object@.Data <- as.numeric(value)
  validObject(object)
  object
})

#' @rdname unitOf
setMethod("unitOf", "Quantity", function(object, ...) {
  object@unit
})

#' @rdname unitOf-set
setReplaceMethod("unitOf", "Quantity", function(object, value) {
  if (is.character(value)) {
    value <- lookupUnit(value)
  }
  object@unit <- value
  validObject(object)
  object
})

#' Format a unit's dimensions
#'
#' Writes the non-zero exponents of a unit as a compact string such as
#' `"L M T^-2"`.
#'
#' @param unit A [Unit-class] object.
#' @return A single string, `"1"` for dimensionless units.
#' @keywords internal
formatDims <- function(unit) {
  abbrev <- c("L", "M", "T", "I", "K", "N", "J")
  powers <- unit@dims
  used <- which(powers != 0)
  if (length(used) == 0L) {
    return("1")
  }
<filename>selected_references
<filename>edit_history
User edited "R/accessors.R":

```diff
@@ -48,1 +48,6 @@
-    stop("`value` must be numeric with length ", length(object), ".", call. = FALSE)
+    stop(
+      "`value` must be numeric with length ",
+      length(object),
+      ".",
+      call. = FALSE
+    )
```
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
  paste(parts, collapse = " ")
}

#' @rdname Unit-class
#' @param object A [Unit-class] object.
setMethod("show", "Unit", function(object) {
  cat("<Unit> ", object@symbol, " [", formatDims(object), "]", sep = "")
  if (object@scale != 1) {
    cat(" x ", format(object@scale, digits = 6), sep = "")
  }
  if (object@offset != 0) {
    cat(" + ", format(object@offset, digits = 6), sep = "")
  }
  cat("\n")
  invisible(object)
})

#' @rdname Quantity-class
#' @param object A [Quantity-class] object.
setMethod("show", "Quantity", function(object) {
  n <- length(object)
  cat("<Quantity> ", n, " value(s) in ", object@unit@symbol, "\n", sep = "")
  if (n == 0L) {
    return(invisible(object))
  }
  shown <- utils::head(object@.Data, 10L)
  print(paste(format(shown, digits = 4), object@unit@symbol), quote = FALSE)
  if (n > 10L) {
    cat("... ", n - 10L, " more\n", sep = "")
  }
  invisible(object)
})

#' @rdname Quantity-class
#' @param x A [Quantity-class] object.
#' @param i Indices of the elements to extract.
#' @param ... Ignored.
#' @param drop Ignored.
setMethod("[", "Quantity", function(x, i, ..., drop = TRUE) {
  new("Quantity", x@.Data[i], unit = x@unit)
})
<<<<<<< CURRENT
  <|user_cursor|>parts <- ifelse(powers[used] == 1, abbrev[used], paste0(abbrev[used], "^", powers[used]))
=======
<[fim-middle]>
````

</details>

## 51. `7b93c4fb451dfe5c1c749eb9` format_propagation / fmt_one_line

- package `synth:relaydesk`, path `plumber.R`
- generator: `{"rule": "fmt_one_line", "seed": "b18-web/relaydesk/plumber.R", "style": "dense", "target_lines": 5}`
- verification: `{"air_idempotent": true, "completion_tokens": 40, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 1438, "reward": 1.0, "reward_comparison": "raw_formatting_target", "reward_exact": true, "reward_flags": ["formatting_only_reference"]}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -130,7 +130,11 @@
   deliveryId <- req$HTTP_X_DELIVERY_ID %||%
     paste(source, event$externalId, sep = ":")
   if (!recordDelivery(con, deliveryId, source)) {
-    return(sendOk(res, list(status = "duplicate", deliveryId = deliveryId), status = 200L))
+    return(sendOk(
+      res,
+      list(status = "duplicate", deliveryId = deliveryId),
+      status = 200L
+    ))
   }
   ticket <- insertTicket(
     con,
```

**Target (gold region)**

```r
    return(sendOk(
      res,
      list(status = "duplicate", deliveryId = deliveryId),
      status = 200L
    ))
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>plumber.R
#* @filter apiKey
function(req, res) {
  path <- req$PATH_INFO
  if (path == "/health" || startsWith(path, "/webhooks/")) {
    return(forward())
  }
  auth <- req$HTTP_AUTHORIZATION %||% ""
  key <- sub("^Bearer\\s+", "", auth)
  if (!nzchar(key) || !key %in% apiKeys) {
    return(sendUnauthorized(res))
  }
  forward()
}

#* Report service health
#* @get /health
#* @serializer unboxedJSON
function(res) {
  dbOk <- tryCatch(
    {
      DBI::dbGetQuery(con, "SELECT 1")
      TRUE
    },
    error = function(e) FALSE
  )
  if (!dbOk) {
    return(sendError(res, 503L, "unavailable", "Database is not reachable."))
  }
  sendOk(res, list(status = "ok", time = storeNow()))
}

#* List tickets, optionally filtered by status and queue
#* @param status:str Filter by ticket status
#* @param queue:str Filter by queue name
#* @param page:int Page number, starting at 1
#* @param perPage:int Number of tickets per page
#* @get /tickets
#* @serializer unboxedJSON
function(res, status = NULL, queue = NULL, page = NULL, perPage = NULL) {
  paging <- validatePaging(page, perPage)
  if (!paging$ok) {
    return(sendInvalid(res, paging$errors))
  }
  if (!is.null(status)) {
    checked <- requireChoice(
      status,
      "status",
      c("open", "in_progress", "waiting", "closed")
    )
    if (!is.null(checked$error)) {
      return(sendInvalid(res, c(status = checked$error)))
    }
    status <- checked$value
  }
  tickets <- listTickets(con, status = status, queue = queue)
  sendOk(
    res,
    paginate(tickets, paging$data$page, paging$data$perPage, "/tickets")
  )
}

#* Fetch a single ticket
#* @param id:int The ticket id
#* @get /tickets/<id:int>
#* @serializer unboxedJSON
function(res, id) {
  tryCatch(
    sendOk(res, getTicket(con, id)),
    error = function(e) sendCondition(res, e)
  )
}

#* Create a ticket
#* @post /tickets
#* @serializer unboxedJSON
function(req, res) {
  body <- req$body
  checked <- validateTicketInput(body)
  if (!checked$ok) {
    return(sendInvalid(res, checked$errors))
  }
  tryCatch(
    {
      ticket <- insertTicket(con, checked$data)
      sendCreated(res, ticket, sprintf("/tickets/%d", ticket$id))
    },
    error = function(e) sendCondition(res, e)
  )
}

#* Change the status of a ticket
#* @param id:int The ticket id
#* @patch /tickets/<id:int>/status
#* @serializer unboxedJSON
function(req, res, id) {
  checked <- requireChoice(
    req$body$status,
    "status",
    c("open", "in_progress", "waiting", "closed")
  )
  if (!is.null(checked$error)) {
    return(sendInvalid(res, c(status = checked$error)))
  }
  tryCatch(
    sendOk(res, updateTicketStatus(con, id, checked$value)),
    error = function(e) sendCondition(res, e)
  )
}

#* Receive a webhook from an external service
#* @param source:str One of git, form or alert
#* @post /webhooks/<source>
#* @serializer unboxedJSON
function(req, res, source) {
  if (!source %in% names(webhookSecrets)) {
    return(sendNotFound(res, sprintf("Webhook source '%s'", source)))
  }
  result <- handleWebhook(source, req, webhookSecrets)
  if (result$status == "rejected") {
    return(sendError(res, 401L, "rejected", result$reason))
  }
  if (result$status == "ignored") {
    return(sendOk(
      res,
      list(status = "ignored", reason = result$reason),
      status = 202L
    ))
  }
  event <- result$event
  deliveryId <- req$HTTP_X_DELIVERY_ID %||%
    paste(source, event$externalId, sep = ":")
  if (!recordDelivery(con, deliveryId, source)) {
<filename>selected_references
<filename>edit_history
User edited "plumber.R":

```diff
@@ -161,1 +161,2 @@
-  deliveryId <- req$HTTP_X_DELIVERY_ID %||% paste(source, event$externalId, sep = ":")
+  deliveryId <- req$HTTP_X_DELIVERY_ID %||%
+    paste(source, event$externalId, sep = ":")
```
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
  }
  ticket <- insertTicket(
    con,
    list(
      title = substr(event$title, 1, 120),
      description = paste("Imported from", event$url %||% source),
      reporter = event$actor,
      priority = if (event$kind == "incident") "urgent" else "normal",
      queue = source,
      tags = c(source, event$kind)
    )
  )
  sendCreated(res, ticket, sprintf("/tickets/%d", ticket$id))
}
<<<<<<< CURRENT
    <|user_cursor|>return(sendOk(res, list(status = "duplicate", deliveryId = deliveryId), status = 200L))
=======
<[fim-middle]>
````

</details>

## 52. `7951912acb1c75ee775fef03` format_propagation / fmt_one_line

- package `synth:burrowlog`, path `tests/testthat/test-messages.R`
- generator: `{"rule": "fmt_one_line", "seed": "b06-infra-tests/burrowlog/tests/testthat/test-messages.R", "style": "sectioned", "target_lines": 4}`
- verification: `{"air_idempotent": true, "completion_tokens": 40, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 1109, "reward": 1.0, "reward_comparison": "raw_formatting_target", "reward_exact": true, "reward_flags": ["formatting_only_reference"]}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -42,7 +42,10 @@
 test_that("burrow_map() returns a named list like lapply()", {
   withr::local_options(burrowlog.quiet = TRUE)
   x <- c(a = 1, b = 2, c = 3)
-  expect_identical(burrow_map(x, function(v) v * 10), lapply(x, function(v) v * 10))
+  expect_identical(
+    burrow_map(x, function(v) v * 10),
+    lapply(x, function(v) v * 10)
+  )
   expect_identical(burrow_map(1:2, ~ .x + 1L), list(2L, 3L))
   expect_identical(burrow_map(list(), identity), list())
   expect_identical(burrow_map(1:2, rep, times = 2), list(c(1L, 1L), c(2L, 2L)))
```

**Target (gold region)**

```r
  expect_identical(
    burrow_map(x, function(v) v * 10),
    lapply(x, function(v) v * 10)
  )
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>tests/testthat/test-messages.R
test_that("message helpers are classed and respect quiet mode", {
  withr::local_options(burrowlog.quiet = FALSE)
  expect_message(
    burrow_inform("Loaded {3} files."),
    class = "burrowlog_message"
  )
  expect_message(burrow_success("Done."), class = "burrowlog_message")
  expect_message(
    burrow_warn("Skipped {.file a.csv}."),
    class = "burrowlog_message"
  )
  withr::local_options(burrowlog.quiet = TRUE)
  expect_silent(burrow_inform("Loaded {3} files."))
  expect_silent(burrow_success("Done."))
  expect_silent(burrow_warn("Skipped {.file a.csv}."))
})

test_that("message helpers interpolate in the caller's frame", {
  withr::local_options(burrowlog.quiet = FALSE)
  n_rows <- 42
  expect_snapshot({
    burrow_inform("Read {n_rows} rows.")
    burrow_success("Wrote {n_rows} rows.")
    burrow_warn("Dropped {n_rows} rows.")
  })
})

test_that("format_summary_bullets() aligns labels and escapes braces", {
  out <- format_summary_bullets(c(rows = 1200, dropped = 3))
  expect_named(out, c("*", "*"))
  expect_identical(unname(out), c("rows    : 1,200", "dropped : 3"))
  out <- format_summary_bullets(c(pattern = "{x}"), bullet = "i")
  expect_identical(unname(out), "pattern : {{x}}")
  expect_identical(format_summary_bullets(numeric()), character())
  expect_error(
    format_summary_bullets(c(1, 2)),
    class = "burrowlog_error_unnamed"
  )
  expect_error(format_summary_bullets(c(a = 1), bullet = "?"))
})

test_that("burrow_map() returns a named list like lapply()", {
  withr::local_options(burrowlog.quiet = TRUE)
  x <- c(a = 1, b = 2, c = 3)
<filename>selected_references
<filename>edit_history
User edited "tests/testthat/test-messages.R":

```diff
@@ -45,1 +45,4 @@
-  expect_error(format_summary_bullets(c(1, 2)), class = "burrowlog_error_unnamed")
+  expect_error(
+    format_summary_bullets(c(1, 2)),
+    class = "burrowlog_error_unnamed"
+  )
```
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
  expect_identical(burrow_map(1:2, ~ .x + 1L), list(2L, 3L))
  expect_identical(burrow_map(list(), identity), list())
  expect_identical(burrow_map(1:2, rep, times = 2), list(c(1L, 1L), c(2L, 2L)))
})

test_that("burrow_map() reports the failing index", {
  withr::local_options(burrowlog.quiet = TRUE)
  fail_on_two <- function(v) if (v == 2) stop("bad value") else v
  err <- expect_error(burrow_map(1:3, fail_on_two), class = "burrowlog_error_map")
  expect_identical(err$index, 2L)
  expect_match(conditionMessage(err$parent), "bad value")
})

test_that("burrow_map() shows progress when not quiet", {
  skip_if_not_installed("cli", minimum_version = "3.6.0")
  withr::local_options(burrowlog.quiet = FALSE, cli.progress_show_after = 0, cli.dynamic = FALSE)
  expect_no_error(burrow_map(1:3, identity, label = "Mapping"))
})

test_that("with_step() returns the value of its code", {
  withr::local_options(burrowlog.quiet = TRUE)
  expect_identical(with_step("Counting", 1 + 1), 2)
  withr::local_options(burrowlog.quiet = FALSE)
  expect_message(value <- with_step("Counting", nrow(mtcars)), "took")
  expect_identical(value, 32L)
})
<<<<<<< CURRENT
  <|user_cursor|>expect_identical(burrow_map(x, function(v) v * 10), lapply(x, function(v) v * 10))
=======
<[fim-middle]>
````

</details>

## 53. `5e06b4125202d252f6118244` format_propagation / fmt_one_line

- package `synth:pkgsmith`, path `R/dependencies.R`
- generator: `{"rule": "fmt_one_line", "seed": "b22-devtools/pkgsmith/R/dependencies.R", "style": "dense", "target_lines": 6}`
- verification: `{"air_idempotent": true, "completion_tokens": 36, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 1373, "reward": 1.0, "reward_comparison": "raw_formatting_target", "reward_exact": true, "reward_flags": ["formatting_only_reference"]}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -101,7 +101,12 @@
 #' @family dependency helpers
 #' @export
 use_linking_to <- function(package, min_version = NULL, path = ".") {
-  changed <- deps_add(package, "LinkingTo", min_version = min_version, path = path)
+  changed <- deps_add(
+    package,
+    "LinkingTo",
+    min_version = min_version,
+    path = path
+  )
   if (changed) {
     message("Added '", package, "' to LinkingTo.")
   }
```

**Target (gold region)**

```r
  changed <- deps_add(
    package,
    "LinkingTo",
    min_version = min_version,
    path = path
  )
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/dependencies.R
#' List all dependencies declared in a DESCRIPTION
#'
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return A data frame of dependencies as returned by [deps_parse()].
#' @export
deps_list <- function(path = ".") {
  types <- c("Depends", "Imports", "Suggests", "LinkingTo")
  parts <- vector("list", length(types))
  for (i in seq_along(types)) {
    raw <- desc_get_field(types[[i]], path = path)
    parts[[i]] <- deps_parse(raw, types[[i]])
  }
  out <- do.call(rbind, parts)
  rownames(out) <- NULL
  return(out)
}

#' Add or update a dependency in one DESCRIPTION field
#'
#' @param package Name of the package to depend on.
#' @param type The DESCRIPTION field to modify.
#' @param min_version Optional minimum version, recorded as `>=`.
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return `TRUE` if the DESCRIPTION changed, invisibly.
#' @keywords internal
deps_add <- function(package, type, min_version = NULL, path = ".") {
  current <- deps_parse(desc_get_field(type, path = path), type)
  op <- if (is.null(min_version)) NA_character_ else ">="
  ver <- if (is.null(min_version)) NA_character_ else as.character(min_version)
  hit <- match(package, current$package)
  if (!is.na(hit)) {
    same <- identical(current$op[[hit]], op) &&
      identical(current$version[[hit]], ver)
    if (same || is.null(min_version)) {
      return(invisible(FALSE))
    }
    current$op[[hit]] <- op
    current$version[[hit]] <- ver
  } else {
    row <- data.frame(
      type = type,
      package = package,
      op = op,
      version = ver,
      stringsAsFactors = FALSE
    )
    current <- rbind(current, row)
  }
  desc_set_field(type, deps_format(current), path = path)
  return(invisible(TRUE))
}

#' Declare a package in Imports
#'
#' @param package Name of the package to depend on.
#' @param min_version Optional minimum version, recorded as `>=`.
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return `TRUE` if the DESCRIPTION changed, invisibly.
#' @family dependency helpers
#' @export
use_import <- function(package, min_version = NULL, path = ".") {
  changed <- deps_add(
    package,
    "Imports",
    min_version = min_version,
    path = path
  )
  if (changed) {
    message("Added '", package, "' to Imports.")
  }
  return(invisible(changed))
}

#' Declare a package in Suggests
#'
#' @param package Name of the package to depend on.
#' @param min_version Optional minimum version, recorded as `>=`.
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return `TRUE` if the DESCRIPTION changed, invisibly.
#' @family dependency helpers
#' @export
use_suggest <- function(package, min_version = NULL, path = ".") {
  changed <- deps_add(
    package,
    "Suggests",
    min_version = min_version,
    path = path
  )
  if (changed) {
    message("Added '", package, "' to Suggests.")
  }
  return(invisible(changed))
}

#' Declare a package in LinkingTo
#'
#' @param package Name of the package to depend on.
#' @param min_version Optional minimum version, recorded as `>=`.
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return `TRUE` if the DESCRIPTION changed, invisibly.
#' @family dependency helpers
#' @export
use_linking_to <- function(package, min_version = NULL, path = ".") {
<filename>selected_references
<filename>edit_history
User edited "R/dependencies.R":

```diff
@@ -146,1 +146,6 @@
-  changed <- deps_add(package, "Suggests", min_version = min_version, path = path)
+  changed <- deps_add(
+    package,
+    "Suggests",
+    min_version = min_version,
+    path = path
+  )
```
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
  if (changed) {
    message("Added '", package, "' to LinkingTo.")
  }
  return(invisible(changed))
}

#' Find packages declared in more than one dependency field
#'
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return A data frame with columns `package` and `types`, one row per
#'   duplicated package.
#' @export
deps_duplicates <- function(path = ".") {
  deps <- deps_list(path)
  deps <- deps[deps$package != "R", , drop = FALSE]
  counts <- table(deps$package)
  dupes <- names(counts)[counts > 1L]
  types <- character(length(dupes))
  for (i in seq_along(dupes)) {
    types[[i]] <- paste(deps$type[deps$package == dupes[[i]]], collapse = ", ")
  }
  return(data.frame(package = dupes, types = types, stringsAsFactors = FALSE))
}
<<<<<<< CURRENT
  <|user_cursor|>changed <- deps_add(package, "LinkingTo", min_version = min_version, path = path)
=======
<[fim-middle]>
````

</details>

## 54. `8ba0531e817d701cd4494bf2` format_propagation / fmt_one_line

- package `synth:termsmith`, path `R/compare-models.R`
- generator: `{"rule": "fmt_one_line", "seed": "b24-modelling/termsmith/R/compare-models.R", "style": "dense", "target_lines": 8}`
- verification: `{"air_idempotent": true, "completion_tokens": 46, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 2022, "reward": 1.0, "reward_comparison": "raw_formatting_target", "reward_exact": true, "reward_flags": ["formatting_only_reference"]}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -118,7 +118,14 @@
   n0 <- stats::nobs(restricted)
   n1 <- stats::nobs(full)
   if (n0 != n1) {
-    stop(sprintf("Models use different numbers of observations (%d vs %d).", n0, n1), call. = FALSE)
+    stop(
+      sprintf(
+        "Models use different numbers of observations (%d vs %d).",
+        n0,
+        n1
+      ),
+      call. = FALSE
+    )
   }
   stat <- 2 * (as.numeric(ll1) - as.numeric(ll0))
   if (stat < 0) {
```

**Target (gold region)**

```r
    stop(
      sprintf(
        "Models use different numbers of observations (%d vs %d).",
        n0,
        n1
      ),
      call. = FALSE
    )
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/compare-models.R
#' @return A single numeric value.
#' @keywords internal
aicc <- function(object) {
  ll <- stats::logLik(object)
  k <- attr(ll, "df")
  n <- attr(ll, "nobs")
  if (is.null(n)) {
    n <- stats::nobs(object)
  }
  aic <- -2 * as.numeric(ll) + 2 * k
  if (n - k - 1 <= 0) {
    warning("AICc is undefined for n <= k + 1; returning AIC.", call. = FALSE)
    return(aic)
  }
  aic + (2 * k * (k + 1)) / (n - k - 1)
}

#' Akaike weights
#'
#' Converts a vector of information criteria into relative model weights that
#' sum to one.
#'
#' @param ic A numeric vector of information criterion values.
#' @return A numeric vector the same length as `ic`.
#' @export
akaike_weights <- function(ic) {
  if (!is.numeric(ic) || length(ic) == 0L) {
    stop("`ic` must be a non-empty numeric vector.", call. = FALSE)
  }
  if (anyNA(ic)) {
    stop("`ic` must not contain missing values.", call. = FALSE)
  }
  delta <- ic - min(ic)
  rel <- exp(-0.5 * delta)
  rel / sum(rel)
}

#' Compare fitted models by an information criterion
#'
#' Fits nothing; it collects log-likelihoods and information criteria from
#' already fitted models and orders them from best to worst.
#'
#' @param ... Two or more fitted model objects, optionally named.
#' @param .names Optional character vector of model labels.
#' @param criterion Which information criterion to rank by.
#' @return A data frame of class `ts_ic_table`, one row per model.
#' @export
ic_table <- function(..., .names = NULL, criterion = c("AIC", "BIC", "AICc")) {
  criterion <- match.arg(criterion)
  models <- list(...)
  if (length(models) < 2L) {
    stop("Supply at least two fitted models.", call. = FALSE)
  }
  if (is.null(.names)) {
    .names <- names(models)
    if (is.null(.names) || any(.names == "")) {
      .names <- paste0("model_", seq_along(models))
    }
  }
  n_models <- length(models)
  ic <- numeric(n_models)
  df <- integer(n_models)
  ll <- numeric(n_models)
  for (i in seq_len(n_models)) {
    fit <- models[[i]]
    loglik <- stats::logLik(fit)
    ll[i] <- as.numeric(loglik)
    df[i] <- as.integer(attr(loglik, "df"))
    ic[i] <- switch(
      criterion,
      AIC = stats::AIC(fit),
      BIC = stats::BIC(fit),
      AICc = aicc(fit)
    )
  }
  out <- data.frame(
    model = .names,
    df = df,
    logLik = ll,
    ic = ic,
    delta = ic - min(ic),
    weight = akaike_weights(ic),
    stringsAsFactors = FALSE
  )
  out <- out[order(out$ic), , drop = FALSE]
  rownames(out) <- NULL
  attr(out, "criterion") <- criterion
  class(out) <- c("ts_ic_table", "data.frame")
  out
}

#' Likelihood-ratio test for nested models
#'
#' Compares a restricted model against a fuller one fitted to the same
#' observations, using the asymptotic chi-squared reference distribution.
#'
#' @param restricted The smaller, nested fitted model.
#' @param full The larger fitted model.
#' @return An object of class `ts_lr_test`.
#' @export
lr_test <- function(restricted, full) {
  if (!identical(class(restricted), class(full))) {
    warning(
      sprintf(
        "Comparing models of different classes (%s vs %s).",
        class(restricted)[1L],
        class(full)[1L]
      ),
      call. = FALSE
    )
  }
  ll0 <- stats::logLik(restricted)
  ll1 <- stats::logLik(full)
  df_diff <- attr(ll1, "df") - attr(ll0, "df")
  if (df_diff <= 0) {
    stop("`full` must have more parameters than `restricted`.", call. = FALSE)
  }
  n0 <- stats::nobs(restricted)
  n1 <- stats::nobs(full)
  if (n0 != n1) {
<filename>selected_references
<filename>edit_history
User edited "R/compare-models.R":

```diff
@@ -75,1 +75,6 @@
-    ic[i] <- switch(criterion, AIC = stats::AIC(fit), BIC = stats::BIC(fit), AICc = aicc(fit))
+    ic[i] <- switch(
+      criterion,
+      AIC = stats::AIC(fit),
+      BIC = stats::BIC(fit),
+      AICc = aicc(fit)
+    )
```
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
  }
  stat <- 2 * (as.numeric(ll1) - as.numeric(ll0))
  if (stat < 0) {
    warning("Negative LR statistic; are the models really nested?", call. = FALSE)
    stat <- 0
  }
  p_value <- stats::pchisq(stat, df = df_diff, lower.tail = FALSE)
  structure(list(statistic = stat, df = df_diff, p.value = p_value, nobs = n1), class = "ts_lr_test")
}

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
    stop("`models` must be a list of at least two fitted models.", call. = FALSE)
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
  out$p.adjusted <- stats::p.adjust(out$p.value, method = "holm")
  out
}

#' Format a model comparison table
#'
#' Rounds the numeric columns and relabels the criterion column for display.
#'
#' @param x An object of class `ts_ic_table`.
#' @param digits Number of decimal places for numeric columns.
#' @param ... Unused.
#' @return A plain data frame of character columns.
#' @export
format.ts_ic_table <- function(x, digits = 2, ...) {
  out <- x
  class(out) <- "data.frame"
  for (col in c("logLik", "ic", "delta")) {
    out[[col]] <- formatC(out[[col]], format = "f", digits = digits)
  }
  out$weight <- formatC(out$weight, format = "f", digits = 3)
  names(out)[names(out) == "ic"] <- attr(x, "criterion")
  out
}

#' Print a model comparison table
#'
#' Prints a header naming the criterion followed by the formatted table.
#'
#' @param x An object of class `ts_ic_table`.
#' @param digits Number of decimal places for numeric columns.
#' @param ... Passed to [print.data.frame()].
<<<<<<< CURRENT
    <|user_cursor|>stop(sprintf("Models use different numbers of observations (%d vs %d).", n0, n1), call. = FALSE)
=======
<[fim-middle]>
````

</details>

## 55. `0324049c865bd43c76e57c7e` format_propagation / fmt_aligned

- package `synth:inkplate`, path `R/table.R`
- generator: `{"rule": "fmt_aligned", "seed": "b10-text/inkplate/R/table.R", "style": "dense", "target_lines": 7}`
- verification: `{"air_idempotent": true, "completion_tokens": 42, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 1778, "reward": 1.0, "reward_comparison": "raw_formatting_target", "reward_exact": true, "reward_flags": ["formatting_only_reference"]}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -100,11 +100,13 @@
   cells <- format.cells(data, digits = digits)
   cells[] <- escape.html(cells)
   css <- c(l = "left", c = "center", r = "right")[align]
-  head <- paste0("<th style=\"text-align:",
-                 css,
-                 "\">",
-                 escape.html(names(data)),
-                 "</th>")
+  head <- paste0(
+    "<th style=\"text-align:",
+    css,
+    "\">",
+    escape.html(names(data)),
+    "</th>"
+  )
   rows <- apply(cells, 1, function(row) {
     tds <- paste0("<td style=\"text-align:", css, "\">", row, "</td>")
     paste0("<tr>", paste(tds, collapse = ""), "</tr>")
```

**Target (gold region)**

```r
  head <- paste0(
    "<th style=\"text-align:",
    css,
    "\">",
    escape.html(names(data)),
    "</th>"
  )
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/table.R
#' Format data frame columns as character
#'
#' Numbers are rounded to `digits` and formatted without scientific
#' notation; missing values become `na`.
#'
#' @param data A data frame to render.
#' @param digits Number of decimal places for numeric columns.
#' @param na String used for missing values.
#' @return A character matrix with the column names of `data`.
#' @noRd
format.cells <- function(data, digits = 2L, na = "") {
  cells <- vapply(
    data,
    function(col) {
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
  cells <- matrix(cells, nrow = nrow(data), dimnames = list(NULL, names(data)))
  cells
}

#' Choose a default alignment for each column
#'
#' @param data A data frame to render.
#' @return A character vector of `"l"` or `"r"`, one per column.
#' @noRd
default.align <- function(data) {
  vapply(
    data,
    function(col) if (is.numeric(col)) "r" else "l",
    character(1),
    USE.NAMES = FALSE
  )
}

#' Render a data frame as a Markdown pipe table
#'
#' @param data A data frame to render.
#' @param align Optional character vector of `"l"`, `"c"` or `"r"`, one per
#'   column.
#' @param digits Number of decimal places for numeric columns.
#' @return A single string containing the table.
#' @family table renderers
#' @export
table.markdown <- function(data, align = NULL, digits = 2L) {
  align <- if (is.null(align)) default.align(data) else align
  cells <- format.cells(data, digits = digits)
  cells[] <- escape.markdown(cells)
  header <- escape.markdown(names(data))
  widths <- pmax(
    nchar(header),
    apply(rbind(cells, ""), 2, function(c) max(nchar(c))),
    3L
  )
  rule <- vapply(
    seq_along(widths),
    function(j) {
      dashes <- strrep("-", widths[j] - 1L)
      switch(
        align[j],
        l = paste0(":", dashes),
        r = paste0(dashes, ":"),
        c = paste0(":", strrep("-", widths[j] - 2L), ":")
      )
    },
    character(1)
  )
  pad.row <- function(row) {
    paste0(
      "| ",
      paste(formatC(row, width = widths, flag = "-"), collapse = " | "),
      " |"
    )
  }
  body <- apply(cells, 1, pad.row)
  paste(
    c(pad.row(header), paste0("| ", paste(rule, collapse = " | "), " |"), body),
    collapse = "\n"
  )
}

#' Render a data frame as an HTML table
#'
#' @param data A data frame to render.
#' @param align Optional character vector of `"l"`, `"c"` or `"r"`, one per
#'   column.
#' @param digits Number of decimal places for numeric columns.
#' @return A single string containing the table.
#' @family table renderers
#' @export
table.html <- function(data, align = NULL, digits = 2L) {
  align <- if (is.null(align)) default.align(data) else align
  cells <- format.cells(data, digits = digits)
  cells[] <- escape.html(cells)
  css <- c(l = "left", c = "center", r = "right")[align]
<filename>selected_references
<filename>edit_history
User edited "R/table.R":

```diff
@@ -83,2 +83,4 @@
-  paste(c(pad.row(header), paste0("| ", paste(rule, collapse = " | "), " |"), body),
-        collapse = "\n")
+  paste(
+    c(pad.row(header), paste0("| ", paste(rule, collapse = " | "), " |"), body),
+    collapse = "\n"
+  )
```
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
  rows <- apply(cells, 1, function(row) {
    tds <- paste0("<td style=\"text-align:", css, "\">", row, "</td>")
    paste0("<tr>", paste(tds, collapse = ""), "</tr>")
  })
  paste(c(
          "<table>",
          paste0("<thead><tr>", paste(head, collapse = ""), "</tr></thead>"),
          "<tbody>",
          rows,
          "</tbody>",
          "</table>"
        ),
        collapse = "\n")
}

#' Render a data frame as a LaTeX tabular
#'
#' @param data A data frame to render.
#' @param align Optional character vector of `"l"`, `"c"` or `"r"`, one per
#'   column.
#' @param digits Number of decimal places for numeric columns.
#' @return A single string containing the table.
#' @family table renderers
#' @export
table.latex <- function(data, align = NULL, digits = 2L) {
  align <- if (is.null(align)) default.align(data) else align
  cells <- format.cells(data, digits = digits)
  cells[] <- escape.latex(cells)
  header <- paste(escape.latex(names(data)), collapse = " & ")
  rows <- apply(cells, 1, function(row) paste(row, collapse = " & "))
  paste(
    c(
      sprintf("\\begin{tabular}{%s}", paste(align, collapse = "")),
      "\\hline",
      paste0(header, " \\\\"),
      "\\hline",
      paste0(rows, " \\\\"),
      "\\hline",
      "\\end{tabular}"
    ),
    collapse = "\n"
  )
}

#' Render a data frame as a table in the chosen format
#'
#' @param data A data frame to render.
#' @param format Target output format.
#' @param ... Passed to the format-specific renderer.
#' @return A single string containing the table.
#' @family table renderers
#' @export
render.table <- function(data, format = c("markdown", "html", "latex"), ...) {
  format <- match.arg(format)
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  switch(format,
         markdown = table.markdown(data, ...),
         html = table.html(data, ...),
         latex = table.latex(data, ...))
}

#' Truncate long cell text with an ellipsis
#'
#' @param x A character vector of cell values.
#' @param width Maximum number of characters to keep, including the
#'   ellipsis.
#' @return A character vector the same length as `x`.
#' @export
<<<<<<< CURRENT
  <|user_cursor|>head <- paste0("<th style=\"text-align:",
                 css,
                 "\">",
                 escape.html(names(data)),
                 "</th>")
=======
<[fim-middle]>
````

</details>

## 56. `967f47c8aee20cb7fc21b537` format_propagation / fmt_aligned

- package `synth:fitwright`, path `R/methods.R`
- generator: `{"rule": "fmt_aligned", "seed": "b02-base-stats/fitwright/R/methods.R", "style": "dense", "target_lines": 8}`
- verification: `{"air_idempotent": true, "completion_tokens": 58, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 1411, "reward": 1.0, "reward_comparison": "raw_formatting_target", "reward_exact": true, "reward_flags": ["formatting_only_reference"]}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -109,12 +109,14 @@
 print.summary.fw_perm <- function(x, digits = 4, ...) {
   cat(x$method, "\n", sep = "")
   cat("  statistic  = ", format(signif(x$statistic, digits)), "\n", sep = "")
-  cat("  p-value    = ",
-      format_p(x$p_value),
-      " (MC s.e. ",
-      format(signif(x$mc_error, 2)),
-      ")\n",
-      sep = "")
+  cat(
+    "  p-value    = ",
+    format_p(x$p_value),
+    " (MC s.e. ",
+    format(signif(x$mc_error, 2)),
+    ")\n",
+    sep = ""
+  )
   cat("Null distribution quantiles:\n")
   print(signif(x$null_quantiles, digits))
   invisible(x)
```

**Target (gold region)**

```r
  cat(
    "  p-value    = ",
    format_p(x$p_value),
    " (MC s.e. ",
    format(signif(x$mc_error, 2)),
    ")\n",
    sep = ""
  )
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/methods.R
#' Summarise a bootstrap result
#'
#' @param object An object of class `fw_boot`.
#' @param level Confidence level for the reported intervals.
#' @param ... Ignored.
#' @return An object of class `summary.fw_boot`.
#' @family fw_boot methods
#' @export
summary.fw_boot <- function(object, level = 0.95, ...) {
  ci <- boot_ci_percentile(object, level)
  table <- data.frame(
    term = names(object$t0),
    original = unname(object$t0),
    bias = unname(colMeans(object$t, na.rm = TRUE) - object$t0),
    std_error = unname(apply(object$t, 2, stats::sd, na.rm = TRUE)),
    lower = unname(ci[, "lower"]),
    upper = unname(ci[, "upper"]),
    row.names = NULL
  )
  failed <- sum(!stats::complete.cases(object$t))
  structure(
    list(table = table, level = level, R = object$R, failed = failed),
    class = "summary.fw_boot"
  )
}

#' Print a bootstrap summary
#'
#' @param x An object of class `summary.fw_boot`.
#' @param digits Number of significant digits to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @family fw_boot methods
#' @export
print.summary.fw_boot <- function(x, digits = 4, ...) {
  cat(
    "Bootstrap summary (",
    x$R,
    " resamples, ",
    x$level * 100,
    "% percentile CI)\n",
    sep = ""
  )
  if (x$failed > 0) {
    cat(x$failed, " resample(s) failed and were dropped\n", sep = "")
  }
  tab <- x$table
  num <- vapply(tab, is.numeric, logical(1))
  tab[num] <- lapply(tab[num], signif, digits = digits)
  print(tab, row.names = FALSE)
  invisible(x)
}

#' Print a permutation test result
#'
#' @param x An object of class `fw_perm`.
#' @param digits Number of significant digits to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @family fw_perm methods
#' @export
print.fw_perm <- function(x, digits = 4, ...) {
  cat(x$method, "\n", sep = "")
  cat("  statistic  = ", format(signif(x$statistic, digits)), "\n", sep = "")
  cat("  p-value    = ", format_p(x$p_value), "\n", sep = "")
  cat(
    "  alternative: ",
    x$alternative,
    " (",
    x$n_perm,
    " permutations)\n",
    sep = ""
  )
  invisible(x)
}

#' Summarise a permutation test result
#'
#' @param object An object of class `fw_perm`.
#' @param probs Quantiles of the null distribution to report.
#' @param ... Ignored.
#' @return An object of class `summary.fw_perm`.
#' @family fw_perm methods
#' @export
summary.fw_perm <- function(object, probs = c(0.025, 0.5, 0.975), ...) {
  p <- object$p_value
  B <- object$n_perm
  structure(
    list(
      method = object$method,
      statistic = object$statistic,
      p_value = p,
      mc_error = sqrt(p * (1 - p) / B),
      null_quantiles = stats::quantile(object$null_dist, probs = probs),
      n_perm = B
    ),
    class = "summary.fw_perm"
  )
}

#' Print a permutation test summary
#'
#' @param x An object of class `summary.fw_perm`.
#' @param digits Number of significant digits to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @family fw_perm methods
#' @export
print.summary.fw_perm <- function(x, digits = 4, ...) {
  cat(x$method, "\n", sep = "")
  cat("  statistic  = ", format(signif(x$statistic, digits)), "\n", sep = "")
<filename>selected_references
<filename>edit_history
User edited "R/methods.R":

```diff
@@ -123,9 +123,11 @@
-  structure(list(
-              method = object$method,
-              statistic = object$statistic,
-              p_value = p,
-              mc_error = sqrt(p * (1 - p) / B),
-              null_quantiles = stats::quantile(object$null_dist, probs = probs),
-              n_perm = B
-            ),
-            class = "summary.fw_perm")
+  structure(
+    list(
+      method = object$method,
+      statistic = object$statistic,
+      p_value = p,
+      mc_error = sqrt(p * (1 - p) / B),
+      null_quantiles = stats::quantile(object$null_dist, probs = probs),
+      n_perm = B
+    ),
+    class = "summary.fw_perm"
+  )
```
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
  cat("Null distribution quantiles:\n")
  print(signif(x$null_quantiles, digits))
  invisible(x)
}
<<<<<<< CURRENT
  <|user_cursor|>cat("  p-value    = ",
      format_p(x$p_value),
      " (MC s.e. ",
      format(signif(x$mc_error, 2)),
      ")\n",
      sep = "")
=======
<[fim-middle]>
````

</details>

## 57. `bf3528368a66da10c6e2b080` format_propagation / fmt_aligned

- package `synth:shelfwatch`, path `R/mod_reorder.R`
- generator: `{"rule": "fmt_aligned", "seed": "b04-shiny/shelfwatch/R/mod_reorder.R", "style": "dense", "target_lines": 8}`
- verification: `{"air_idempotent": true, "completion_tokens": 48, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 1701, "reward": 1.0, "reward_comparison": "raw_formatting_target", "reward_exact": true, "reward_flags": ["formatting_only_reference"]}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -104,12 +104,14 @@
 #' @return A data frame with zero rows.
 #' @keywords internal
 empty_order_table <- function() {
-  data.frame(sku = character(),
-             supplier = character(),
-             qty = integer(),
-             unit_cost = numeric(),
-             line_total = numeric(),
-             stringsAsFactors = FALSE)
+  data.frame(
+    sku = character(),
+    supplier = character(),
+    qty = integer(),
+    unit_cost = numeric(),
+    line_total = numeric(),
+    stringsAsFactors = FALSE
+  )
 }
 
 #' Build a single order line
```

**Target (gold region)**

```r
  data.frame(
    sku = character(),
    supplier = character(),
    qty = integer(),
    unit_cost = numeric(),
    line_total = numeric(),
    stringsAsFactors = FALSE
  )
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/mod_reorder.R
#' Reorder module UI
#'
#' A sidebar layout with a small form for drafting purchase orders and a
#' main panel showing the pending lines and their totals by supplier.
#'
#' @param id The module namespace id.
#' @return A [bslib::layout_sidebar()] containing the module inputs and outputs.
#' @family reorder module
#' @export
mod_reorder_ui <- function(id) {
  ns <- shiny::NS(id)
  bslib::layout_sidebar(
    sidebar = bslib::sidebar(
      title = "New order line",
      shiny::selectizeInput(ns("sku"), "SKU", choices = NULL),
      shiny::numericInput(ns("qty"), "Quantity", value = NA, min = 1, step = 1),
      shiny::textInput(ns("supplier"), "Supplier"),
      shiny::actionButton(ns("add"), "Add line", class = "btn-primary"),
      shiny::actionButton(
        ns("clear"),
        "Clear all",
        class = "btn-outline-danger"
      )
    ),
    shiny::textOutput(ns("feedback")),
    DT::DTOutput(ns("orders")),
    shiny::plotOutput(ns("by_supplier"), height = "250px")
  )
}

#' Reorder module server
#'
#' Validates new order lines, keeps the pending lines in a reactive value and
#' suggests a quantity whenever a SKU is chosen.
#'
#' @param id The module namespace id.
#' @param stock A reactive returning a data frame of current stock levels.
#' @return A reactive returning the data frame of pending order lines.
#' @family reorder module
#' @export
mod_reorder_server <- function(id, stock) {
  stopifnot(shiny::is.reactive(stock))
  shiny::moduleServer(id, function(input, output, session) {
    orders <- shiny::reactiveVal(empty_order_table())
    feedback <- shiny::reactiveVal("")
    shiny::observe({
      shiny::updateSelectizeInput(
        session,
        "sku",
        choices = stock()$sku,
        server = TRUE
      )
    })
    shiny::observeEvent(input$sku, {
      item <- stock()[stock()$sku == input$sku, , drop = FALSE]
      shiny::req(nrow(item) == 1)
      suggested <- reorder_quantity(
        item$on_hand,
        item$reorder_point,
        item$target_level
      )
      shiny::updateNumericInput(session, "qty", value = suggested)
    })
    shiny::observeEvent(input$add, {
      errors <- collect_errors(
        need_sku(input$sku, "SKU"),
        need_positive_int(input$qty, "Quantity"),
        need_text(input$supplier, "Supplier")
      )
      if (length(errors) > 0) {
        feedback(errors[[1]])
        return()
      }
      line <- build_order_row(
        stock(),
        clean_sku(input$sku),
        input$qty,
        input$supplier
      )
      orders(rbind(orders(), line))
      feedback(sprintf("Added %d x %s.", as.integer(input$qty), line$sku))
    })
    shiny::observeEvent(input$clear, {
      orders(empty_order_table())
      feedback("Cleared all pending lines.")
    })
    output$feedback <- shiny::renderText(feedback())
    output$orders <- DT::renderDT({
      DT::datatable(orders(), rownames = FALSE, options = list(dom = "tp"))
    })
    output$by_supplier <- shiny::renderPlot({
      shiny::req(nrow(orders()) > 0)
      plot_order_totals(order_totals_by_supplier(orders()))
    })
    shiny::reactive(orders())
  })
}

#' Create an empty order table
#'
#' Returns a zero-row data frame with the columns used for pending order
#' lines so that later rows can be bound onto it.
#'
#' @return A data frame with zero rows.
#' @keywords internal
empty_order_table <- function() {
<filename>selected_references
<filename>edit_history
User edited "R/mod_reorder.R":

```diff
@@ -74,4 +74,6 @@
-      line <- build_order_row(stock(),
-                              clean_sku(input$sku),
-                              input$qty,
-                              input$supplier)
+      line <- build_order_row(
+        stock(),
+        clean_sku(input$sku),
+        input$qty,
+        input$supplier
+      )
```
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
}

#' Build a single order line
#'
#' Looks up the unit cost for `sku` in the stock data and returns one row
#' matching the layout of [empty_order_table()].
#'
#' @param stock A data frame of current stock levels.
#' @param sku A single cleaned SKU.
#' @param qty The number of units to order.
#' @param supplier The supplier name as typed by the user.
#' @return A one-row data frame.
#' @keywords internal
build_order_row <- function(stock, sku, qty, supplier) {
  unit_cost <- stock$unit_cost[match(sku, stock$sku)]
  if (is.na(unit_cost)) {
    stop("Unknown SKU: ", sku, call. = FALSE)
  }
  qty <- as.integer(qty)
  data.frame(sku = sku,
             supplier = clean_supplier(supplier),
             qty = qty,
             unit_cost = unit_cost,
             line_total = qty * unit_cost,
             stringsAsFactors = FALSE)
}

#' Suggest a reorder quantity
#'
#' Orders enough units to bring stock back up to the target level once it
#' has dropped to the reorder point, rounded up to whole cases.
#'
#' @param on_hand The current quantity on hand.
#' @param reorder_point The level at or below which an order is suggested.
#' @param target_level The level stock should be refilled to.
#' @param case_size The number of units per case.
#' @return An integer number of units, zero when no order is needed.
#' @keywords internal
reorder_quantity <- function(
  on_hand,
  reorder_point,
  target_level,
  case_size = 1L
) {
  if (on_hand > reorder_point) {
    return(0L)
  }
  shortfall <- max(target_level - on_hand, 0)
  as.integer(ceiling(shortfall / case_size) * case_size)
}

#' Total pending orders by supplier
#'
#' Sums line totals and quantities for each supplier, largest spend first.
#'
#' @param orders A data frame of pending order lines.
#' @return A data frame with columns `supplier`, `qty` and `line_total`.
#' @keywords internal
<<<<<<< CURRENT
  <|user_cursor|>data.frame(sku = character(),
             supplier = character(),
             qty = integer(),
             unit_cost = numeric(),
             line_total = numeric(),
             stringsAsFactors = FALSE)
=======
<[fim-middle]>
````

</details>

## 58. `5f20aec557172aa825dbff87` format_propagation / fmt_aligned

- package `synth:foldwise`, path `R/splitters.R`
- generator: `{"rule": "fmt_aligned", "seed": "b11-ml/foldwise/R/splitters.R", "style": "dense", "target_lines": 6}`
- verification: `{"air_idempotent": true, "completion_tokens": 51, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 1266, "reward": 1.0, "reward_comparison": "raw_formatting_target", "reward_exact": true, "reward_flags": ["formatting_only_reference"]}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -94,10 +94,12 @@
     attr(x, "n")
   ))
   for (fold in x) {
-    cat(sprintf("  %-9s analysis %5d | assessment %5d\n",
-                fold$id,
-                length(fold$analysis),
-                length(fold$assessment)))
+    cat(sprintf(
+      "  %-9s analysis %5d | assessment %5d\n",
+      fold$id,
+      length(fold$analysis),
+      length(fold$assessment)
+    ))
   }
   return(invisible(x))
 }
```

**Target (gold region)**

```r
    cat(sprintf(
      "  %-9s analysis %5d | assessment %5d\n",
      fold$id,
      length(fold$analysis),
      length(fold$assessment)
    ))
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/splitters.R
#' Grouped k-fold cross-validation splits
#'
#' Keeps every row of a group in the same fold, so that no group appears in
#' both the analysis and assessment sets of a split.
#'
#' @param groups A vector of group identifiers, one per row.
#' @param k Number of folds.
#' @param seed Optional integer seed for reproducible assignment.
#' @return An object of class `fw_splits`.
#' @family splitters
#' @export
grouped_splits <- function(groups, k = 5, seed = NULL) {
  n <- length(groups)
  unique_groups <- unique(groups)
  check_split_size(length(unique_groups), k)
  if (!is.null(seed)) {
    set.seed(seed)
  }
  # greedy balancing: biggest groups first, each into the lightest fold
  sizes <- table(groups)[as.character(unique_groups)]
  order_idx <- order(-as.integer(sizes), sample(length(sizes)))
  load <- integer(k)
  group_fold <- integer(length(unique_groups))
  for (j in order_idx) {
    target <- which.min(load)
    group_fold[j] <- target
    load[target] <- load[target] + sizes[[j]]
  }
  fold_id <- group_fold[match(groups, unique_groups)]
  holdout <- split(seq_len(n), factor(fold_id, levels = seq_len(k)))
  return(new_splits(unname(holdout), n, method = "grouped"))
}

#' Leave-one-group-out splits
#'
#' Creates one split per distinct group, holding that group out entirely.
#'
#' @param groups A vector of group identifiers, one per row.
#' @return An object of class `fw_splits`.
#' @family splitters
#' @export
logo_splits <- function(groups) {
  n <- length(groups)
  unique_groups <- unique(groups)
  if (length(unique_groups) < 2L) {
    stop("Need at least two groups for leave-one-group-out.", call. = FALSE)
  }
  holdout <- lapply(unique_groups, function(g) which(groups == g))
  return(new_splits(holdout, n, method = "logo"))
}

#' Rolling-origin splits for ordered data
#'
#' Produces forward-chaining splits where each assessment window follows the
#' analysis window in time order.
#'
#' @param n Number of rows to split.
#' @param initial Number of rows in the first analysis window.
#' @param assess Number of rows in each assessment window.
#' @param cumulative If `TRUE`, the analysis window grows; otherwise it slides.
#' @return An object of class `fw_splits`.
#' @family splitters
#' @export
rolling_splits <- function(n, initial, assess = 1L, cumulative = TRUE) {
  if (initial + assess > n) {
    stop("`initial + assess` must not exceed `n`.", call. = FALSE)
  }
  starts <- seq.int(initial + 1L, n - assess + 1L, by = assess)
  folds <- vector("list", length(starts))
  for (i in seq_along(starts)) {
    first <- if (cumulative) 1L else starts[i] - initial
    folds[[i]] <- list(
      id = sprintf("Slice%03d", i),
      analysis = seq.int(first, starts[i] - 1L),
      assessment = seq.int(starts[i], starts[i] + assess - 1L)
    )
  }
  return(structure(folds, n = n, method = "rolling", class = "fw_splits"))
}

#' Print a set of splits
#'
#' Shows the splitting method and the analysis/assessment sizes of each fold.
#'
#' @param x An object of class `fw_splits`.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.fw_splits <- function(x, ...) {
  cat(sprintf(
    "<fw_splits: %s, %d folds, n = %d>\n",
    attr(x, "method"),
    length(x),
    attr(x, "n")
  ))
  for (fold in x) {
<filename>selected_references
<filename>edit_history
User edited "R/splitters.R":

```diff
@@ -197,4 +197,6 @@
-  cat(sprintf("<fw_splits: %s, %d folds, n = %d>\n",
-              attr(x, "method"),
-              length(x),
-              attr(x, "n")))
+  cat(sprintf(
+    "<fw_splits: %s, %d folds, n = %d>\n",
+    attr(x, "method"),
+    length(x),
+    attr(x, "n")
+  ))
```
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
  }
  return(invisible(x))
}
<<<<<<< CURRENT
    <|user_cursor|>cat(sprintf("  %-9s analysis %5d | assessment %5d\n",
                fold$id,
                length(fold$analysis),
                length(fold$assessment)))
=======
<[fim-middle]>
````

</details>

## 59. `8101eae2c64689e3507cfe6d` format_propagation / fmt_aligned

- package `synth:cellarlink`, path `R/upsert.R`
- generator: `{"rule": "fmt_aligned", "seed": "b12-db/cellarlink/R/upsert.R", "style": "sectioned", "target_lines": 7}`
- verification: `{"air_idempotent": true, "completion_tokens": 44, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 1368, "reward": 1.0, "reward_comparison": "raw_formatting_target", "reward_exact": true, "reward_flags": ["formatting_only_reference"]}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -112,11 +112,13 @@
     idx <- starts[i]:min(starts[i] + batch_size - 1L, n_rows)
     affected[i] <- writer(con, table, data[idx, , drop = FALSE], keys)
     if (verbose) {
-      message(sprintf("[%s] batch %d/%d: %d rows",
-                      table,
-                      i,
-                      length(starts),
-                      affected[i]))
+      message(sprintf(
+        "[%s] batch %d/%d: %d rows",
+        table,
+        i,
+        length(starts),
+        affected[i]
+      ))
     }
   }
   names(affected) <- paste0("batch_", seq_along(starts))
```

**Target (gold region)**

```r
      message(sprintf(
        "[%s] batch %d/%d: %d rows",
        table,
        i,
        length(starts),
        affected[i]
      ))
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/upsert.R
#' Upsert rows using delete-then-insert
#'
#' Deletes rows of `table` whose keys appear in `data`, then appends `data`,
#' inside one transaction. Useful on backends without `ON CONFLICT`.
#'
#' @param con A `DBIConnection` object.
#' @param table Name of the target table.
#' @param data A data frame of rows to write.
#' @param keys A character vector of key column names.
#' @return The number of rows affected, invisibly.
#' @family upserts
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

#' Insert only rows whose keys are new
#'
#' Appends rows of `data` whose keys do not already exist in `table`;
#' existing rows are left untouched.
#'
#' @param con A `DBIConnection` object.
#' @param table Name of the target table.
#' @param data A data frame of rows to write.
#' @param keys A character vector of key column names.
#' @return The number of rows affected, invisibly.
#' @family upserts
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

#' Upsert rows in batches
#'
#' Splits `data` into chunks of `batch_size` rows and upserts each chunk
#' with the chosen strategy, reporting progress as it goes.
#'
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
  n_rows <- nrow(data)
  if (n_rows == 0) {
    return(invisible(integer()))
  }
  starts <- seq(1L, n_rows, by = batch_size)
  affected <- integer(length(starts))
  for (i in seq_along(starts)) {
    idx <- starts[i]:min(starts[i] + batch_size - 1L, n_rows)
    affected[i] <- writer(con, table, data[idx, , drop = FALSE], keys)
    if (verbose) {
<filename>selected_references
<filename>edit_history
User edited "R/upsert.R":

```diff
@@ -183,4 +183,6 @@
-  writer <- switch(method,
-                   on_conflict = upsert_on_conflict,
-                   replace = upsert_replace,
-                   insert_new = insert_new)
+  writer <- switch(
+    method,
+    on_conflict = upsert_on_conflict,
+    replace = upsert_replace,
+    insert_new = insert_new
+  )
```
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
    }
  }
  names(affected) <- paste0("batch_", seq_along(starts))
  return(invisible(affected))
}
<<<<<<< CURRENT
      <|user_cursor|>message(sprintf("[%s] batch %d/%d: %d rows",
                      table,
                      i,
                      length(starts),
                      affected[i]))
=======
<[fim-middle]>
````

</details>

## 60. `1c9b34792be39429e86f9598` format_propagation / fmt_aligned

- package `synth:shelfwatch`, path `R/mod_reorder.R`
- generator: `{"rule": "fmt_aligned", "seed": "b04-shiny/shelfwatch/R/mod_reorder.R", "style": "dense", "target_lines": 8}`
- verification: `{"air_idempotent": true, "completion_tokens": 61, "family_problems": [], "parse_after": true, "parse_before_status": "ok", "passed": true, "prompt_tokens": 1677, "reward": 1.0, "reward_comparison": "raw_formatting_target", "reward_exact": true, "reward_flags": ["formatting_only_reference"]}`

**Edit (window diff, before -> after)**

```diff
--- before
+++ after
@@ -101,12 +101,14 @@
     stop("Unknown SKU: ", sku, call. = FALSE)
   }
   qty <- as.integer(qty)
-  data.frame(sku = sku,
-             supplier = clean_supplier(supplier),
-             qty = qty,
-             unit_cost = unit_cost,
-             line_total = qty * unit_cost,
-             stringsAsFactors = FALSE)
+  data.frame(
+    sku = sku,
+    supplier = clean_supplier(supplier),
+    qty = qty,
+    unit_cost = unit_cost,
+    line_total = qty * unit_cost,
+    stringsAsFactors = FALSE
+  )
 }
 
 #' Suggest a reorder quantity
```

**Target (gold region)**

```r
  data.frame(
    sku = sku,
    supplier = clean_supplier(supplier),
    qty = qty,
    unit_cost = unit_cost,
    line_total = qty * unit_cost,
    stringsAsFactors = FALSE
  )
```

<details><summary>Rendered prompt (prompt_text)</summary>

````text
Edit the current region. Output the complete replacement region, preserving whitespace. For no change, output the ordinary line [NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an empty body is reserved for deletion of a nonempty range.
<filename>R/mod_reorder.R
#' Reorder module server
#'
#' Validates new order lines, keeps the pending lines in a reactive value and
#' suggests a quantity whenever a SKU is chosen.
#'
#' @param id The module namespace id.
#' @param stock A reactive returning a data frame of current stock levels.
#' @return A reactive returning the data frame of pending order lines.
#' @family reorder module
#' @export
mod_reorder_server <- function(id, stock) {
  stopifnot(shiny::is.reactive(stock))
  shiny::moduleServer(id, function(input, output, session) {
    orders <- shiny::reactiveVal(empty_order_table())
    feedback <- shiny::reactiveVal("")
    shiny::observe({
      shiny::updateSelectizeInput(
        session,
        "sku",
        choices = stock()$sku,
        server = TRUE
      )
    })
    shiny::observeEvent(input$sku, {
      item <- stock()[stock()$sku == input$sku, , drop = FALSE]
      shiny::req(nrow(item) == 1)
      suggested <- reorder_quantity(
        item$on_hand,
        item$reorder_point,
        item$target_level
      )
      shiny::updateNumericInput(session, "qty", value = suggested)
    })
    shiny::observeEvent(input$add, {
      errors <- collect_errors(
        need_sku(input$sku, "SKU"),
        need_positive_int(input$qty, "Quantity"),
        need_text(input$supplier, "Supplier")
      )
      if (length(errors) > 0) {
        feedback(errors[[1]])
        return()
      }
      line <- build_order_row(
        stock(),
        clean_sku(input$sku),
        input$qty,
        input$supplier
      )
      orders(rbind(orders(), line))
      feedback(sprintf("Added %d x %s.", as.integer(input$qty), line$sku))
    })
    shiny::observeEvent(input$clear, {
      orders(empty_order_table())
      feedback("Cleared all pending lines.")
    })
    output$feedback <- shiny::renderText(feedback())
    output$orders <- DT::renderDT({
      DT::datatable(orders(), rownames = FALSE, options = list(dom = "tp"))
    })
    output$by_supplier <- shiny::renderPlot({
      shiny::req(nrow(orders()) > 0)
      plot_order_totals(order_totals_by_supplier(orders()))
    })
    shiny::reactive(orders())
  })
}

#' Create an empty order table
#'
#' Returns a zero-row data frame with the columns used for pending order
#' lines so that later rows can be bound onto it.
#'
#' @return A data frame with zero rows.
#' @keywords internal
empty_order_table <- function() {
  data.frame(
    sku = character(),
    supplier = character(),
    qty = integer(),
    unit_cost = numeric(),
    line_total = numeric(),
    stringsAsFactors = FALSE
  )
}

#' Build a single order line
#'
#' Looks up the unit cost for `sku` in the stock data and returns one row
#' matching the layout of [empty_order_table()].
#'
#' @param stock A data frame of current stock levels.
#' @param sku A single cleaned SKU.
#' @param qty The number of units to order.
#' @param supplier The supplier name as typed by the user.
#' @return A one-row data frame.
#' @keywords internal
build_order_row <- function(stock, sku, qty, supplier) {
  unit_cost <- stock$unit_cost[match(sku, stock$sku)]
  if (is.na(unit_cost)) {
    stop("Unknown SKU: ", sku, call. = FALSE)
  }
  qty <- as.integer(qty)
<filename>selected_references
<filename>edit_history
User edited "R/mod_reorder.R":

```diff
@@ -107,6 +107,8 @@
-  data.frame(sku = character(),
-             supplier = character(),
-             qty = integer(),
-             unit_cost = numeric(),
-             line_total = numeric(),
-             stringsAsFactors = FALSE)
+  data.frame(
+    sku = character(),
+    supplier = character(),
+    qty = integer(),
+    unit_cost = numeric(),
+    line_total = numeric(),
+    stringsAsFactors = FALSE
+  )
```
<filename>diagnostics
<filename>retrieval
<[fim-suffix]>
}

#' Suggest a reorder quantity
#'
#' Orders enough units to bring stock back up to the target level once it
#' has dropped to the reorder point, rounded up to whole cases.
#'
#' @param on_hand The current quantity on hand.
#' @param reorder_point The level at or below which an order is suggested.
#' @param target_level The level stock should be refilled to.
#' @param case_size The number of units per case.
#' @return An integer number of units, zero when no order is needed.
#' @keywords internal
reorder_quantity <- function(
  on_hand,
  reorder_point,
  target_level,
  case_size = 1L
) {
  if (on_hand > reorder_point) {
    return(0L)
  }
  shortfall <- max(target_level - on_hand, 0)
  as.integer(ceiling(shortfall / case_size) * case_size)
}

#' Total pending orders by supplier
#'
#' Sums line totals and quantities for each supplier, largest spend first.
#'
#' @param orders A data frame of pending order lines.
#' @return A data frame with columns `supplier`, `qty` and `line_total`.
#' @keywords internal
order_totals_by_supplier <- function(orders) {
  totals <- stats::aggregate(cbind(qty, line_total) ~ supplier,
                             data = orders,
                             FUN = sum)
  totals[order(-totals$line_total), , drop = FALSE]
}

#' Plot pending order totals by supplier
#'
#' Draws a horizontal bar chart of spend per supplier.
#'
#' @param totals A data frame as returned by [order_totals_by_supplier()].
#' @return A ggplot object.
#' @keywords internal
plot_order_totals <- function(totals) {
  totals$supplier <- factor(totals$supplier, levels = rev(totals$supplier))
  ggplot2::ggplot(totals, ggplot2::aes(x = line_total, y = supplier)) +
    ggplot2::geom_col(fill = "#2c7fb8") +
    ggplot2::labs(x = "Order value", y = NULL) +
    ggplot2::theme_minimal(base_size = 13)
}
<<<<<<< CURRENT
  <|user_cursor|>data.frame(sku = sku,
             supplier = clean_supplier(supplier),
             qty = qty,
             unit_cost = unit_cost,
             line_total = qty * unit_cost,
             stringsAsFactors = FALSE)
=======
<[fim-middle]>
````

</details>
