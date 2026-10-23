#' Format numbers for report text
#'
#' @param x A numeric vector.
#' @param type One of `"count"` (rounded, with thousands separators),
#'   `"pct"` (a proportion shown as a percentage), `"money"` (two decimals
#'   with a currency symbol) or `"delta"` (a signed difference).
#' @param digits Number of decimal places for `"pct"` and `"delta"`.
#' @param symbol Currency symbol placed before `"money"` amounts.
#' @param na Text used for missing values.
#' @return A character vector the same length as `x`.
#' @keywords internal
fmt.value = function(x, type = c("count", "pct", "money", "delta"),
                     digits = 1L, symbol = "$", na = "-") {
  type = match.arg(type)
  if (!is.numeric(x)) {
    stop("`x` must be numeric.", call. = FALSE)
  }
  if (!is.character(na) || length(na) != 1L) {
    stop("`na` must be a single string.", call. = FALSE)
  }

  out = switch(type,
    count = {
      formatC(round(x), format = "d", big.mark = ",")
    },
    pct = {
      paste0(formatC(100 * x, format = "f", digits = digits), "%")
    },
    money = {
      body = formatC(abs(x), format = "f", digits = 2L, big.mark = ",")
      paste0(ifelse(x < 0, "-", ""), symbol, body)
    },
    delta = {
      body = formatC(abs(x), format = "f", digits = digits)
      paste0(ifelse(x < 0, "-", "+"), body)
    }
  )
  out[is.na(x)] = na
  out
}

#' Label numbers on a fixed ladder
#'
#' @param x A numeric vector.
#' @param kind Which ladder to use. Each ladder is listed from the top rung
#'   down in `cuts` below.
#' @param cuts Lower bounds per ladder: `size` labels counts as `"huge"`,
#'   `"large"`, `"medium"`, `"small"` or `"tiny"`; `trend` labels relative
#'   changes as `"strong_up"`, `"up"`, `"flat"`, `"down"` or
#'   `"strong_down"`; `share` labels proportions as `"dominant"`,
#'   `"major"`, `"minor"` or `"marginal"`.
#' @return A factor with the labels of the chosen ladder as levels, from
#'   the lowest rung to the highest.
#' @export
label.ladder = function(x, kind = c("size", "trend", "share"),
                        cuts = list(
                          size = c(huge = 10000, large = 1000, medium = 100, small = 10),
                          trend = c(strong_up = 0.1, up = 0.01, flat = -0.01, down = -0.1),
                          share = c(dominant = 0.5, major = 0.2, minor = 0.05)
                        )) {
  kind = match.arg(kind)
  rungs = cuts[[kind]]
  out = rep(NA_character_, length(x))

  # Fill from the lowest rung up so that higher rungs overwrite lower ones
  switch(kind,
    size = {
      out[!is.na(x)] = "tiny"
      bottom = "tiny"
    },
    trend = {
      out[!is.na(x)] = "strong_down"
      bottom = "strong_down"
    },
    share = {
      out[!is.na(x)] = "marginal"
      bottom = "marginal"
    }
  )
  for (rung in rev(names(rungs))) {
    out[!is.na(x) & x >= rungs[[rung]]] = rung
  }
  factor(out, levels = c(bottom, rev(names(rungs))))
}

#' Describe the change in a value between two periods
#'
#' @param before,after Numeric values for the two periods.
#' @param name Name of the value used in the sentence.
#' @param tolerance Relative change below which the value counts as
#'   unchanged.
#' @return A single sentence.
#' @export
describe.change = function(before, after, name, tolerance = 0.01) {
  if (is.na(before) && is.na(after)) {
    return(sprintf("No %s was recorded in either period.", name))
  }
  if (is.na(before)) {
    return(sprintf("%s was %s; no earlier value.", name, fmt.value(after, "count")))
  }
  if (is.na(after)) {
    return(sprintf("%s was %s; no later value.", name, fmt.value(before, "count")))
  }

  # Relative change against the earlier value
  ratio = if (before == 0) NA_real_ else (after - before) / abs(before)
  if (!is.na(ratio) && abs(ratio) < tolerance) {
    sprintf("%s was unchanged at %s.", name, fmt.value(after, "count"))
  } else if (after > before) {
    sprintf("%s rose from %s to %s (%s).", name, fmt.value(before, "count"),
            fmt.value(after, "count"), fmt.value(ratio, "pct"))
  } else {
    sprintf("%s fell from %s to %s (%s).", name, fmt.value(before, "count"),
            fmt.value(after, "count"), fmt.value(ratio, "pct"))
  }
}

#' Turn a per-group statistics table into sentences
#'
#' @param table A data frame with columns `group`, `n`, `mean`, `sd`,
#'   `min` and `max`, such as the result of [group.stats.table()].
#' @param value Name of the summarised value used in the sentences.
#' @return A named character vector with one sentence per group.
#' @export
group.sentences = function(table, value) {
  needed = c("group", "n", "mean", "sd", "min", "max")
  if (!all(needed %in% names(table))) {
    stop("`table` must have columns ", toString(needed), ".", call. = FALSE)
  }
  out = character(nrow(table))

  for (i in seq_len(nrow(table))) {
    row = table[i, , drop = FALSE]
    parts = character()
    # Count
    parts = c(parts, sprintf("%s rows", fmt.value(row$n, "count")))
    # Mean
    parts = c(parts, sprintf("mean %s %s", value, fmt.value(row$mean, "count")))
    # Spread
    parts = c(parts, sprintf("sd %s", fmt.value(row$sd, "count")))
    # Range
    parts = c(parts, sprintf("range %s to %s", fmt.value(row$min, "count"),
                             fmt.value(row$max, "count")))
    out[i] = sprintf("%s: %s.", row$group, paste(parts, collapse = ", "))
  }
  stats::setNames(out, as.character(table$group))
}

#' Add trend and size labels to a change table
#'
#' @param table A tibble from [group.change.table()].
#' @param tolerance Relative change below which a value counts as
#'   unchanged in the sentence.
#' @return `table` with the factor columns `trend` and `size` from
#'   [label.ladder()] and the character column `sentence` added.
#' @export
label.trends = function(table, tolerance = 0.01) {
  table %>%
    dplyr::mutate(
      trend = label.ladder(change_pct / 100, "trend"),
      size = label.ladder(after_n, "size"),
      sentence = purrr::pmap_chr(
        list(before_mean, after_mean, as.character(group)),
        function(b, a, g) describe.change(b, a, g, tolerance = tolerance)
      )
    )
}
