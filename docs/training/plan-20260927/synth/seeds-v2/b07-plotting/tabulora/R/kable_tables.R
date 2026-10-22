#' Round and format numeric columns for display
#'
#' Formats every numeric column of a data frame as character with a fixed
#' number of decimals and a thousands separator, leaving other columns as is.
#'
#' @param data A data frame to be shown as a table.
#' @param digits Number of decimal places, either one value for all columns
#'   or a named vector giving digits per column.
#' @param big_mark Thousands separator.
#' @return A data frame with numeric columns converted to character.
#' @export
format_numeric_cols <- function(data, digits = 2, big_mark = ",") {
  num_cols <- names(data)[vapply(data, is.numeric, logical(1))]

  for (col in num_cols) {
    # Per-column digits win; otherwise fall back to the first value
    d <- if (!is.null(names(digits)) && col %in% names(digits)) digits[[col]] else digits[[1]]
    vals <- data[[col]]
    out <- formatC(vals, format = "f", digits = d, big.mark = big_mark)
    out[is.na(vals)] <- ""
    data[[col]] <- out
  }

  data
}

#' Right-align numbers and left-align text
#'
#' Computes the `align` string for [knitr::kable()] from column types.
#'
#' @param data A data frame to be shown as a table.
#' @noRd
column_alignment <- function(data) {
  ifelse(vapply(data, is.numeric, logical(1)), "r", "l")
}

#' Summary statistics table
#'
#' Computes count, mean, standard deviation, median and range for the
#' numeric columns of a data frame and renders them with [knitr::kable()].
#'
#' @param data A data frame to be shown as a table.
#' @param vars Character vector of columns to summarise. Defaults to all
#'   numeric columns.
#' @param digits Number of decimal places.
#' @param format Output format passed to [knitr::kable()].
#' @param caption Table caption.
#' @return A `knitr_kable` object.
#' @export
kable_summary <- function(data, vars = NULL, digits = 2,
                          format = c("pipe", "html", "latex"), caption = NULL) {
  format <- match.arg(format)
  if (is.null(vars)) {
    vars <- names(data)[vapply(data, is.numeric, logical(1))]
  }

  # One row of statistics per variable, stacked afterwards
  rows <- lapply(vars, function(v) {
    x <- data[[v]]
    data.frame(
      Variable = v,
      N = sum(!is.na(x)),
      Mean = mean(x, na.rm = TRUE),
      SD = stats::sd(x, na.rm = TRUE),
      Median = stats::median(x, na.rm = TRUE),
      Min = min(x, na.rm = TRUE),
      Max = max(x, na.rm = TRUE)
    )
  })
  tbl <- do.call(rbind, rows)

  tbl |>
    format_numeric_cols(digits = c(N = 0, Mean = digits, SD = digits, Median = digits,
                                   Min = digits, Max = digits)) |>
    knitr::kable(format = format, caption = caption, align = c("l", rep("r", 6)),
                 row.names = FALSE)
}

#' Regression coefficient table
#'
#' Tabulates estimates, confidence intervals and p-values from a fitted
#' model and renders them with [knitr::kable()].
#'
#' @param model A fitted model supporting [stats::coef()] and
#'   [stats::confint()].
#' @param digits Number of decimal places.
#' @param format Output format passed to [knitr::kable()].
#' @param caption Table caption.
#' @return A `knitr_kable` object.
#' @export
kable_coefficients <- function(model, digits = 2,
                               format = c("pipe", "html", "latex"), caption = NULL) {
  format <- match.arg(format)
  coefs <- summary(model)$coefficients
  # confint() keeps rows for aliased terms that summary() drops, so align
  # the intervals to the coefficient table by name
  ci <- suppressMessages(stats::confint(model))
  ci <- ci[rownames(coefs), , drop = FALSE]

  p_col <- grep("^Pr\\(", colnames(coefs), value = TRUE)
  fmt <- function(v) formatC(v, format = "f", digits = digits)

  tbl <- data.frame(
    Term = rownames(coefs),
    Estimate = fmt(coefs[, 1]),
    `95% CI` = paste0("(", fmt(ci[, 1]), ", ", fmt(ci[, 2]), ")"),
    `p-value` = ifelse(coefs[, p_col] < 0.001, "< 0.001", fmt(coefs[, p_col])),
    check.names = FALSE
  )
  knitr::kable(tbl, format = format, caption = caption, align = c("l", "r", "r", "r"),
               row.names = FALSE)
}

#' Cross-tabulation with row percentages
#'
#' Counts combinations of two categorical columns and shows each count with
#' its percentage of the row total.
#'
#' @param data A data frame to be shown as a table.
#' @param row_var Name of the column defining table rows.
#' @param col_var Name of the column defining table columns.
#' @param format Output format passed to [knitr::kable()].
#' @param caption Table caption.
#' @return A `knitr_kable` object.
#' @export
kable_crosstab <- function(data, row_var, col_var,
                           format = c("pipe", "html", "latex"), caption = NULL) {
  format <- match.arg(format)
  counts <- table(data[[row_var]], data[[col_var]], useNA = "no")
  pct <- prop.table(counts, margin = 1) * 100
  # Empty rows give 0/0; show them as 0% rather than NaN%
  pct[is.nan(pct)] <- 0

  cells <- matrix(
    paste0(counts, " (", formatC(pct, format = "f", digits = 1), "%)"),
    nrow = nrow(counts), dimnames = dimnames(counts)
  )
  tbl <- data.frame(rownames(counts), cells, Total = rowSums(counts),
                    check.names = FALSE, row.names = NULL)
  names(tbl)[1] <- row_var

  knitr::kable(tbl, format = format, caption = caption,
               align = c("l", rep("r", ncol(tbl) - 1)))
}

#' Render any data frame as a report table
#'
#' Formats numeric columns, humanises column names and renders the result
#' with [knitr::kable()] using type-based alignment.
#'
#' @param data A data frame to be shown as a table.
#' @param digits Number of decimal places.
#' @param format Output format passed to [knitr::kable()].
#' @param caption Table caption.
#' @return A `knitr_kable` object.
#' @export
kable_report <- function(data, digits = 2,
                         format = c("pipe", "html", "latex"), caption = NULL) {
  format <- match.arg(format)
  align <- column_alignment(data)

  # Same idea as humanised axis labels: underscores out, first letter up
  pretty_names <- gsub("_", " ", names(data), fixed = TRUE)
  pretty_names <- paste0(toupper(substring(pretty_names, 1, 1)), substring(pretty_names, 2))

  data |>
    format_numeric_cols(digits = digits) |>
    knitr::kable(format = format, caption = caption, align = align,
                 col.names = pretty_names, row.names = FALSE)
}
