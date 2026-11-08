#' Create a report section
#'
#' @param name Section name, used as its title.
#' @param items A named character vector of values to report.
#' @return A list of class `"mill_section"` with elements `name`, `items`
#'   and `empty`.
#' @keywords internal
new.section = function(name, items = character()) {
  structure(
    list(name = name, items = items, empty = length(items) == 0L),
    class = "mill_section"
  )
}

#' @title Overview section for one group
#' @description Reports the number of rows and the mean, median and
#'   standard deviation of `value`.
#' @param df A data frame holding the rows of one group.
#' @param value Name of the numeric column to describe.
#' @param digits Number of decimal places.
#' @return A `mill_section` with items `rows`, `mean`, `median`, `sd` and
#'   `cv` and the attribute `"rows"`. The other section builders return the items listed in
#'   [group.sections()].
#' @rdname group.sections
section.overview = function(df, value, digits = 1L) {
  if (!value %in% names(df)) {
    stop(sprintf("Column `%s` not found in `df`.", value), call. = FALSE)
  }
  x = df[[value]]
  x = x[!is.na(x)]
  if (length(x) == 0L) {
    return(new.section("overview"))
  }

  # Items in report order
  spread = if (length(x) > 1L) stats::sd(x) else NA_real_
  items = c(
    rows = format(nrow(df)),
    mean = format(round(mean(x), digits), nsmall = digits),
    median = format(round(stats::median(x), digits), nsmall = digits),
    sd = if (is.na(spread)) "-" else format(round(spread, digits), nsmall = digits),
    cv = if (is.na(spread) || mean(x) == 0) "-" else format(round(spread / mean(x), 2L))
  )
  section = new.section("overview", items)
  attr(section, "rows") = nrow(df)
  section
}

#' @rdname group.sections
section.distribution = function(df, value, digits = 1L) {
  if (!value %in% names(df)) {
    stop(sprintf("Column `%s` not found in `df`.", value), call. = FALSE)
  }
  x = df[[value]]
  x = x[!is.na(x)]
  if (length(x) == 0L) {
    return(new.section("distribution"))
  }

  # Items in report order
  q = stats::quantile(x, c(0.25, 0.5, 0.75), names = FALSE)
  items = c(
    min = format(round(min(x), digits), nsmall = digits),
    q1 = format(round(q[1L], digits), nsmall = digits),
    median = format(round(q[2L], digits), nsmall = digits),
    q3 = format(round(q[3L], digits), nsmall = digits),
    max = format(round(max(x), digits), nsmall = digits)
  )
  section = new.section("distribution", items)
  attr(section, "rows") = nrow(df)
  section
}

#' @rdname group.sections
section.missing = function(df, value, digits = 1L) {
  if (!value %in% names(df)) {
    stop(sprintf("Column `%s` not found in `df`.", value), call. = FALSE)
  }
  x = df[[value]]
  gaps = which(is.na(x))
  if (length(x) == 0L) {
    return(new.section("missing"))
  }

  # Items in report order
  items = c(
    n_present = format(length(x) - length(gaps)),
    n_missing = format(length(gaps)),
    pct_missing = format(round(100 * length(gaps) / length(x), digits), nsmall = digits),
    first_missing = if (length(gaps) > 0L) format(gaps[1L]) else "-",
    last_missing = if (length(gaps) > 0L) format(gaps[length(gaps)]) else "-"
  )
  section = new.section("missing", items)
  attr(section, "rows") = nrow(df)
  section
}

#' @rdname group.sections
section.extremes = function(df, value, digits = 1L) {
  if (!value %in% names(df)) {
    stop(sprintf("Column `%s` not found in `df`.", value), call. = FALSE)
  }
  x = df[[value]]
  keep = which(!is.na(x))
  if (length(keep) == 0L) {
    return(new.section("extremes"))
  }

  # Items in report order
  top = keep[which.max(x[keep])]
  bottom = keep[which.min(x[keep])]
  items = c(
    top_row = format(top),
    top_value = format(round(x[top], digits), nsmall = digits),
    bottom_row = format(bottom),
    bottom_value = format(round(x[bottom], digits), nsmall = digits),
    range = format(round(x[top] - x[bottom], digits), nsmall = digits)
  )
  section = new.section("extremes", items)
  attr(section, "rows") = nrow(df)
  section
}

#' Build report sections for every group
#'
#' Splits `data` by `group` and builds the requested sections for each
#' group with [section.overview()], [section.distribution()],
#' [section.missing()] and [section.extremes()]. Their items are:
#' * overview: `rows`, `mean`, `median`, `sd`, `cv`;
#' * distribution: `min`, `q1`, `median`, `q3`, `max`;
#' * missing: `n_present`, `n_missing`, `pct_missing`, `first_missing`,
#'   `last_missing`;
#' * extremes: `top_row`, `top_value`, `bottom_row`, `bottom_value`,
#'   `range`.
#'
#' @param data A data frame.
#' @param group Name of the grouping column.
#' @param value Name of the numeric column to describe.
#' @param parts Sections to build, in order.
#' @param digits Number of decimal places.
#' @return A named list with one element per group, each a list of
#'   `mill_section` objects named by `parts`.
#' @export
group.sections = function(data, group, value,
                          parts = c("overview", "distribution", "missing", "extremes"),
                          digits = 1L) {
  parts = match.arg(parts, several.ok = TRUE)
  if (!is.character(group) || length(group) != 1L || !group %in% names(data)) {
    stop(sprintf("`%s` must name one column of `data`.", "group"), call. = FALSE)
  }
  if (!is.character(value) || length(value) != 1L || !value %in% names(data)) {
    stop(sprintf("`%s` must name one column of `data`.", "value"), call. = FALSE)
  }
  if (!is.numeric(data[[value]])) {
    stop(sprintf("Column `%s` must be numeric.", value), call. = FALSE)
  }
  if (!is.numeric(digits) || length(digits) != 1L || digits < 0) {
    stop("`digits` must be a single non-negative number.", call. = FALSE)
  }

  pieces = split(data, data[[group]], drop = TRUE)

  # One named list of sections per group
  purrr::map(pieces, function(df) {
    sections = purrr::map(parts, function(part) {
      switch(part,
        overview = section.overview(df, value, digits),
        distribution = section.distribution(df, value, digits),
        missing = section.missing(df, value, digits),
        extremes = section.extremes(df, value, digits)
      )
    })
    stats::setNames(sections, parts)
  })
}
