#' Caption for a per-group table
#'
#' @param table One of `"counts"`, `"stats"`, `"change"` or `"top"`.
#' @param group Name of the grouping column.
#' @param value Name of the numeric column.
#' @param n Number of rows shown by the `"top"` table.
#' @return A single string.
#' @keywords internal
table.caption = function(table = c("counts", "stats", "change", "top"),
                         group, value, n = 3L) {
  table = match.arg(table)
  switch(table,
    counts = sprintf("Rows per %s.", group),
    stats = sprintf("Summary of %s per %s.", value, group),
    change = sprintf("Change in %s per %s between periods.", value, group),
    top = sprintf("Top %d rows by %s per %s.", n, value, group)
  )
}

#' Row counts per group
#'
#' @param data A data frame.
#' @param group Name of the grouping column.
#' @param value Name of the numeric column (used to count missing values).
#' @return A tibble with columns `group`, `rows`, `missing`, `missing_pct`,
#'   `distinct`, `share` and `cumshare`.
#' @importFrom dplyr group_by summarise mutate arrange n n_distinct
#' @importFrom magrittr %>%
#' @export
group.counts.table = function(data, group, value) {
  data %>%
    group_by(group = .data[[group]]) %>%
    summarise(
      rows = n(),
      missing = sum(is.na(.data[[value]])),
      distinct = n_distinct(.data[[value]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(group) %>%
    mutate(
      missing_pct = round(100 * missing / rows, 1),
      share = round(100 * rows / sum(rows), 1),
      cumshare = cumsum(share)
    ) %>%
    dplyr::select(group, rows, missing, missing_pct, distinct, share, cumshare)
}

#' Summary statistics per group
#'
#' @inheritParams group.counts.table
#' @param digits Number of decimal places.
#' @return A tibble with columns `group`, `n`, `mean`, `sd`, `min`, `q1`,
#'   `median`, `q3` and `max`.
#' @export
group.stats.table = function(data, group, value, digits = 1L) {
  data %>%
    group_by(group = .data[[group]]) %>%
    summarise(
      n = sum(!is.na(.data[[value]])),
      mean = mean(.data[[value]], na.rm = TRUE),
      sd = stats::sd(.data[[value]], na.rm = TRUE),
      min = min(.data[[value]], na.rm = TRUE),
      q1 = stats::quantile(.data[[value]], 0.25, na.rm = TRUE, names = FALSE),
      median = stats::median(.data[[value]], na.rm = TRUE),
      q3 = stats::quantile(.data[[value]], 0.75, na.rm = TRUE, names = FALSE),
      max = max(.data[[value]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(dplyr::across(mean:max, function(x) round(x, digits))) %>%
    arrange(group)
}

#' Change between two periods per group
#'
#' @inheritParams group.counts.table
#' @param period Name of the column that holds the period labels.
#' @param before,after The two period labels to compare.
#' @return A tibble with columns `group`, `before_n`, `before_mean`,
#'   `before_sd`, `after_n`, `after_mean`, `after_sd`, `change`,
#'   `change_pct` and `change_sd`.
#' @export
group.change.table = function(data, group, value, period, before, after) {
  data %>%
    dplyr::filter(.data[[period]] %in% c(before, after)) %>%
    # Flag the rows of each period once
    mutate(
      .x = .data[[value]],
      .before = .data[[period]] == before,
      .after = .data[[period]] == after
    ) %>%
    group_by(group = .data[[group]]) %>%
    summarise(
      before_n = sum(.before & !is.na(.x)),
      before_mean = mean(.x[.before], na.rm = TRUE),
      before_sd = stats::sd(.x[.before], na.rm = TRUE),
      after_n = sum(.after & !is.na(.x)),
      after_mean = mean(.x[.after], na.rm = TRUE),
      after_sd = stats::sd(.x[.after], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      change = after_mean - before_mean,
      change_pct = dplyr::if_else(before_mean != 0, 100 * change / before_mean, NA_real_),
      change_sd = after_sd - before_sd
    ) %>%
    arrange(group)
}

#' Largest rows per group
#'
#' @inheritParams group.counts.table
#' @param n Number of rows to keep per group.
#' @return A tibble with columns `group`, `rank` and `value`.
#' @export
group.top.table = function(data, group, value, n = 3L) {
  data %>%
    dplyr::filter(!is.na(.data[[value]])) %>%
    group_by(group = .data[[group]]) %>%
    dplyr::slice_max(.data[[value]], n = n, with_ties = FALSE) %>%
    mutate(rank = dplyr::row_number(), value = .data[[value]]) %>%
    dplyr::ungroup() %>%
    dplyr::select(group, rank, value) %>%
    arrange(group, rank)
}

#' Build several per-group tables
#'
#' @inheritParams group.counts.table
#' @param tables Which tables to build.
#' @param ... Further arguments: `period`, `before` and `after` for the
#'   `"change"` table, `n` for the `"top"` table.
#' @return A named list with one element per table, each a list with
#'   elements `caption` and `data`.
#' @export
group.tables = function(data, group, value,
                        tables = c("counts", "stats", "change", "top"), ...) {
  tables = match.arg(tables, several.ok = TRUE)
  args = list(...)
  n = if (is.null(args$n)) 3L else args$n

  # The change table needs the period column and both labels
  if ("change" %in% tables) {
    for (arg in c("period", "before", "after")) {
      if (is.null(args[[arg]])) {
        stop(sprintf("`%s` is required for the change table.", arg), call. = FALSE)
      }
    }
  }

  out = purrr::map(tables, function(table) {
    built = switch(table,
      counts = group.counts.table(data, group, value),
      stats = group.stats.table(data, group, value),
      change = group.change.table(data, group, value, args$period, args$before, args$after),
      top = group.top.table(data, group, value, n)
    )
    list(caption = table.caption(table, group, value, n), data = built)
  })
  stats::setNames(out, tables)
}

#' Split a grouped table into one data frame per group
#'
#' @param table A data frame with a `group` column, such as the result of
#'   [group.stats.table()].
#' @return A named list of data frames without the `group` column, one per
#'   group in order of first appearance.
#' @export
per.group.frames = function(table) {
  groups = unique(as.character(table$group))
  out = lapply(groups, function(g) {
    rows = table[as.character(table$group) == g, setdiff(names(table), "group"), drop = FALSE]
    rownames(rows) = NULL
    rows
  })
  names(out) = groups
  out
}
