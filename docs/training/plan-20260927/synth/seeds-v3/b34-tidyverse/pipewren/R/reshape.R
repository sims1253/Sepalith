#' Reshape a wide table to long form with base R
#'
#' @param data A data frame.
#' @param idCols Columns that identify a row.
#' @param measureCols Columns to stack. Defaults to every other column.
#' @param keyName,valueName Names of the key and value columns.
#' @param dropNa Drop rows whose value is missing?
#' @return A data frame with `idCols`, `keyName` and `valueName`, with the
#'   rows for each measure stacked in the order of `measureCols`.
#' @family reshaping
#' @export
meltBase <- function(data, idCols, measureCols = NULL,
                     keyName = "key", valueName = "value", dropNa = FALSE) {
  if (is.null(measureCols)) {
    measureCols <- setdiff(names(data), idCols)
  }
  unknown <- setdiff(c(idCols, measureCols), names(data))
  if (length(unknown) > 0L) {
    stop("Unknown columns: ", toString(unknown), call. = FALSE)
  }

  # One block of rows per measure column
  pieces <- lapply(measureCols, function(col) {
    piece <- data[idCols]
    piece[[keyName]] <- rep(col, nrow(data))
    piece[[valueName]] <- data[[col]]
    piece
  })
  out <- do.call(rbind, pieces)
  if (dropNa) {
    out <- out[!is.na(out[[valueName]]), , drop = FALSE]
  }
  rownames(out) <- NULL
  out
}

#' Reshape a long table to wide form with base R
#'
#' @param data A data frame in long form.
#' @param idCols Columns that identify a row.
#' @param keyName,valueName Names of the key and value columns.
#' @param fill Value used for combinations that do not occur.
#' @return A data frame with `idCols` followed by one column per distinct
#'   key, in order of first appearance.
#' @family reshaping
#' @export
castBase <- function(data, idCols, keyName = "key", valueName = "value", fill = NA) {
  unknown <- setdiff(c(idCols, keyName, valueName), names(data))
  if (length(unknown) > 0L) {
    stop("Unknown columns: ", toString(unknown), call. = FALSE)
  }

  # One output row per distinct combination of the id columns
  rowId <- do.call(paste, c(data[idCols], sep = "|"))
  if (anyDuplicated(paste(rowId, data[[keyName]], sep = "|"))) {
    warning("Duplicate keys within an id; keeping the first value.", call. = FALSE)
  }
  keys <- unique(data[[keyName]])
  out <- data[!duplicated(rowId), idCols, drop = FALSE]
  outId <- rowId[!duplicated(rowId)]
  for (key in keys) {
    hit <- data[[keyName]] == key
    value <- data[[valueName]][hit][match(outId, rowId[hit])]
    value[is.na(match(outId, rowId[hit]))] <- fill
    out[[key]] <- value
  }
  rownames(out) <- NULL
  out
}

#' Split a delimited column into several columns
#'
#' @param data A data frame.
#' @param col Name of the column to split.
#' @param into Names of the new columns.
#' @param sep A fixed separator string.
#' @param extra What to do with pieces beyond `length(into)`: `"drop"` them
#'   or `"merge"` them into the last column.
#' @param fill Where to put missing pieces when there are too few:
#'   `"right"` fills the last columns with `NA`, `"left"` the first ones.
#' @return `data` without `col` and with the columns `into` appended.
#' @family reshaping
#' @export
splitColumn <- function(data, col, into, sep = "_",
                        extra = c("drop", "merge"), fill = c("right", "left")) {
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
  })

  data[[col]] <- NULL
  for (i in seq_len(k)) {
    data[[into[i]]] <- vapply(parts, `[`, character(1), i)
  }
  data
}

#' Reshape metric tables with tidyr
#'
#' Metric columns are named `<metric>_<stat>`, as produced by
#' [summariseSales()].
#'
#' @param data A data frame of summaries.
#' @param idCols Columns that identify a row.
#' @param shape Target shape: `"longer"` gives columns `metric`, `stat` and
#'   `value`; `"wider"` reverses `"longer"`; `"nested"` nests everything but
#'   `idCols` into a list column `metrics`; `"unnested"` reverses
#'   `"nested"`.
#' @return A tibble in the requested shape, sorted by `idCols`.
#' @family reshaping
#' @importFrom tidyr pivot_longer pivot_wider nest unnest
#' @export
reshapeMetrics <- function(data, idCols,
                           shape = c("longer", "wider", "nested", "unnested")) {
  shape <- match.arg(shape)

  out <- switch(shape,
    longer = {
      data |>
        pivot_longer(
          cols = -dplyr::all_of(idCols),
          names_to = c("metric", "stat"),
          names_pattern = "^(.*)_([^_]+)$",
          values_to = "value"
        )
    },
    wider = {
      data |>
        pivot_wider(
          id_cols = dplyr::all_of(idCols),
          names_from = c(metric, stat),
          names_sep = "_",
          values_from = value
        )
    },
    nested = {
      data |>
        nest(metrics = -dplyr::all_of(idCols))
    },
    unnested = {
      data |>
        unnest(cols = metrics)
    }
  )

  # Sort by the id columns whatever the shape
  dplyr::arrange(out, dplyr::across(dplyr::all_of(idCols)))
}

#' Complete a store by month grid
#'
#' @param data A data frame of sales with columns `store`, `month`, `units`,
#'   `revenue`, `cost` and `discount`.
#' @param months The months that every store should have.
#' @return A tibble with one row per store and month. Missing `units`,
#'   `revenue`, `cost` and `discount` are filled with zero and flagged in
#'   `units_imputed`, `revenue_imputed`, `cost_imputed` and
#'   `discount_imputed`; `imputed` is `TRUE` for rows added by the grid.
#' @family reshaping
#' @export
completeGrid <- function(data, months) {
  data |>
    dplyr::mutate(.observed = TRUE) |>
    tidyr::complete(store, month = months, fill = list(.observed = FALSE)) |>
    dplyr::mutate(
      imputed = !.observed,
      units_imputed = is.na(units),
      units = tidyr::replace_na(units, 0),
      revenue_imputed = is.na(revenue),
      revenue = tidyr::replace_na(revenue, 0),
      cost_imputed = is.na(cost),
      cost = tidyr::replace_na(cost, 0),
      discount_imputed = is.na(discount),
      discount = tidyr::replace_na(discount, 0)
    ) |>
    dplyr::select(-.observed) |>
    dplyr::arrange(store, month)
}

#' Nest sales by group and summarise each nest
#'
#' @param data A data frame of sales.
#' @param group Name of the grouping column.
#' @param cols Numeric columns passed to [summaryTable()].
#' @return A tibble with the grouping column, the list columns `data` and
#'   `summary`, and the counts `rows`, `complete` and `incomplete`.
#' @family reshaping
#' @export
nestByGroup <- function(data, group, cols) {
  data |>
    tidyr::nest(.by = dplyr::all_of(group)) |>
    dplyr::mutate(
      summary = purrr::map(data, \(d) summaryTable(d, cols)),
      rows = purrr::map_int(data, nrow),
      complete = purrr::map_int(data, \(d) sum(stats::complete.cases(d[cols]))),
      incomplete = rows - complete
    )
}
