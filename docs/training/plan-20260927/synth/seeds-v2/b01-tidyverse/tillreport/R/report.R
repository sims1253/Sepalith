#' Format numbers as currency
#'
#' @export
#' @param x A numeric vector.
#' @param symbol Currency symbol placed before the amount.
#' @param digits Number of decimal places to show.
#' @return A character vector the same length as `x`.
formatCurrency <- function(x, symbol = "EUR ", digits = 2L) {
  # Format the absolute value so the minus sign ends up before the symbol.
  out <- paste0(symbol, formatC(abs(x), format = "f", digits = digits, big.mark = ","))
  out <- ifelse(!is.na(x) & x < 0, paste0("-", out), out)
  ifelse(is.na(x), "-", out)
}

#' Format proportions as percentages
#'
#' @export
#' @param x A numeric vector.
#' @param digits Number of decimal places to show.
#' @return A character vector the same length as `x`.
formatPercent <- function(x, digits = 1L) {
  out <- paste0(formatC(100 * x, format = "f", digits = digits), "%")
  ifelse(is.na(x), "-", out)
}

#' Format changes with an explicit sign
#'
#' @export
#' @param x A numeric vector.
#' @param digits Number of decimal places to show.
#' @return A character vector the same length as `x`.
formatDelta <- function(x, digits = 1L) {
  out <- formatC(100 * x, format = "f", digits = digits, flag = "+")
  out <- paste0(out, " pp")
  ifelse(is.na(x), "-", out)
}

#' Create a sales report object
#'
#' Bundles a summarised sales table with the metadata needed to print it,
#' so the same summary can be rendered to the console or exported later.
#'
#' @export
#' @param summary A summarised sales table, e.g. from [summariseSales()].
#' @param title Report title.
#' @param from,to First and last date covered by the report.
#' @return An object of class `salesReport`.
newSalesReport <- function(summary, title, from, to) {
  stopifnot(
    is.data.frame(summary),
    is.character(title), length(title) == 1,
    lubridate::as_date(from) <= lubridate::as_date(to)
  )
  structure(
    list(
      summary = tibble::as_tibble(summary),
      title = title,
      from = lubridate::as_date(from),
      to = lubridate::as_date(to),
      created = Sys.time()
    ),
    class = "salesReport"
  )
}

#' Format a sales report for display
#'
#' @export
#' @param x A `salesReport` object.
#' @param n Maximum number of summary rows to include.
#' @param ... Ignored.
#' @return A character vector of lines.
format.salesReport <- function(x, n = 10L, ...) {
  shown <- utils::head(x$summary, n)

  # Only the columns we know how to format are touched; anything else is
  # printed as it comes.
  if ("revenue" %in% names(shown)) {
    shown$revenue <- formatCurrency(shown$revenue)
  }
  if ("share" %in% names(shown)) {
    shown$share <- formatPercent(shown$share)
  }
  if ("pctChange" %in% names(shown)) {
    shown$pctChange <- formatDelta(shown$pctChange)
  }

  header <- c(
    x$title,
    sprintf("Period: %s to %s", format(x$from), format(x$to)),
    strrep("-", nchar(x$title))
  )
  body <- utils::capture.output(print(as.data.frame(shown), row.names = FALSE))
  footer <- if (nrow(x$summary) > n) {
    sprintf("... %d more rows", nrow(x$summary) - n)
  }

  c(header, body, footer)
}

#' Print a sales report
#'
#' @export
#' @param x A `salesReport` object.
#' @param ... Passed to [format.salesReport()].
#' @return `x`, invisibly.
print.salesReport <- function(x, ...) {
  cat(format(x, ...), sep = "\n")
  invisible(x)
}

#' Summarise a sales report
#'
#' @export
#' @param object A `salesReport` object.
#' @param ... Ignored.
#' @return A named list with the number of rows, days covered and totals.
summary.salesReport <- function(object, ...) {
  tbl <- object$summary
  totals <- list(
    rows = nrow(tbl),
    days = as.integer(object$to - object$from) + 1L,
    revenue = if ("revenue" %in% names(tbl)) sum(tbl$revenue, na.rm = TRUE) else NA_real_,
    units = if ("units" %in% names(tbl)) sum(tbl$units, na.rm = TRUE) else NA_real_
  )

  totals$revenuePerDay <- totals$revenue / totals$days
  totals
}

#' Build a period-over-period sales report
#'
#' Summarises sales for the current period and the one before it, by the
#' requested dimension, and returns a `salesReport` ready to print.
#'
#' @export
#' @param sales A data frame of till transactions with `revenue`, `units`,
#'   `transactionId` and `saleDate` columns.
#' @param asOf Last date to include in the current period.
#' @param unit One of `"week"`, `"month"` or `"quarter"`.
#' @param by One of `"store"` or `"category"`.
#' @return A `salesReport` object whose summary includes `pctChange`.
#' @importFrom lubridate %m-%
buildPeriodReport <- function(sales, asOf = Sys.Date(), unit = c("week", "month", "quarter"),
                              by = c("store", "category")) {
  unit <- match.arg(unit)
  by <- match.arg(by)
  asOf <- lubridate::as_date(asOf)

  # The current period runs to `asOf`; the previous one is a full period.
  currentStart <- periodStart(asOf, unit)
  previousStart <- switch(unit,
    week = currentStart - lubridate::weeks(1),
    month = currentStart %m-% lubridate::period(months = 1),
    quarter = currentStart %m-% lubridate::period(months = 3)
  )
  summariseBy <- switch(by,
    store = summariseByStore,
    category = summariseByCategory
  )

  current <- sales |>
    dplyr::filter(.data$saleDate >= currentStart, .data$saleDate <= asOf) |>
    summariseBy()
  previous <- sales |>
    dplyr::filter(.data$saleDate >= previousStart, .data$saleDate < currentStart) |>
    summariseBy() |>
    dplyr::select(1, previousRevenue = "revenue")

  # The grouping column always comes first in a summariseSales() result.
  keyCol <- names(current)[[1]]
  combined <- current |>
    dplyr::left_join(previous, by = keyCol) |>
    dplyr::mutate(
      pctChange = dplyr::if_else(
        is.na(.data$previousRevenue) | .data$previousRevenue == 0,
        NA_real_,
        .data$revenue / .data$previousRevenue - 1
      )
    ) |>
    addShare() |>
    dplyr::select(dplyr::all_of(keyCol), "revenue", "share", "pctChange", "transactions")

  title <- sprintf("Sales by %s, %s to date", by, unit)
  newSalesReport(combined, title = title, from = currentStart, to = asOf)
}
