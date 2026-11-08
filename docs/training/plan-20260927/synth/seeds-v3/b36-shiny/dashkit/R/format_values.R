#' Format a KPI value for display
#'
#' @param x A numeric vector of KPI values.
#' @param type Kind of KPI: one of `"currency"`, `"percent"`, `"count"`,
#'   `"duration"` or `"ratio"`.
#' @param digits Number of decimal places.
#' @param currency Currency symbol used when `type` is `"currency"`.
#' @return A character vector the same length as `x`; `NA` values become
#'   `"-"`.
#' @family formatters
#' @export
format_kpi <- function(x, type = c("currency", "percent", "count", "duration", "ratio"),
                       digits = 1, currency = "$") {
  type <- match.arg(type)
  out <- switch(type,
    currency = {
      body <- formatC(abs(x), format = "f", digits = digits, big.mark = ",")
      sign <- ifelse(x < 0, "-", "")
      paste0(sign, currency, body)
    },
    percent = {
      body <- formatC(100 * x, format = "f", digits = digits)
      paste0(body, "%")
    },
    count = {
      body <- formatC(round(x), format = "d", big.mark = ",")
      trimws(body)
    },
    duration = {
      minutes <- floor(x / 60)
      seconds <- round(x - 60 * minutes)
      sprintf("%dm %02ds", as.integer(minutes), as.integer(seconds))
    },
    ratio = {
      body <- formatC(x, format = "f", digits = digits)
      paste0(body, "x")
    }
  )
  out[is.na(x)] <- "-"
  out
}

#' Describe the change between two periods
#'
#' @param current Value in the current period.
#' @param previous Value in the previous period.
#' @param digits Number of decimal places in the percentage.
#' @return A list with elements `text`, `direction` (`"up"`, `"down"` or
#'   `"flat"`) and `icon`.
#' @family formatters
#' @export
format_change <- function(current, previous, digits = 1) {
  if (is.na(current) || is.na(previous) || previous == 0) {
    return(list(text = "n/a", direction = "flat", icon = "minus"))
  }
  change <- (current - previous) / abs(previous)
  if (change > 0) {
    text <- paste0("+", formatC(100 * change, format = "f", digits = digits), "%")
    direction <- "up"
    icon <- "arrow-up"
  } else if (change < 0) {
    text <- paste0("-", formatC(-100 * change, format = "f", digits = digits), "%")
    direction <- "down"
    icon <- "arrow-down"
  } else {
    text <- paste0(formatC(0, format = "f", digits = digits), "%")
    direction <- "flat"
    icon <- "minus"
  }
  list(text = text, direction = direction, icon = icon)
}

#' Abbreviate large numbers
#'
#' Values of a thousand or more are shortened with the suffixes `K`, `M`
#' and `B`.
#'
#' @param x A numeric vector.
#' @param digits Number of decimal places kept after scaling.
#' @return A character vector the same length as `x`.
#' @family formatters
#' @export
format_compact <- function(x, digits = 1) {
  units <- c(K = 1e3, M = 1e6, B = 1e9)
  out <- character(length(x))
  for (i in seq_along(x)) {
    value <- x[[i]]
    size <- abs(value)
    if (is.na(value)) {
      out[[i]] <- "-"
    } else if (size >= units[["B"]]) {
      out[[i]] <- paste0(formatC(value / units[["B"]], format = "f", digits = digits), "B")
    } else if (size >= units[["M"]]) {
      out[[i]] <- paste0(formatC(value / units[["M"]], format = "f", digits = digits), "M")
    } else if (size >= units[["K"]]) {
      out[[i]] <- paste0(formatC(value / units[["K"]], format = "f", digits = digits), "K")
    } else {
      out[[i]] <- formatC(value, format = "f", digits = 0)
    }
  }
  out
}

#' Label a reporting period
#'
#' @param start First day of the period, a `Date`.
#' @param by Length of the period: one of `"day"`, `"week"`, `"month"`,
#'   `"quarter"` or `"year"`.
#' @return A single string such as `"Week of 2024-03-04"` or `"Q1 2024"`.
#' @family formatters
#' @export
format_period_label <- function(start, by = c("day", "week", "month", "quarter", "year")) {
  by <- match.arg(by)
  start <- as.Date(start)
  year <- format(start, "%Y")
  switch(by,
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
format_table_columns <- function(data, currency = character(0), percent = character(0),
                                 count = character(0), date = character(0)) {
  wanted <- c(currency, percent, count, date)
  missing <- setdiff(wanted, names(data))
  if (length(missing) > 0L) {
    stop("Unknown columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }

  # Format `currency` columns
  for (col in currency) {
    data[[col]] <- format_kpi(data[[col]], "currency", digits = 2)
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
    data[[col]] <- ifelse(is.na(data[[col]]), "-", format(as.Date(data[[col]]), "%d %b %Y"))
  }
  data
}

#' Build the display specification of a value box
#'
#' @param value Value in the current period.
#' @param previous Value in the previous period.
#' @param type Kind of KPI, passed to [format_kpi()].
#' @param title Title shown above the value.
#' @return A list with elements `title`, `value`, `change`, `direction`,
#'   `icon` and `color`.
#' @family formatters
#' @export
value_box_spec <- function(value, previous, type, title) {
  change <- format_change(value, previous)
  color <- switch(change$direction,
    up = "green",
    down = "red",
    flat = "gray"
  )
  list(
    title = title,
    value = format_kpi(value, type),
    change = change$text,
    direction = change$direction,
    icon = change$icon,
    color = color
  )
}

#' Tabulate KPI changes between two periods
#'
#' @param current A list with elements `revenue`, `orders`, `conversion` and
#'   `order_value` for the current period.
#' @param previous A list with the same elements for the previous period.
#' @return A data frame with columns `kpi`, `current`, `previous` and
#'   `change`, one row per KPI.
#' @family formatters
#' @export
format_delta_table <- function(current, previous) {
  types <- c(revenue = "currency", orders = "count", conversion = "percent",
             order_value = "currency")
  rows <- vector("list", length(types))
  names(rows) <- names(types)

  # Row for `revenue`
  rows$revenue <- data.frame(
    kpi = "revenue",
    current = format_kpi(current$revenue, types[["revenue"]]),
    previous = format_kpi(previous$revenue, types[["revenue"]]),
    change = format_change(current$revenue, previous$revenue)$text
  )

  # Row for `orders`
  rows$orders <- data.frame(
    kpi = "orders",
    current = format_kpi(current$orders, types[["orders"]]),
    previous = format_kpi(previous$orders, types[["orders"]]),
    change = format_change(current$orders, previous$orders)$text
  )

  # Row for `conversion`
  rows$conversion <- data.frame(
    kpi = "conversion",
    current = format_kpi(current$conversion, types[["conversion"]]),
    previous = format_kpi(previous$conversion, types[["conversion"]]),
    change = format_change(current$conversion, previous$conversion)$text
  )

  # Row for `order_value`
  rows$order_value <- data.frame(
    kpi = "order_value",
    current = format_kpi(current$order_value, types[["order_value"]]),
    previous = format_kpi(previous$order_value, types[["order_value"]]),
    change = format_change(current$order_value, previous$order_value)$text
  )
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}
