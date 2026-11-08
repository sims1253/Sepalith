#' Compute the headline KPIs
#'
#' @param data A data frame of orders with columns `order_id`,
#'   `customer_id`, `amount`, `sessions` and `duration`.
#' @return A list with elements `revenue`, `orders`, `customers`,
#'   `conversion`, `order_value` and `duration`.
#' @family value boxes
#' @export
kpi_values <- function(data) {
  # 1. revenue: total amount
  revenue <- sum(data$amount, na.rm = TRUE)

  # 2. orders: distinct order ids
  orders <- length(unique(data$order_id[!is.na(data$order_id)]))

  # 3. customers: distinct customer ids
  customers <- length(unique(data$customer_id[!is.na(data$customer_id)]))

  # 4. conversion: orders per session
  sessions <- sum(data$sessions, na.rm = TRUE)
  conversion <- if (sessions > 0) orders / sessions else NA_real_

  # 5. order_value: revenue per order
  order_value <- if (orders > 0) revenue / orders else NA_real_

  # 6. duration: median session length in seconds
  duration <- if (nrow(data) > 0) stats::median(data$duration, na.rm = TRUE) else NA_real_
  list(
    revenue = revenue,
    orders = orders,
    customers = customers,
    conversion = conversion,
    order_value = order_value,
    duration = duration
  )
}

#' Select comparison periods for a date range
#'
#' The previous period has the same number of days and ends the day before
#' `start`; the year-ago period covers the same days one year earlier.
#'
#' @param data A data frame of orders with a `date` column.
#' @param start First day of the current period.
#' @param end Last day of the current period.
#' @return A list with elements `current`, `previous` and `year_ago`, each a
#'   data frame.
#' @family value boxes
#' @export
kpi_periods <- function(data, start, end) {
  start <- as.Date(start)
  end <- as.Date(end)
  if (end < start) {
    stop("`end` must not be before `start`.", call. = FALSE)
  }
  span <- as.integer(end - start) + 1L
  day <- as.Date(data$date)

  # Current period
  current_from <- start
  current_to <- end
  current <- data[!is.na(day) & day >= current_from & day <= current_to, , drop = FALSE]

  # Previous period
  previous_from <- start - span
  previous_to <- start - 1L
  previous <- data[!is.na(day) & day >= previous_from & day <= previous_to, , drop = FALSE]

  # Year-ago period
  year_ago_from <- seq(start, by = "-1 year", length.out = 2L)[[2]]
  year_ago_to <- seq(end, by = "-1 year", length.out = 2L)[[2]]
  year_ago <- data[!is.na(day) & day >= year_ago_from & day <= year_ago_to, , drop = FALSE]
  list(current = current, previous = previous, year_ago = year_ago)
}

#' Collect KPI targets from numeric inputs
#'
#' Empty inputs (`NULL` or `NA`) mean that the KPI has no target.
#'
#' @param revenue Target for `revenue`.
#' @param orders Target for `orders`.
#' @param customers Target for `customers`.
#' @param conversion Target for `conversion`, a share between 0 and 1.
#' @param order_value Target for `order_value`.
#' @return A named list holding the targets that were given, suitable for
#'   [kpi_status()].
#' @family value boxes
#' @export
kpi_targets <- function(revenue = NULL, orders = NULL, customers = NULL,
                        conversion = NULL, order_value = NULL) {
  targets <- list()

  # Target for `revenue`
  if (length(revenue) == 1L && !is.na(revenue)) {
    if (revenue < 0) {
      stop(sprintf("The `%s` target must not be negative.", "revenue"), call. = FALSE)
    }
    targets$revenue <- revenue
  }

  # Target for `orders`
  if (length(orders) == 1L && !is.na(orders)) {
    if (orders < 0) {
      stop(sprintf("The `%s` target must not be negative.", "orders"), call. = FALSE)
    }
    targets$orders <- orders
  }

  # Target for `customers`
  if (length(customers) == 1L && !is.na(customers)) {
    if (customers < 0) {
      stop(sprintf("The `%s` target must not be negative.", "customers"), call. = FALSE)
    }
    targets$customers <- customers
  }

  # Target for `conversion`
  if (length(conversion) == 1L && !is.na(conversion)) {
    if (conversion < 0 || conversion > 1) {
      stop(
        sprintf("The `%s` target must be a share between 0 and 1.", "conversion"),
        call. = FALSE
      )
    }
    targets$conversion <- conversion
  }

  # Target for `order_value`
  if (length(order_value) == 1L && !is.na(order_value)) {
    if (order_value < 0) {
      stop(sprintf("The `%s` target must not be negative.", "order_value"), call. = FALSE)
    }
    targets$order_value <- order_value
  }
  targets
}

#' Compare KPIs with their targets
#'
#' @param values A list as returned by [kpi_values()].
#' @param targets A named list of targets, using the names of `values`.
#'   KPIs without a target are reported as `"none"`.
#' @param tolerance Relative distance from a target that still counts as
#'   on target.
#' @return A named character vector with one of `"above"`, `"on"`,
#'   `"below"` or `"none"` per KPI.
#' @family value boxes
#' @export
kpi_status <- function(values, targets, tolerance = 0.05) {
  kpis <- c("revenue", "orders", "customers", "conversion", "order_value")
  status <- stats::setNames(rep("none", length(kpis)), kpis)
  for (kpi in kpis) {
    value <- values[[kpi]]
    target <- targets[[kpi]]
    if (is.null(target) || is.na(target) || is.na(value)) {
      next
    }
    lower <- target * (1 - tolerance)
    upper <- target * (1 + tolerance)
    if (value > upper) {
      status[[kpi]] <- "above"
    } else if (value < lower) {
      status[[kpi]] <- "below"
    } else {
      status[[kpi]] <- "on"
    }
  }
  status
}

#' Render one value box
#'
#' @param spec A list as returned by [value_box_spec()].
#' @return A `shiny::div()` holding the title, value and change.
#' @family value boxes
#' @keywords internal
render_box <- function(spec) {
  shiny::div(
    class = paste("dk-valuebox", paste0("dk-", spec$color)),
    shiny::div(class = "dk-valuebox-title", spec$title),
    shiny::div(class = "dk-valuebox-value", spec$value),
    shiny::div(
      class = "dk-valuebox-change",
      shiny::icon(spec$icon),
      spec$change
    )
  )
}

#' Value box row UI
#'
#' @param id Module id.
#' @return A `shiny::fluidRow()` with one output per KPI.
#' @family value boxes
#' @export
valueboxes_ui <- function(id) {
  ns <- shiny::NS(id)
  shiny::fluidRow(
    shiny::column(2, shiny::uiOutput(ns("revenue"))),
    shiny::column(2, shiny::uiOutput(ns("orders"))),
    shiny::column(2, shiny::uiOutput(ns("customers"))),
    shiny::column(2, shiny::uiOutput(ns("conversion"))),
    shiny::column(2, shiny::uiOutput(ns("order_value"))),
    shiny::column(2, shiny::uiOutput(ns("duration")))
  )
}

#' Value box row server
#'
#' @param id Module id.
#' @param data A reactive returning the order data frame.
#' @param dates A reactive returning the selected first and last day.
#' @return `NULL`, invisibly; called for its outputs.
#' @family value boxes
#' @export
valueboxes_server <- function(id, data, dates) {
  titles <- c(
    revenue = "Revenue", orders = "Orders", customers = "Customers",
    conversion = "Conversion", order_value = "Order value", duration = "Session length"
  )
  types <- c(
    revenue = "currency", orders = "count", customers = "count",
    conversion = "percent", order_value = "currency", duration = "duration"
  )
  shiny::moduleServer(id, function(input, output, session) {
    kpis <- shiny::reactive({
      shiny::req(dates())
      periods <- kpi_periods(data(), dates()[[1]], dates()[[2]])
      list(
        current = kpi_values(periods$current),
        previous = kpi_values(periods$previous)
      )
    })

    output$revenue <- shiny::renderUI({
      k <- kpis()
      render_box(value_box_spec(k$current$revenue, k$previous$revenue,
                                types[["revenue"]], titles[["revenue"]]))
    })
    output$orders <- shiny::renderUI({
      k <- kpis()
      render_box(value_box_spec(k$current$orders, k$previous$orders,
                                types[["orders"]], titles[["orders"]]))
    })
    output$customers <- shiny::renderUI({
      k <- kpis()
      render_box(value_box_spec(k$current$customers, k$previous$customers,
                                types[["customers"]], titles[["customers"]]))
    })
    output$conversion <- shiny::renderUI({
      k <- kpis()
      render_box(value_box_spec(k$current$conversion, k$previous$conversion,
                                types[["conversion"]], titles[["conversion"]]))
    })
    output$order_value <- shiny::renderUI({
      k <- kpis()
      render_box(value_box_spec(k$current$order_value, k$previous$order_value,
                                types[["order_value"]], titles[["order_value"]]))
    })
    output$duration <- shiny::renderUI({
      k <- kpis()
      render_box(value_box_spec(k$current$duration, k$previous$duration,
                                types[["duration"]], titles[["duration"]]))
    })
    invisible(NULL)
  })
}
