#' Category budget module UI
#'
#' Controls for choosing a month and setting a spending limit per category,
#' alongside a table and chart comparing limits with actual spending.
#'
#' @export
#' @param id The module namespace id.
#' @param categories A character vector of spending categories.
#' @return A [bslib::layout_columns()] containing the module inputs and outputs.
categoryBudgetUI = function(id, categories) {
  ns = shiny::NS(id)
  bslib::layout_columns(
    col_widths = c(4, 8),
    bslib::card(
      bslib::card_header("Limits"),
      shiny::selectInput(ns("month"), "Month", choices = NULL),
      shiny::selectInput(ns("category"), "Category", choices = categories),
      shiny::numericInput(ns("limit"), "Monthly limit", value = NA, min = 0, step = 10),
      shiny::actionButton(ns("setLimit"), "Save limit", class = "btn-primary"),
      shiny::textOutput(ns("message"))
    ),
    bslib::card(
      bslib::card_header("Spending vs. budget"),
      shiny::tableOutput(ns("varianceTable")),
      shiny::plotOutput(ns("variancePlot"), height = "260px")
    )
  )
}

#' Category budget module server
#'
#' Keeps the per-category limits in a reactive value, validates new limits
#' and compares them with the spending recorded for the chosen month.
#'
#' @export
#' @param id The module namespace id.
#' @param transactions A reactive returning a data frame of transactions.
#' @param limits A named numeric vector of starting limits per category.
#' @return A reactive returning the current named vector of limits.
categoryBudgetServer = function(id, transactions, limits) {
  stopifnot(shiny::is.reactive(transactions), is.numeric(limits))
  shiny::moduleServer(id, function(input, output, session) {
    currentLimits = shiny::reactiveVal(limits)
    message = shiny::reactiveVal("")

    # newest month first, and select it so the table is filled straight away
    shiny::observe({
      months = sort(unique(format(transactions()$date, "%Y-%m")), decreasing = TRUE)
      shiny::updateSelectInput(session, "month", choices = months, selected = months[1])
    })

    # show the saved limit for whichever category is picked
    shiny::observeEvent(input$category, {
      shiny::updateNumericInput(session, "limit", value = unname(currentLimits()[input$category]))
    })

    shiny::observeEvent(input$setLimit, {
      if (is.null(input$limit) || is.na(input$limit) || input$limit < 0) {
        message("Enter a limit of zero or more.")
        return()
      }

      updated = currentLimits()
      updated[[input$category]] = input$limit
      currentLimits(updated)
      message(sprintf("Limit for %s set to %s.", input$category, formatMoney(input$limit)))
    })

    variance = shiny::reactive({
      shiny::req(input$month)
      spent = spendByCategory(transactions(), input$month)
      budgetVariance(spent, currentLimits())
    })

    output$message = shiny::renderText(message())
    output$varianceTable = shiny::renderTable({
      shiny::validate(shiny::need(nrow(variance()) > 0, "No limits have been set yet."))
      formatVarianceTable(variance())
    })
    output$variancePlot = shiny::renderPlot({
      shiny::req(nrow(variance()) > 0)
      plotVariance(variance())
    })

    shiny::reactive(currentLimits())
  })
}

#' Sum spending by category for one month
#'
#' Keeps outgoing transactions in `month` and totals their absolute amounts
#' per category.
#'
#' @keywords internal
#' @param transactions A data frame of transactions with columns `date`,
#'   `category` and `amount`, where spending is negative.
#' @param month A month in `"YYYY-MM"` form.
#' @return A named numeric vector of amounts spent.
spendByCategory = function(transactions, month) {
  # which() so that rows with a missing date or amount are simply left out
  keep = which(format(transactions$date, "%Y-%m") == month & transactions$amount < 0)
  inMonth = transactions[keep, , drop = FALSE]
  if (nrow(inMonth) == 0) {
    return(stats::setNames(numeric(), character()))
  }

  # spending is stored as negative amounts; report it as positive
  totals = tapply(-inMonth$amount, inMonth$category, sum)
  stats::setNames(as.numeric(totals), names(totals))
}

#' Compare spending with limits
#'
#' Lines up spending and limits by category; categories without spending
#' count as zero spent.
#'
#' @keywords internal
#' @param spent A named numeric vector of amounts spent.
#' @param limits A named numeric vector of limits per category.
#' @return A data frame with columns `category`, `limit`, `spent`,
#'   `remaining` and `status`.
budgetVariance = function(spent, limits) {
  categories = names(limits)
  # indexing by name gives NA for categories with no spending
  spentAligned = unname(spent[categories])
  spentAligned[is.na(spentAligned)] = 0
  remaining = unname(limits) - spentAligned

  data.frame(
    category = categories,
    limit = unname(limits),
    spent = spentAligned,
    remaining = remaining,
    status = varianceStatus(spentAligned, unname(limits)),
    stringsAsFactors = FALSE
  )
}

#' Classify spending against a limit
#'
#' A category is `"over"` once spending exceeds the limit and `"near"` once
#' it passes `warnAt` of the limit.
#'
#' @keywords internal
#' @param spent A numeric vector of amounts spent.
#' @param limit A numeric vector of limits, recycled to `spent`.
#' @param warnAt The fraction of the limit at which to warn.
varianceStatus = function(spent, limit, warnAt = 0.9) {
  status = rep("under", length(spent))
  status[spent >= warnAt * limit] = "near"
  status[spent > limit] = "over"
  status
}

#' Format an amount of money
#'
#' Rounds to cents and adds a thousands separator and currency symbol.
#'
#' @keywords internal
#' @param x A numeric vector of amounts.
#' @param symbol The currency symbol to prefix.
formatMoney = function(x, symbol = "$") {
  # put the sign in front of the symbol: -$5.00, not $-5.00
  sign = ifelse(x < 0, "-", "")
  body = formatC(abs(x), format = "f", digits = 2, big.mark = ",")
  paste0(sign, symbol, body)
}

#' Format the variance table for display
#'
#' Formats money columns and gives the columns readable headers.
#'
#' @keywords internal
#' @param variance A data frame as returned by [budgetVariance()].
formatVarianceTable = function(variance) {
  data.frame(
    Category = variance$category,
    Limit = formatMoney(variance$limit),
    Spent = formatMoney(variance$spent),
    Remaining = formatMoney(variance$remaining),
    Status = variance$status,
    stringsAsFactors = FALSE
  )
}

#' Plot spending against limits
#'
#' Draws spent amounts as bars coloured by status with the limit marked as
#' a point for each category.
#'
#' @keywords internal
#' @param variance A data frame as returned by [budgetVariance()].
plotVariance = function(variance) {
  # largest spending at the top of the chart
  variance$category = stats::reorder(variance$category, variance$spent)
  ggplot2::ggplot(variance, ggplot2::aes(y = category)) +
    ggplot2::geom_col(ggplot2::aes(x = spent, fill = status)) +
    ggplot2::geom_point(ggplot2::aes(x = limit), shape = 124, size = 6) +
    ggplot2::scale_fill_manual(values = c(under = "#4daf4a", near = "#ff7f00", over = "#e41a1c")) +
    ggplot2::labs(x = "Amount", y = NULL, fill = NULL) +
    ggplot2::theme_minimal(base_size = 13)
}
