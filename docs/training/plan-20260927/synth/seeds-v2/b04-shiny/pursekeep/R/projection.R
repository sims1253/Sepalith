#' Balance projection module UI
#'
#' Inputs describing a savings plan and a plot of the projected balance.
#'
#' @export
#' @param id The module namespace id.
#' @return A [bslib::layout_sidebar()] containing the module inputs and outputs.
projectionUI = function(id) {
  ns = shiny::NS(id)
  bslib::layout_sidebar(
    sidebar = bslib::sidebar(
      shiny::numericInput(ns("start"), "Starting balance", value = 1000, step = 100),
      shiny::numericInput(ns("monthly"), "Monthly saving", value = 200, step = 25),
      shiny::sliderInput(
        ns("rate"), "Annual interest (%)",
        min = 0, max = 10, value = 2, step = 0.25
      ),
      shiny::sliderInput(ns("months"), "Months ahead", min = 1, max = 120, value = 36)
    ),
    shiny::plotOutput(ns("balancePlot")),
    shiny::tableOutput(ns("milestones"))
  )
}

#' Balance projection module server
#'
#' Validates the plan inputs and projects the balance month by month.
#'
#' @export
#' @param id The module namespace id.
#' @return A reactive returning the projection data frame.
projectionServer = function(id) {
  shiny::moduleServer(id, function(input, output, session) {
    projection = shiny::reactive({
      shiny::validate(
        checkAmount(input$start, "Starting balance"),
        checkAmount(input$monthly, "Monthly saving"),
        checkRate(input$rate),
        checkMonths(input$months)
      )
      projectBalance(input$start, input$monthly, input$rate / 100, input$months)
    }) |>
      # wait for the slider to settle before recomputing
      shiny::debounce(400)

    output$balancePlot = shiny::renderPlot(plotProjection(projection()))
    output$milestones = shiny::renderTable(summarizeProjection(projection()), digits = 2)
    projection
  })
}

#' Check a money amount input
#'
#' Amounts must be present and zero or more.
#'
#' @keywords internal
#' @param value The value entered by the user.
#' @param label A human-readable field name used in the message.
#' @return `NULL` when valid, otherwise a message for [shiny::validate()].
checkAmount = function(value, label) {
  if (is.null(value) || is.na(value)) {
    return(paste(label, "is required."))
  }
  if (value < 0) {
    return(paste(label, "cannot be negative."))
  }
  NULL
}

#' Check an interest rate input
#'
#' Rates must be present and between 0 and 25 percent.
#'
#' @noRd
#' @param value The value entered by the user, in percent.
checkRate = function(value) {
  if (is.null(value) || is.na(value)) {
    return("Interest rate is required.")
  }
  if (value < 0 || value > 25) {
    return("Interest rate must be between 0% and 25%.")
  }
  NULL
}

#' Check a number of months input
#'
#' Months must be present and a whole number of at least one.
#'
#' @noRd
#' @param value The value entered by the user.
checkMonths = function(value) {
  if (is.null(value) || is.na(value)) {
    return("Number of months is required.")
  }
  if (value < 1 || value != round(value)) {
    return("Number of months must be a whole number of at least 1.")
  }
  NULL
}

#' Project a savings balance
#'
#' Applies monthly compounding interest and then adds the monthly saving,
#' one month at a time.
#'
#' @export
#' @param start The starting balance.
#' @param monthly The amount saved each month.
#' @param annualRate The annual interest rate as a fraction.
#' @param months The number of months to project.
#' @return A data frame with columns `month`, `contributed`, `interest` and `balance`.
projectBalance = function(start, monthly, annualRate, months) {
  monthlyRate = annualRate / 12
  balance = numeric(months)
  interest = numeric(months)
  current = start

  # interest is earned on last month's balance, before this month's saving
  for (i in seq_len(months)) {
    interest[i] = current * monthlyRate
    current = current + interest[i] + monthly
    balance[i] = current
  }

  data.frame(
    month = seq_len(months),
    contributed = start + monthly * seq_len(months),
    interest = cumsum(interest),
    balance = balance
  )
}

#' Summarise a projection at yearly milestones
#'
#' Picks the end of each year, plus the final month if it is not already a
#' year end.
#'
#' @keywords internal
#' @param projection A data frame as returned by [projectBalance()].
#' @return A data frame with one row per milestone.
summarizeProjection = function(projection) {
  last = max(projection$month)
  # seq(12, last, by = 12) fails for plans shorter than a year
  keep = unique(c(seq_len(last %/% 12) * 12, last))
  keep = keep[keep >= 1 & keep <= last]

  out = projection[projection$month %in% keep, c("month", "contributed", "interest", "balance")]
  names(out) = c("Month", "Contributed", "Interest earned", "Balance")
  out
}

#' Plot a projected balance
#'
#' Shows contributions and total balance over time as stacked areas.
#'
#' @keywords internal
#' @param projection A data frame as returned by [projectBalance()].
plotProjection = function(projection) {
  # long format with one row per month and part, for stacked areas
  long = data.frame(
    month = rep(projection$month, 2),
    part = rep(c("Contributed", "Interest"), each = nrow(projection)),
    amount = c(projection$contributed, projection$balance - projection$contributed)
  )
  ggplot2::ggplot(long, ggplot2::aes(x = month, y = amount, fill = part)) +
    ggplot2::geom_area(alpha = 0.85) +
    ggplot2::scale_fill_manual(values = c(Contributed = "#9ecae1", Interest = "#3182bd")) +
    ggplot2::labs(x = "Month", y = "Balance", fill = NULL) +
    ggplot2::theme_minimal(base_size = 13)
}
