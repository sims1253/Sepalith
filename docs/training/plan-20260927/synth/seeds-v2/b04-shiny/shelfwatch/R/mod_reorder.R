#' Reorder module UI
#'
#' A sidebar layout with a small form for drafting purchase orders and a
#' main panel showing the pending lines and their totals by supplier.
#'
#' @param id The module namespace id.
#' @return A [bslib::layout_sidebar()] containing the module inputs and outputs.
#' @export
mod_reorder_ui <- function(id) {
  ns <- shiny::NS(id)
  bslib::layout_sidebar(
    sidebar = bslib::sidebar(
      title = "New order line",
      shiny::selectizeInput(ns("sku"), "SKU", choices = NULL),
      shiny::numericInput(ns("qty"), "Quantity", value = NA, min = 1, step = 1),
      shiny::textInput(ns("supplier"), "Supplier"),
      shiny::actionButton(ns("add"), "Add line", class = "btn-primary"),
      shiny::actionButton(ns("clear"), "Clear all", class = "btn-outline-danger")
    ),
    shiny::textOutput(ns("feedback")),
    DT::DTOutput(ns("orders")),
    shiny::plotOutput(ns("by_supplier"), height = "250px")
  )
}

#' Reorder module server
#'
#' Validates new order lines, keeps the pending lines in a reactive value and
#' suggests a quantity whenever a SKU is chosen.
#'
#' @param id The module namespace id.
#' @param stock A reactive returning a data frame of current stock levels.
#' @return A reactive returning the data frame of pending order lines.
#' @export
mod_reorder_server <- function(id, stock) {
  stopifnot(shiny::is.reactive(stock))
  shiny::moduleServer(id, function(input, output, session) {
    orders <- shiny::reactiveVal(empty_order_table())
    feedback <- shiny::reactiveVal("")

    # server-side selectize copes with long SKU lists
    shiny::observe({
      shiny::updateSelectizeInput(session, "sku", choices = stock()$sku, server = TRUE)
    })

    # pre-fill a suggested quantity whenever a SKU is picked
    shiny::observeEvent(input$sku, {
      item <- stock()[stock()$sku == input$sku, , drop = FALSE]
      shiny::req(nrow(item) == 1)
      suggested <- reorder_quantity(item$on_hand, item$reorder_point, item$target_level)
      shiny::updateNumericInput(session, "qty", value = suggested)
    })

    shiny::observeEvent(input$add, {
      errors <- collect_errors(
        need_sku(input$sku, "SKU"),
        need_positive_int(input$qty, "Quantity"),
        need_text(input$supplier, "Supplier")
      )
      # only the first problem is shown, in field order
      if (length(errors) > 0) {
        feedback(errors[[1]])
        return()
      }

      # a well-formed SKU may still be unknown; build_order_row() would stop
      sku <- clean_sku(input$sku)
      if (!sku %in% stock()$sku) {
        feedback(sprintf("SKU %s is not in stock data.", sku))
        return()
      }

      line <- build_order_row(stock(), sku, input$qty, input$supplier)
      orders(rbind(orders(), line))
      feedback(sprintf("Added %d x %s.", as.integer(input$qty), line$sku))
    })

    shiny::observeEvent(input$clear, {
      orders(empty_order_table())
      feedback("Cleared all pending lines.")
    })

    output$feedback <- shiny::renderText(feedback())
    output$orders <- DT::renderDT({
      DT::datatable(orders(), rownames = FALSE, options = list(dom = "tp"))
    })
    output$by_supplier <- shiny::renderPlot({
      shiny::req(nrow(orders()) > 0)
      plot_order_totals(order_totals_by_supplier(orders()))
    })

    shiny::reactive(orders())
  })
}

#' Create an empty order table
#'
#' Returns a zero-row data frame with the columns used for pending order
#' lines so that later rows can be bound onto it.
#'
#' @noRd
empty_order_table <- function() {
  data.frame(
    sku = character(),
    supplier = character(),
    qty = integer(),
    unit_cost = numeric(),
    line_total = numeric(),
    stringsAsFactors = FALSE
  )
}

#' Build a single order line
#'
#' Looks up the unit cost for `sku` in the stock data and returns one row
#' matching the layout of [empty_order_table()].
#'
#' @param stock A data frame of current stock levels.
#' @param sku A single cleaned SKU.
#' @param qty The number of units to order.
#' @param supplier The supplier name as typed by the user.
#' @keywords internal
build_order_row <- function(stock, sku, qty, supplier) {
  unit_cost <- stock$unit_cost[match(sku, stock$sku)]
  if (is.na(unit_cost)) {
    stop("Unknown SKU: ", sku, call. = FALSE)
  }

  qty <- as.integer(qty)
  data.frame(
    sku = sku,
    supplier = clean_supplier(supplier),
    qty = qty,
    unit_cost = unit_cost,
    line_total = qty * unit_cost,
    stringsAsFactors = FALSE
  )
}

#' Suggest a reorder quantity
#'
#' Orders enough units to bring stock back up to the target level once it
#' has dropped to the reorder point, rounded up to whole cases.
#'
#' @param on_hand The current quantity on hand.
#' @param reorder_point The level at or below which an order is suggested.
#' @param target_level The level stock should be refilled to.
#' @param case_size The number of units per case.
#' @return An integer number of units, zero when no order is needed.
#' @keywords internal
reorder_quantity <- function(on_hand, reorder_point, target_level, case_size = 1L) {
  if (on_hand > reorder_point) {
    return(0L)
  }

  shortfall <- max(target_level - on_hand, 0)
  as.integer(ceiling(shortfall / case_size) * case_size)
}

#' Total pending orders by supplier
#'
#' Sums line totals and quantities for each supplier, largest spend first.
#'
#' @param orders A data frame of pending order lines.
#' @return A data frame with columns `supplier`, `qty` and `line_total`.
#' @keywords internal
order_totals_by_supplier <- function(orders) {
  # formula interface: sum qty and line_total within each supplier
  totals <- stats::aggregate(
    cbind(qty, line_total) ~ supplier,
    data = orders,
    FUN = sum
  )
  totals[order(-totals$line_total), , drop = FALSE]
}

#' Plot pending order totals by supplier
#'
#' Draws a horizontal bar chart of spend per supplier.
#'
#' @param totals A data frame as returned by [order_totals_by_supplier()].
#' @keywords internal
plot_order_totals <- function(totals) {
  # reversed levels put the biggest supplier at the top of the chart
  totals$supplier <- factor(totals$supplier, levels = rev(totals$supplier))
  ggplot2::ggplot(totals, ggplot2::aes(x = line_total, y = supplier)) +
    ggplot2::geom_col(fill = "#2c7fb8") +
    ggplot2::labs(x = "Order value", y = NULL) +
    ggplot2::theme_minimal(base_size = 13)
}
