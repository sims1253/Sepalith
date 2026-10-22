#' Stock filter module UI
#'
#' Sidebar controls for narrowing the stock table by warehouse, category and
#' stock status.
#'
#' @param id The module namespace id.
#' @param warehouses A character vector of warehouse codes offered as choices.
#' @return A [shiny::tagList()] of input controls.
#' @export
mod_stock_filter_ui <- function(id, warehouses) {
  ns <- shiny::NS(id)
  shiny::tagList(
    shiny::selectInput(ns("warehouse"), "Warehouse", choices = c("All", warehouses)),
    shiny::selectizeInput(ns("category"), "Category", choices = NULL, multiple = TRUE),
    shiny::checkboxGroupInput(
      ns("status"),
      "Status",
      choices = c("ok", "low", "out"),
      selected = c("ok", "low", "out"),
      inline = TRUE
    )
  )
}

#' Stock filter module server
#'
#' Keeps the category choices in sync with the data and returns the filtered
#' stock as a reactive.
#'
#' @param id The module namespace id.
#' @param stock A reactive returning a data frame of current stock levels.
#' @return A reactive returning the filtered data frame.
#' @export
mod_stock_filter_server <- function(id, stock) {
  stopifnot(shiny::is.reactive(stock))
  shiny::moduleServer(id, function(input, output, session) {
    # offer only the categories present in the current data
    shiny::observe({
      categories <- sort(unique(stock()$category))
      shiny::updateSelectizeInput(session, "category", choices = categories)
    })

    shiny::reactive({
      data <- stock() |>
        transform(status = stock_status(on_hand, reorder_point))
      if (!identical(input$warehouse, "All")) {
        data <- data[data$warehouse %in% input$warehouse, , drop = FALSE]
      }
      # no category selected means all categories
      if (length(input$category) > 0) {
        data <- data[data$category %in% input$category, , drop = FALSE]
      }
      data[data$status %in% input$status, , drop = FALSE]
    })
  })
}

#' Stock table module UI
#'
#' A card holding summary value boxes above an interactive stock table.
#'
#' @param id The module namespace id.
#' @return A [bslib::card()] containing the module outputs.
#' @export
mod_stock_table_ui <- function(id) {
  ns <- shiny::NS(id)
  bslib::card(
    full_screen = TRUE,
    bslib::card_header("Current stock"),
    shiny::uiOutput(ns("summary")),
    DT::DTOutput(ns("table"))
  )
}

#' Stock table module server
#'
#' Renders the summary boxes and the formatted stock table for the filtered
#' data, and returns the SKU of the selected row.
#'
#' @param id The module namespace id.
#' @param stock A reactive returning a data frame of current stock levels.
#' @return A reactive returning the selected SKU, or `NULL` when nothing is selected.
#' @export
mod_stock_table_server <- function(id, stock) {
  stopifnot(shiny::is.reactive(stock))
  shiny::moduleServer(id, function(input, output, session) {
    output$summary <- shiny::renderUI({
      shiny::req(nrow(stock()) > 0)
      stock_summary_boxes(stock())
    })

    output$table <- DT::renderDT({
      shiny::validate(shiny::need(nrow(stock()) > 0, "No items match the current filters."))
      # keep the severity order from format_stock_table(); sorting the Status
      # column in the browser would be alphabetical ("low" < "ok" < "out")
      DT::datatable(
        format_stock_table(stock()),
        selection = "single",
        rownames = FALSE,
        options = list(pageLength = 15, order = list())
      )
    })

    shiny::reactive({
      row <- input$table_rows_selected
      if (is.null(row)) {
        return(NULL)
      }
      # DT reports row numbers of the data passed to datatable(), whatever
      # order the user has sorted the display into
      format_stock_table(stock())$SKU[row]
    })
  })
}

#' Classify stock levels
#'
#' Labels each item as `"out"`, `"low"` or `"ok"` by comparing the quantity
#' on hand with its reorder point.
#'
#' @param on_hand A numeric vector of quantities on hand.
#' @param reorder_point A numeric vector of reorder points, recycled to `on_hand`.
#' @noRd
stock_status <- function(on_hand, reorder_point) {
  # later assignments win, so "out" overrides "low"
  status <- rep("ok", length(on_hand))
  status[on_hand <= reorder_point] <- "low"
  status[on_hand <= 0] <- "out"
  status
}

#' Prepare stock data for display
#'
#' Selects and renames the columns shown to users and orders rows so that
#' items needing attention come first.
#'
#' @param data A data frame of current stock levels.
#' @noRd
format_stock_table <- function(data) {
  severity <- match(data$status, c("out", "low", "ok"))
  data <- data[order(severity, data$sku), , drop = FALSE]

  data.frame(
    SKU = data$sku,
    Item = data$name,
    Warehouse = data$warehouse,
    `On hand` = data$on_hand,
    Status = data$status,
    `Unit cost` = sprintf("%.2f", data$unit_cost),
    check.names = FALSE
  )
}

#' Build summary value boxes for stock data
#'
#' Shows the number of distinct items, the number below their reorder point
#' and the total value of stock held.
#'
#' @param data A data frame of current stock levels.
#' @keywords internal
stock_summary_boxes <- function(data) {
  n_items <- length(unique(data$sku))
  n_low <- sum(data$status %in% c("low", "out"))
  total_value <- sum(data$on_hand * data$unit_cost, na.rm = TRUE)

  bslib::layout_column_wrap(
    width = 1 / 3,
    bslib::value_box("Items", n_items, theme = "primary"),
    bslib::value_box("Need reorder", n_low, theme = if (n_low > 0) "warning" else "success"),
    bslib::value_box(
      "Stock value",
      format(round(total_value), big.mark = ","),
      theme = "secondary"
    )
  )
}
