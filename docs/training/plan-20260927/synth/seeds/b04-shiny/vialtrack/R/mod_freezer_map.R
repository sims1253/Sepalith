#' Freezer map module UI
#'
#' Lets the user pick a freezer and rack and shows a grid of box positions
#' coloured by occupancy, next to a table of rack fill levels.
#'
#' @param id The module namespace id.
#' @return A [bslib::layout_sidebar()] containing the module inputs and outputs.
#' @importFrom shiny NS selectInput plotOutput tableOutput
#' @importFrom bslib layout_sidebar sidebar card card_header
#' @export
mod_freezer_map_ui <- function(id) {
  ns <- NS(id)
  ui <- layout_sidebar(
    sidebar = sidebar(
      selectInput(ns("freezer"), "Freezer", choices = paste0("F", 1:4)),
      selectInput(ns("rack"), "Rack", choices = paste0("R", 1:6)),
      tableOutput(ns("rack_fill"))
    ),
    card(
      card_header("Box positions"),
      plotOutput(ns("grid"), height = "420px", click = ns("grid_click"))
    ),
    card(
      card_header("Selected position"),
      tableOutput(ns("selected"))
    )
  )
  return(ui)
}

#' Freezer map module server
#'
#' Draws the occupancy grid for the chosen rack and shows the sample stored
#' in whichever position the user clicks.
#'
#' @param id The module namespace id.
#' @param registry A reactive returning the data frame of registered samples.
#' @return A reactive returning the slot code of the clicked position, or `NULL`.
#' @importFrom shiny moduleServer reactive renderPlot renderTable req validate need isTruthy
#' @importFrom magrittr %>%
#' @export
mod_freezer_map_server <- function(id, registry) {
  moduleServer(id, function(input, output, session) {
    rack_samples <- reactive({
      slots <- parse_slot(registry()$freezer_slot)
      keep <- slots$freezer == input$freezer & slots$rack == input$rack
      registry()[keep, , drop = FALSE]
    })
    clicked_slot <- reactive({
      req(input$grid_click)
      col <- round(input$grid_click$x)
      row <- round(input$grid_click$y)
      req(col >= 1, col <= 10, row >= 1, row <= 10)
      format_slot(input$freezer, input$rack, (row - 1) * 10 + col)
    })
    output$grid <- renderPlot({
      box_occupancy(rack_samples()$freezer_slot) %>%
        plot_box_grid(highlight = if (isTruthy(input$grid_click)) clicked_slot())
    })
    output$rack_fill <- renderTable({
      validate(need(nrow(registry()) > 0, "The registry is empty."))
      rack_fill_table(registry(), input$freezer)
    }, digits = 0)
    output$selected <- renderTable({
      sample <- registry()[registry()$freezer_slot == clicked_slot(), , drop = FALSE]
      validate(need(nrow(sample) == 1, "This position is empty."))
      sample[, c("barcode", "matrix", "volume_ul", "collected_at")]
    })
    return(clicked_slot)
  })
}

#' Split slot codes into their parts
#'
#' Turns codes like `F2-R3-B07` into freezer, rack and box position columns.
#'
#' @param slot A character vector of slot codes.
#' @return A data frame with columns `freezer`, `rack` and `position`.
#' @seealso [format_slot()] for the inverse.
#' @keywords internal
parse_slot <- function(slot) {
  parts <- strsplit(slot, "-", fixed = TRUE)
  out <- data.frame(
    freezer = vapply(parts, `[`, character(1), 1),
    rack = vapply(parts, `[`, character(1), 2),
    position = as.integer(sub("^B", "", vapply(parts, `[`, character(1), 3))),
    stringsAsFactors = FALSE
  )
  return(out)
}

#' Combine slot parts into codes
#'
#' Builds codes like `F2-R3-B07` from freezer, rack and box position.
#'
#' @param freezer A character vector of freezer codes.
#' @param rack A character vector of rack codes.
#' @param position An integer vector of box positions.
#' @return A character vector of slot codes.
#' @seealso [parse_slot()] for the inverse.
#' @keywords internal
format_slot <- function(freezer, rack, position) {
  return(sprintf("%s-%s-B%02d", freezer, rack, as.integer(position)))
}

#' Compute occupancy for a 10 by 10 box
#'
#' Marks each of the 100 positions as occupied or free given the slot codes
#' of the samples stored in one rack.
#'
#' @param slot A character vector of slot codes.
#' @return A data frame with one row per position and columns `col`, `row`,
#'   `position`, `occupied` and `slot`.
#' @keywords internal
box_occupancy <- function(slot) {
  grid <- expand.grid(col = 1:10, row = 1:10)
  grid$position <- (grid$row - 1L) * 10L + grid$col
  taken <- parse_slot(slot)$position
  grid$occupied <- grid$position %in% taken
  grid$slot <- NA_character_
  if (length(slot) > 0) {
    grid$slot[match(taken, grid$position)] <- slot
  }
  return(grid)
}

#' Plot box occupancy as a grid
#'
#' Draws filled tiles for occupied positions and outlines the highlighted
#' slot, if any.
#'
#' @param grid A data frame as returned by [box_occupancy()].
#' @param highlight A single slot code to outline, or `NULL`.
#' @return A ggplot object.
#' @keywords internal
plot_box_grid <- function(grid, highlight = NULL) {
  p <- ggplot2::ggplot(grid, ggplot2::aes(x = col, y = row)) +
    ggplot2::geom_tile(ggplot2::aes(fill = occupied), colour = "white", linewidth = 0.8) +
    ggplot2::scale_fill_manual(
      values = c(`FALSE` = "grey90", `TRUE` = "#3b8bba"),
      guide = "none"
    ) +
    ggplot2::scale_y_reverse(breaks = 1:10) +
    ggplot2::scale_x_continuous(breaks = 1:10, position = "top") +
    ggplot2::coord_fixed() +
    ggplot2::theme_void(base_size = 12)
  if (!is.null(highlight)) {
    hit <- grid[which(grid$position == parse_slot(highlight)$position), , drop = FALSE]
    p <- p + ggplot2::geom_tile(data = hit, fill = NA, colour = "#d7301f", linewidth = 1.5)
  }
  return(p)
}

#' Summarise how full each rack is
#'
#' Counts stored samples per rack in one freezer and the positions still
#' free out of the 100 available in each rack.
#'
#' @param registry A data frame of registered samples.
#' @param freezer A single freezer code.
#' @return A data frame with columns `Rack`, `Samples` and `Free`.
#' @keywords internal
rack_fill_table <- function(registry, freezer) {
  slots <- parse_slot(registry$freezer_slot)
  slots <- slots[slots$freezer == freezer, , drop = FALSE]
  counts <- table(factor(slots$rack, levels = paste0("R", 1:6)))
  out <- data.frame(
    Rack = names(counts),
    Samples = as.integer(counts),
    Free = 100L - as.integer(counts),
    stringsAsFactors = FALSE
  )
  return(out)
}
