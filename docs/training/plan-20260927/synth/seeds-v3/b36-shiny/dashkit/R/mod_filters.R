#' Collect the choices offered by the filter panel
#'
#' @param data A data frame of sales with columns `region`, `channel`,
#'   `segment`, `product` and `date`.
#' @return A list with elements `region`, `channel`, `segment`, `product`
#'   and `dates`, where `dates` holds the first and last date.
#' @family filter module
#' @keywords internal
filter_choices <- function(data) {
  required <- c("region", "channel", "segment", "product", "date")
  missing <- setdiff(required, names(data))
  if (length(missing) > 0L) {
    stop("`data` is missing columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }

  # One sorted set of values per filter
  region <- sort(unique(as.character(data$region)))
  channel <- sort(unique(as.character(data$channel)))
  segment <- sort(unique(as.character(data$segment)))
  product <- sort(unique(as.character(data$product)))

  dates <- range(as.Date(data$date), na.rm = TRUE)
  list(
    region = region,
    channel = channel,
    segment = segment,
    product = product,
    dates = dates
  )
}

#' Filter sales data by the panel selections
#'
#' An empty selection leaves the corresponding column unfiltered.
#'
#' @param data A data frame of sales with columns `region`, `channel`,
#'   `segment`, `product` and `date`.
#' @param region Selected regions.
#' @param channel Selected sales channels.
#' @param segment Selected customer segments.
#' @param product Selected product lines.
#' @param dates Length-two vector with the first and last day to keep.
#' @return The rows of `data` that match every selection.
#' @family filter module
#' @export
apply_filters <- function(data, region = NULL, channel = NULL, segment = NULL,
                          product = NULL, dates = NULL) {
  keep <- rep(TRUE, nrow(data))

  # Filter on `region`
  if (length(region) > 0L) {
    keep <- keep & data$region %in% region
  }

  # Filter on `channel`
  if (length(channel) > 0L) {
    keep <- keep & data$channel %in% channel
  }

  # Filter on `segment`
  if (length(segment) > 0L) {
    keep <- keep & data$segment %in% segment
  }

  # Filter on `product`
  if (length(product) > 0L) {
    keep <- keep & data$product %in% product
  }

  # Filter on `dates`
  if (length(dates) == 2L) {
    day <- as.Date(data$date)
    keep <- keep & !is.na(day) & day >= as.Date(dates[[1]]) & day <= as.Date(dates[[2]])
  }
  data[keep, , drop = FALSE]
}

#' Describe the active filters in one sentence
#'
#' @param region Selected regions.
#' @param channel Selected sales channels.
#' @param segment Selected customer segments.
#' @param product Selected product lines.
#' @param max_items Number of values listed before the rest are counted.
#' @return A single string, `"All data"` when nothing is selected.
#' @family filter module
#' @export
describe_filters <- function(region = NULL, channel = NULL, segment = NULL,
                             product = NULL, max_items = 2L) {
  shorten <- function(values) {
    if (length(values) <= max_items) {
      return(paste(values, collapse = ", "))
    }
    paste0(paste(values[seq_len(max_items)], collapse = ", "),
           " +", length(values) - max_items)
  }
  parts <- character(0)

  if (length(region) > 0L) {
    parts <- c(parts, paste0("region: ", shorten(region)))
  }
  if (length(channel) > 0L) {
    parts <- c(parts, paste0("channel: ", shorten(channel)))
  }
  if (length(segment) > 0L) {
    parts <- c(parts, paste0("segment: ", shorten(segment)))
  }
  if (length(product) > 0L) {
    parts <- c(parts, paste0("product: ", shorten(product)))
  }

  if (length(parts) == 0L) {
    return("All data")
  }
  paste(parts, collapse = "; ")
}

#' Count the rows each filter keeps on its own
#'
#' @param data A data frame of sales with columns `region`, `channel`,
#'   `segment`, `product` and `date`.
#' @param region Selected regions.
#' @param channel Selected sales channels.
#' @param segment Selected customer segments.
#' @param product Selected product lines.
#' @return A data frame with columns `filter`, `selected`, `kept` and
#'   `share`, one row per filter.
#' @family filter module
#' @export
filter_counts <- function(data, region = NULL, channel = NULL, segment = NULL,
                          product = NULL) {
  total <- nrow(data)
  rows <- list()

  # Rows kept by `region`
  kept <- if (length(region) > 0L) sum(data$region %in% region) else total
  rows[["region"]] <- data.frame(filter = "region", selected = length(region),
                                 kept = kept, share = kept / total)

  # Rows kept by `channel`
  kept <- if (length(channel) > 0L) sum(data$channel %in% channel) else total
  rows[["channel"]] <- data.frame(filter = "channel", selected = length(channel),
                                  kept = kept, share = kept / total)

  # Rows kept by `segment`
  kept <- if (length(segment) > 0L) sum(data$segment %in% segment) else total
  rows[["segment"]] <- data.frame(filter = "segment", selected = length(segment),
                                  kept = kept, share = kept / total)

  # Rows kept by `product`
  kept <- if (length(product) > 0L) sum(data$product %in% product) else total
  rows[["product"]] <- data.frame(filter = "product", selected = length(product),
                                  kept = kept, share = kept / total)
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

#' Filter panel UI
#'
#' @param id Module id.
#' @param choices A list as returned by [filter_choices()].
#' @return A `shiny::tagList()` with one input per filter and a reset button.
#' @family filter module
#' @export
filters_ui <- function(id, choices) {
  ns <- shiny::NS(id)
  labels <- c(region = "Region", channel = "Channel", segment = "Segment",
              product = "Product", dates = "Dates")
  shiny::tagList(
    shiny::selectInput(
      ns("region"), labels[["region"]],
      choices = choices$region, selected = choices$region, multiple = TRUE
    ),
    shiny::selectInput(
      ns("channel"), labels[["channel"]],
      choices = choices$channel, selected = choices$channel, multiple = TRUE
    ),
    shiny::selectInput(
      ns("segment"), labels[["segment"]],
      choices = choices$segment, selected = choices$segment, multiple = TRUE
    ),
    shiny::selectInput(
      ns("product"), labels[["product"]],
      choices = choices$product, selected = choices$product, multiple = TRUE
    ),
    shiny::dateRangeInput(
      ns("dates"), labels[["dates"]],
      start = choices$dates[[1]], end = choices$dates[[2]],
      min = choices$dates[[1]], max = choices$dates[[2]]
    ),
    shiny::actionButton(ns("reset"), "Reset filters", icon = shiny::icon("rotate-left")),
    shiny::textOutput(ns("summary"))
  )
}

#' Filter panel server
#'
#' @param id Module id.
#' @param data A reactive returning the full sales data frame.
#' @return A reactive returning the filtered data frame.
#' @family filter module
#' @export
filters_server <- function(id, data) {
  shiny::moduleServer(id, function(input, output, session) {
    choices <- shiny::reactive(filter_choices(data()))

    # Restore every filter to its full set of choices
    shiny::observeEvent(input$reset, {
      all <- choices()
      shiny::updateSelectInput(session, "region", selected = all$region)
      shiny::updateSelectInput(session, "channel", selected = all$channel)
      shiny::updateSelectInput(session, "segment", selected = all$segment)
      shiny::updateSelectInput(session, "product", selected = all$product)
      shiny::updateDateRangeInput(session, "dates",
                                  start = all$dates[[1]], end = all$dates[[2]])
    })

    filtered <- shiny::reactive({
      all <- choices()
      problems <- check_filter_inputs(input$region, input$channel, input$segment,
                                      input$product, all)
      shiny::validate(
        shiny::need(!"region" %in% names(problems), problems["region"]),
        shiny::need(!"channel" %in% names(problems), problems["channel"]),
        shiny::need(!"segment" %in% names(problems), problems["segment"]),
        shiny::need(!"product" %in% names(problems), problems["product"])
      )
      apply_filters(data(), input$region, input$channel, input$segment,
                    input$product, input$dates)
    })

    output$summary <- shiny::renderText({
      describe_filters(input$region, input$channel, input$segment, input$product)
    })
    filtered
  })
}
