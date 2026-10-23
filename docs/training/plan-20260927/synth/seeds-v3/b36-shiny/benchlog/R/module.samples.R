#' @title Create an empty sample table
#' @return A zero-row data frame with columns `sample_id`, `volume`,
#'   `concentration`, `temperature`, `status` and `updated`.
#' @keywords internal
#' @noRd
samples.empty <- function() {
  data.frame(
    sample_id = character(0),
    volume = numeric(0),
    concentration = numeric(0),
    temperature = numeric(0),
    status = character(0),
    updated = as.POSIXct(character(0), tz = "UTC"),
    stringsAsFactors = FALSE
  )
}

#' @title Add a sample
#' @param samples A sample table as created by `samples.empty()`.
#' @param record A list with elements `sample_id`, `volume`, `concentration`
#'   and `temperature`.
#' @param time Time stamp stored in `updated`.
#' @return `samples` with one more row, whose status is `"received"`.
#' @keywords internal
#' @noRd
samples.add <- function(samples, record, time = Sys.time()) {
  if (record$sample_id %in% samples$sample_id) {
    stop("Sample ", record$sample_id, " already exists.", call. = FALSE)
  }
  row <- data.frame(
    sample_id = record$sample_id,
    volume = record$volume,
    concentration = record$concentration,
    temperature = record$temperature,
    status = "received",
    updated = as.POSIXct(time, tz = "UTC"),
    stringsAsFactors = FALSE
  )
  rbind(samples, row)
}

#' @title Update a sample
#' @param samples A sample table as created by `samples.empty()`.
#' @param sample_id Identifier of the sample to change.
#' @param record A list with elements `volume`, `concentration` and
#'   `temperature`.
#' @param time Time stamp stored in `updated`.
#' @return `samples` with the matching row changed; its status becomes
#'   `"stored"`.
#' @keywords internal
#' @noRd
samples.update <- function(samples, sample_id, record, time = Sys.time()) {
  row <- match(sample_id, samples$sample_id)
  if (is.na(row)) {
    stop("Sample ", sample_id, " does not exist.", call. = FALSE)
  }
  samples$volume[row] <- record$volume
  samples$concentration[row] <- record$concentration
  samples$temperature[row] <- record$temperature
  samples$status[row] <- "stored"
  samples$updated[row] <- as.POSIXct(time, tz = "UTC")
  samples
}

#' @title Take volume from a sample
#' @param samples A sample table as created by `samples.empty()`.
#' @param sample_id Identifier of the sample to change.
#' @param amount Volume taken, in microlitres.
#' @param time Time stamp stored in `updated`.
#' @return `samples` with the matching row changed; its status becomes
#'   `"in_use"`, or `"depleted"` when no volume is left.
#' @keywords internal
#' @noRd
samples.consume <- function(samples, sample_id, amount, time = Sys.time()) {
  row <- match(sample_id, samples$sample_id)
  if (is.na(row)) {
    stop("Sample ", sample_id, " does not exist.", call. = FALSE)
  }
  if (amount > samples$volume[row]) {
    stop("Sample ", sample_id, " has only ", samples$volume[row], " uL left.", call. = FALSE)
  }
  samples$volume[row] <- samples$volume[row] - amount
  samples$status[row] <- if (samples$volume[row] == 0) "depleted" else "in_use"
  samples$updated[row] <- as.POSIXct(time, tz = "UTC")
  samples
}

#' @title Discard a sample
#' @param samples A sample table as created by `samples.empty()`.
#' @param sample_id Identifier of the sample to change.
#' @param time Time stamp stored in `updated`.
#' @return `samples` with the matching row changed; its status becomes
#'   `"discarded"` and its volume zero.
#' @keywords internal
#' @noRd
samples.discard <- function(samples, sample_id, time = Sys.time()) {
  row <- match(sample_id, samples$sample_id)
  if (is.na(row)) {
    stop("Sample ", sample_id, " does not exist.", call. = FALSE)
  }
  if (samples$status[row] == "discarded") {
    stop("Sample ", sample_id, " is already discarded.", call. = FALSE)
  }
  samples$volume[row] <- 0
  samples$status[row] <- "discarded"
  samples$updated[row] <- as.POSIXct(time, tz = "UTC")
  samples
}

#' @title Summarise the sample table
#' @param samples A sample table as created by `samples.empty()`.
#' @return A list with elements `n`, `received`, `stored`, `in_use`,
#'   `depleted`, `discarded` and `volume` (total microlitres left in
#'   samples that are not discarded).
#' @keywords internal
#' @noRd
samples.stats <- function(samples) {
  status <- samples$status

  # Count per status
  received <- sum(status == "received")
  stored <- sum(status == "stored")
  in_use <- sum(status == "in_use")
  depleted <- sum(status == "depleted")
  discarded <- sum(status == "discarded")

  # Volume left
  volume <- sum(samples$volume[status != "discarded"], na.rm = TRUE)
  list(
    n = nrow(samples),
    received = received,
    stored = stored,
    in_use = in_use,
    depleted = depleted,
    discarded = discarded,
    volume = volume
  )
}

#' @title Sample entry form UI
#' @param id Module id.
#' @return A `shiny::tagList()` with one input per field, one button per
#'   action and the sample table.
#' @export
samples.ui <- function(id) {
  ns <- shiny::NS(id)
  shiny::tagList(
    shiny::textInput(ns("sample_id"), "Sample ID", placeholder = "BL-2024-00042"),
    shiny::numericInput(ns("volume"), "Volume (uL)", value = NA, min = 0),
    shiny::numericInput(ns("concentration"), "Concentration (ng/uL)", value = NA, min = 0),
    shiny::selectInput(ns("temperature"), "Storage (C)", choices = c(-196, -80, -20, 4)),
    shiny::numericInput(ns("amount"), "Amount taken (uL)", value = NA, min = 0),
    shiny::actionButton(ns("add"), "Add"),
    shiny::actionButton(ns("update"), "Update"),
    shiny::actionButton(ns("consume"), "Take volume"),
    shiny::actionButton(ns("discard"), "Discard"),
    shiny::textOutput(ns("form_status")),
    shiny::tableOutput(ns("table"))
  )
}

#' @title Sample entry form server
#' @param id Module id.
#' @param store A `shiny::reactiveValues()` object with element `samples`.
#' @return `NULL`, invisibly; called for its side effects on `store`.
#' @export
samples.server <- function(id, store) {
  shiny::moduleServer(id, function(input, output, session) {
    # Validated form values, one need() per field
    record <- shiny::reactive({
      temperature <- as.numeric(input$temperature)
      shiny::validate(
        shiny::need(is.null(check.sample.id(input$sample_id)),
                    check.sample.id(input$sample_id)),
        shiny::need(is.null(check.volume(input$volume)),
                    check.volume(input$volume)),
        shiny::need(is.null(check.concentration(input$concentration)),
                    check.concentration(input$concentration)),
        shiny::need(is.null(check.temperature(temperature)),
                    check.temperature(temperature))
      )
      list(sample_id = trimws(input$sample_id), volume = input$volume,
           concentration = input$concentration, temperature = temperature)
    })

    output$form_status <- shiny::renderText({
      record()
      "Form is complete."
    })

    # Add a sample
    shiny::observeEvent(input$add, {
      tryCatch(
        store$samples <- samples.add(store$samples, record()),
        error = function(e) shiny::showNotification(conditionMessage(e), type = "error")
      )
    })

    # Update a sample
    shiny::observeEvent(input$update, {
      tryCatch(
        store$samples <- samples.update(store$samples, record()$sample_id, record()),
        error = function(e) shiny::showNotification(conditionMessage(e), type = "error")
      )
    })

    # Take volume from a sample
    shiny::observeEvent(input$consume, {
      shiny::req(input$amount)
      tryCatch(
        store$samples <- samples.consume(store$samples, record()$sample_id, input$amount),
        error = function(e) shiny::showNotification(conditionMessage(e), type = "error")
      )
    })

    # Discard a sample
    shiny::observeEvent(input$discard, {
      tryCatch(
        store$samples <- samples.discard(store$samples, record()$sample_id),
        error = function(e) shiny::showNotification(conditionMessage(e), type = "error")
      )
    })

    output$table <- shiny::renderTable(store$samples)
    invisible(NULL)
  })
}
