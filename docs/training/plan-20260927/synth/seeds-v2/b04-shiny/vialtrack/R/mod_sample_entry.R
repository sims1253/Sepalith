#' Sample entry module UI
#'
#' A form for registering a newly received sample, with live feedback for
#' each field.
#'
#' @param id The module namespace id.
#' @return A [bslib::card()] containing the form.
#' @importFrom shiny NS textInput numericInput selectInput actionButton uiOutput
#' @importFrom bslib card card_header card_body layout_columns
#' @export
mod_sample_entry_ui <- function(id) {
  ns <- NS(id)

  form <- card(
    card_header("Register sample"),
    card_body(
      layout_columns(
        textInput(ns("barcode"), "Barcode", placeholder = "VT-000123"),
        selectInput(ns("matrix"), "Matrix", choices = sample_matrices()),
        numericInput(ns("volume_ul"), "Volume (uL)", value = NA, min = 0)
      ),
      layout_columns(
        textInput(ns("collected_at"), "Collected at", placeholder = "2026-10-21 14:30"),
        textInput(ns("freezer_slot"), "Freezer slot", placeholder = "F2-R3-B07")
      ),
      uiOutput(ns("problems")),
      actionButton(ns("submit"), "Register", class = "btn-success")
    )
  )
  return(form)
}

#' Sample entry module server
#'
#' Validates the form on submit, rejects duplicate barcodes and emits the
#' new sample record for the caller to persist.
#'
#' @param id The module namespace id.
#' @param registry A reactive returning the data frame of registered samples.
#' @return A reactive returning a one-row data frame for each accepted sample.
#' @importFrom shiny moduleServer reactiveVal observeEvent renderUI req tags
#' @export
mod_sample_entry_server <- function(id, registry) {
  moduleServer(id, function(input, output, session) {
    problems <- reactiveVal(character())
    accepted <- reactiveVal(NULL)

    observeEvent(input$submit, {
      # run every check so the user sees all problems at once
      found <- c(
        check_barcode(input$barcode, registry()$barcode),
        check_volume(input$volume_ul, input$matrix),
        check_collected_at(input$collected_at),
        check_freezer_slot(input$freezer_slot, registry()$freezer_slot)
      )
      problems(found)
      req(length(found) == 0)

      record <- make_sample_record(
        barcode = input$barcode,
        matrix = input$matrix,
        volume_ul = input$volume_ul,
        collected_at = input$collected_at,
        freezer_slot = input$freezer_slot
      )
      accepted(record)
      # clear the barcode so the next tube can be scanned straight away
      shiny::updateTextInput(session, "barcode", value = "")
    })

    output$problems <- renderUI({
      if (length(problems()) == 0) {
        return(NULL)
      }
      tags$div(class = "alert alert-danger", tags$ul(lapply(problems(), tags$li)))
    })

    return(shiny::reactive(accepted()))
  })
}

#' Check a sample barcode or freezer slot
#'
#' Barcodes must match `VT-` followed by six digits; slots are written as
#' freezer, rack and box position, for example `F2-R3-B07`. Neither may
#' already be in the registry.
#'
#' @param value The value entered in the form.
#' @param existing A character vector of values already in the registry.
#' @return A character vector of problems, empty when the value is valid.
#' @keywords internal
check_barcode <- function(value, existing) {
  # a cleared input can arrive as NULL rather than ""
  value <- trimws(if (is.null(value)) "" else value)
  if (length(value) != 1 || is.na(value) || !nzchar(value)) {
    return("Barcode is required.")
  }
  if (!grepl("^VT-[0-9]{6}$", value)) {
    return("Barcode must look like VT-000123.")
  }
  if (value %in% existing) {
    return(sprintf("Barcode %s is already registered.", value))
  }
  return(character())
}

#' @rdname check_barcode
check_freezer_slot <- function(value, existing) {
  value <- toupper(trimws(if (is.null(value)) "" else value))
  if (length(value) != 1 || is.na(value) || !nzchar(value)) {
    return("Freezer slot is required.")
  }
  if (!grepl("^F[0-9]-R[0-9]-B[0-9]{2}$", value)) {
    return("Freezer slot must look like F2-R3-B07.")
  }
  if (value %in% existing) {
    return(sprintf("Freezer slot %s is already occupied.", value))
  }
  return(character())
}

#' Check a sample volume
#'
#' Volumes must be positive and within the range accepted for the chosen
#' sample matrix.
#'
#' @param value The value entered in the form.
#' @param matrix The sample matrix chosen in the form.
#' @return A character vector of problems, empty when the value is valid.
#' @keywords internal
check_volume <- function(value, matrix) {
  if (is.null(value) || is.na(value)) {
    return("Volume is required.")
  }

  # accepted range in uL per matrix; unknown matrices get a wide default
  limits <- switch(matrix,
    plasma = c(50, 2000),
    serum = c(50, 2000),
    urine = c(500, 10000),
    saliva = c(100, 5000),
    c(1, 10000)
  )
  if (value < limits[1] || value > limits[2]) {
    return(sprintf("Volume for %s must be between %g and %g uL.", matrix, limits[1], limits[2]))
  }
  return(character())
}

#' Check a collection timestamp
#'
#' Timestamps must parse as `YYYY-MM-DD HH:MM` and must not lie in the
#' future.
#'
#' @param value The value entered in the form.
#' @keywords internal
check_collected_at <- function(value) {
  parsed <- as.POSIXct(trimws(value), format = "%Y-%m-%d %H:%M", tz = "UTC")
  if (is.na(parsed)) {
    return("Collection time must look like 2026-10-21 14:30.")
  }
  if (parsed > Sys.time()) {
    return("Collection time cannot be in the future.")
  }
  return(character())
}

#' Build a sample record
#'
#' Normalises the form values into a one-row data frame matching the
#' `samples` table. Timestamps are stored as UTC text, which is what SQLite
#' gets anyway.
#'
#' @param barcode The sample barcode.
#' @param matrix The sample matrix.
#' @param volume_ul The sample volume in microlitres.
#' @param collected_at The collection time as entered.
#' @param freezer_slot The freezer slot code.
#' @return A one-row data frame.
#' @keywords internal
make_sample_record <- function(barcode, matrix, volume_ul, collected_at, freezer_slot) {
  # parse with the same format that check_collected_at() accepted
  collected <- as.POSIXct(trimws(collected_at), format = "%Y-%m-%d %H:%M", tz = "UTC")

  record <- data.frame(
    barcode = trimws(barcode),
    matrix = matrix,
    volume_ul = as.numeric(volume_ul),
    collected_at = format(collected, "%Y-%m-%d %H:%M:%S"),
    freezer_slot = toupper(trimws(freezer_slot)),
    registered_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S", tz = "UTC"),
    discarded = 0L,
    stringsAsFactors = FALSE
  )
  return(record)
}

#' Sample matrix choices
#'
#' The sample types the lab accepts, as labelled choices for a select input.
#'
#' @noRd
sample_matrices <- function() {
  return(c(Plasma = "plasma", Serum = "serum", Urine = "urine", Saliva = "saliva"))
}
