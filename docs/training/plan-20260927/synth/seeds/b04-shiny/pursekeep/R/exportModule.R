#' Export module UI
#'
#' A format picker and a download button for the current budget data.
#'
#' @param id The module namespace id.
#' @return A [shiny::tagList()] of controls.
#' @family transfer modules
#' @export
exportUI = function(id) {
  ns = shiny::NS(id)
  shiny::tagList(
    shiny::radioButtons(ns("format"), "Format", choices = c("csv", "tsv", "rds"), inline = TRUE),
    shiny::downloadButton(ns("download"), "Download")
  )
}

#' Export module server
#'
#' Serves the data returned by `data` as a file in the chosen format.
#'
#' @param id The module namespace id.
#' @param data A reactive returning the data frame to export.
#' @param prefix A short name used at the start of the file name.
#' @return Called for its side effects; returns `NULL` invisibly.
#' @family transfer modules
#' @export
exportServer = function(id, data, prefix = "budget") {
  stopifnot(shiny::is.reactive(data))
  shiny::moduleServer(id, function(input, output, session) {
    output$download = shiny::downloadHandler(
      filename = function() exportFilename(prefix, input$format),
      content = function(file) {
        shiny::req(nrow(data()) > 0)
        writeBudgetFile(data(), file, format = input$format)
      }
    )
    invisible(NULL)
  })
}

#' Import module UI
#'
#' A file picker accepting the same formats the export module writes.
#'
#' @param id The module namespace id.
#' @return A [shiny::tagList()] of controls.
#' @family transfer modules
#' @export
importUI = function(id) {
  ns = shiny::NS(id)
  shiny::tagList(
    shiny::fileInput(ns("upload"), "Upload", accept = c(".csv", ".tsv", ".rds")),
    shiny::textOutput(ns("status"))
  )
}

#' Import module server
#'
#' Reads an uploaded file, checks that it has the expected columns and
#' returns the parsed data.
#'
#' @param id The module namespace id.
#' @return A reactive returning the imported data frame.
#' @family transfer modules
#' @export
importServer = function(id) {
  shiny::moduleServer(id, function(input, output, session) {
    imported = shiny::reactive({
      shiny::req(input$upload)
      format = tolower(tools::file_ext(input$upload$name))
      shiny::validate(
        shiny::need(format %in% c("csv", "tsv", "rds"), "Please upload a csv, tsv or rds file.")
      )
      data = readBudgetFile(input$upload$datapath, format = format)
      missing = setdiff(c("date", "category", "amount"), names(data))
      shiny::validate(
        shiny::need(length(missing) == 0, paste("Missing columns:", toString(missing)))
      )
      data
    })
    output$status = shiny::renderText(sprintf("Imported %d transactions.", nrow(imported())))
    imported
  })
}

#' Write budget data to a file
#'
#' Writes `data` in one of the supported formats without row names.
#'
#' @param data A data frame of transactions.
#' @param file Path of the file to write.
#' @param format One of `"csv"`, `"tsv"` or `"rds"`.
#' @return `file`, invisibly.
#' @seealso [readBudgetFile()]
#' @keywords internal
writeBudgetFile = function(data, file, format = c("csv", "tsv", "rds")) {
  format = match.arg(format)
  switch(format,
    csv = utils::write.csv(data, file, row.names = FALSE),
    tsv = utils::write.table(data, file, sep = "\t", row.names = FALSE, quote = FALSE),
    rds = saveRDS(data, file)
  )
  invisible(file)
}

#' Read budget data from a file
#'
#' Reads a file written by [writeBudgetFile()] and restores the date column.
#'
#' @param file Path of the file to read.
#' @param format One of `"csv"`, `"tsv"` or `"rds"`.
#' @return A data frame of transactions with `date` parsed as a Date.
#' @seealso [writeBudgetFile()]
#' @keywords internal
readBudgetFile = function(file, format = c("csv", "tsv", "rds")) {
  format = match.arg(format)
  data = switch(format,
    csv = utils::read.csv(file, stringsAsFactors = FALSE),
    tsv = utils::read.delim(file, stringsAsFactors = FALSE),
    rds = readRDS(file)
  )
  if ("date" %in% names(data)) {
    data$date = as.Date(data$date)
  }
  data
}

#' Build an export file name
#'
#' Combines a prefix, today's date and the format extension.
#'
#' @param prefix A short name used at the start of the file name.
#' @param format One of `"csv"`, `"tsv"` or `"rds"`.
#' @return A single file name.
#' @keywords internal
exportFilename = function(prefix, format) {
  stamp = format(Sys.Date(), "%Y%m%d")
  prefix |>
    tolower() |>
    gsub(pattern = "[^a-z0-9]+", replacement = "-") |>
    paste0("-", stamp, ".", format)
}
