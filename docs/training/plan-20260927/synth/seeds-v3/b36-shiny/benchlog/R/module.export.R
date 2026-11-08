#' @title Describe an export format
#' @param format One of `"csv"`, `"tsv"`, `"json"` or `"md"`.
#' @param prefix Start of the file name.
#' @param time Time stamp appended to the file name.
#' @return A list with elements `filename`, `mime` and `label`.
#' @family exporters
#' @export
export.format <- function(format = c("csv", "tsv", "json", "md"), prefix = "samples",
                          time = Sys.time()) {
  format <- match.arg(format)
  stamp <- format(time, "%Y%m%d-%H%M")
  filename <- paste0(prefix, "-", stamp, ".", format)
  switch(format,
    csv = list(
      filename = filename,
      mime = "text/csv",
      label = "CSV"
    ),
    tsv = list(
      filename = filename,
      mime = "text/tab-separated-values",
      label = "TSV"
    ),
    json = list(
      filename = filename,
      mime = "application/json",
      label = "JSON"
    ),
    md = list(
      filename = filename,
      mime = "text/markdown",
      label = "Markdown"
    )
  )
}

#' @title Write samples as CSV
#' @param data A data frame of samples.
#' @param path File to write.
#' @param na Text written for missing values.
#' @return `path`, invisibly.
#' @family exporters
#' @export
export.csv <- function(data, path, na = "") {
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  data[] <- lapply(data, function(col) if (inherits(col, "POSIXt")) format(col) else col)
  utils::write.table(
    data, path,
    sep = ",", na = na, row.names = FALSE, quote = TRUE, qmethod = "double",
    fileEncoding = "UTF-8"
  )
  invisible(path)
}

#' @title Write samples as TSV
#' @param data A data frame of samples.
#' @param path File to write.
#' @param na Text written for missing values.
#' @return `path`, invisibly.
#' @family exporters
#' @export
export.tsv <- function(data, path, na = "") {
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  data[] <- lapply(data, function(col) if (inherits(col, "POSIXt")) format(col) else col)
  utils::write.table(
    data, path,
    sep = "\t", na = na, row.names = FALSE, quote = FALSE,
    fileEncoding = "UTF-8"
  )
  invisible(path)
}

#' @title Write samples as JSON
#' @description Writes an array with one object per row; numbers are written
#'   bare, everything else as strings, and missing values as `null`.
#' @param data A data frame of samples.
#' @param path File to write.
#' @return `path`, invisibly.
#' @family exporters
#' @export
export.json <- function(data, path) {
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  quote.json <- function(x) paste0("\"", gsub("\"", "\\\\\"", x), "\"")
  fields <- lapply(names(data), function(col) {
    value <- data[[col]]
    text <- if (is.numeric(value)) as.character(value) else quote.json(as.character(value))
    text[is.na(value)] <- "null"
    paste0(quote.json(col), ": ", text)
  })
  rows <- do.call(paste, c(fields, sep = ", "))
  objects <- if (nrow(data) > 0L) paste0("  {", rows, "}") else character(0)
  writeLines(c("[", paste(objects, collapse = ",\n"), "]"), path)
  invisible(path)
}

#' @title Write samples as a Markdown table
#' @param data A data frame of samples.
#' @param path File to write.
#' @param na Text written for missing values.
#' @return `path`, invisibly.
#' @family exporters
#' @export
export.md <- function(data, path, na = "") {
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  data[] <- lapply(data, function(col) ifelse(is.na(col), na, as.character(col)))
  header <- paste0("| ", paste(names(data), collapse = " | "), " |")
  rule <- paste0("|", paste(rep("---", ncol(data)), collapse = "|"), "|")
  body <- if (nrow(data) > 0L) {
    paste0("| ", do.call(paste, c(unname(as.list(data)), sep = " | ")), " |")
  } else {
    character(0)
  }
  writeLines(c(header, rule, body), path)
  invisible(path)
}

#' @title Write samples in several formats at once
#' @param data A data frame of samples.
#' @param dir Directory the files are written to; created if needed.
#' @param formats Formats to write, any of `"csv"`, `"tsv"`, `"json"` and
#'   `"md"`.
#' @param prefix Start of the file names.
#' @param time Time stamp appended to the file names.
#' @return A named character vector of the written paths, one per format.
#' @family exporters
#' @export
export.bundle <- function(data, dir, formats = c("csv", "tsv", "json", "md"),
                          prefix = "samples", time = Sys.time()) {
  formats <- match.arg(formats, several.ok = TRUE)
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
  paths <- character(0)

  # Write `csv`
  if ("csv" %in% formats) {
    info <- export.format("csv", prefix, time)
    paths[["csv"]] <- export.csv(data, file.path(dir, info$filename))
  }

  # Write `tsv`
  if ("tsv" %in% formats) {
    info <- export.format("tsv", prefix, time)
    paths[["tsv"]] <- export.tsv(data, file.path(dir, info$filename))
  }

  # Write `json`
  if ("json" %in% formats) {
    info <- export.format("json", prefix, time)
    paths[["json"]] <- export.json(data, file.path(dir, info$filename))
  }

  # Write `md`
  if ("md" %in% formats) {
    info <- export.format("md", prefix, time)
    paths[["md"]] <- export.md(data, file.path(dir, info$filename))
  }
  paths
}

#' @title Export panel UI
#' @param id Module id.
#' @return A `shiny::tagList()` with one download button per format.
#' @family export module
#' @export
export.ui <- function(id) {
  ns <- shiny::NS(id)
  shiny::tagList(
    shiny::checkboxInput(ns("active_only"), "Leave out discarded samples", value = TRUE),
    shiny::downloadButton(ns("csv"), export.format("csv")$label),
    shiny::downloadButton(ns("tsv"), export.format("tsv")$label),
    shiny::downloadButton(ns("json"), export.format("json")$label),
    shiny::downloadButton(ns("md"), export.format("md")$label)
  )
}

#' @title Export panel server
#' @param id Module id.
#' @param samples A reactive returning the sample table.
#' @return `NULL`, invisibly; called for its download handlers.
#' @family export module
#' @export
export.server <- function(id, samples) {
  shiny::moduleServer(id, function(input, output, session) {
    selected <- shiny::reactive({
      data <- samples()
      if (isTRUE(input$active_only)) {
        data <- data[data$status != "discarded", , drop = FALSE]
      }
      data
    })

    output$csv <- shiny::downloadHandler(
      filename = function() export.format("csv")$filename,
      content = function(file) export.csv(selected(), file),
      contentType = export.format("csv")$mime
    )
    output$tsv <- shiny::downloadHandler(
      filename = function() export.format("tsv")$filename,
      content = function(file) export.tsv(selected(), file),
      contentType = export.format("tsv")$mime
    )
    output$json <- shiny::downloadHandler(
      filename = function() export.format("json")$filename,
      content = function(file) export.json(selected(), file),
      contentType = export.format("json")$mime
    )
    output$md <- shiny::downloadHandler(
      filename = function() export.format("md")$filename,
      content = function(file) export.md(selected(), file),
      contentType = export.format("md")$mime
    )
    invisible(NULL)
  })
}
