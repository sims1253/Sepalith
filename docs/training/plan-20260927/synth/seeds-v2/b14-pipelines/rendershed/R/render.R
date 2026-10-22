#' @title Detect the rendering engine for a document
#'
#' @description
#' Quarto documents (`.qmd`) are rendered with quarto, R Markdown documents
#' (`.Rmd`) with rmarkdown. Anything else is rejected.
#'
#' @param input Path to the source document.
#' @return Either `"quarto"` or `"rmarkdown"`.
#' @keywords internal
detectEngine = function(input) {
  ext = tolower(tools::file_ext(input))
  if (ext == "qmd") {
    return("quarto")
  }
  if (ext %in% c("rmd", "rmarkdown")) {
    return("rmarkdown")
  }
  stop("Cannot render '", basename(input), "': unsupported extension.", call. = FALSE)
}

#' @title Build the output path for a rendered report
#'
#' @description
#' The output file name combines the source file name, an optional run
#' label and the extension implied by `format`.
#'
#' @param input Path to the source document.
#' @param outputDir Directory that receives the rendered file.
#' @param format The output format, one of `"html"`, `"pdf"` or `"docx"`.
#' @param label An optional run label such as a date, appended to the name.
#' @keywords internal
outputPathFor = function(input, outputDir, format = c("html", "pdf", "docx"), label = NULL) {
  format = match.arg(format)
  stem = tools::file_path_sans_ext(basename(input))

  # Labels come from params and may contain spaces or colons
  if (!is.null(label) && nzchar(label)) {
    stem = paste(stem, gsub("[^A-Za-z0-9_-]", "-", label), sep = "_")
  }
  file.path(outputDir, paste0(stem, ".", format))
}

#' @title Render an R Markdown document
#'
#' @description
#' Renders in a clean environment and in a temporary intermediate directory
#' so that parallel renders of the same file do not collide.
#'
#' @param input Path to the source document.
#' @param outputDir Directory that receives the rendered file.
#' @param params A named list of document parameters.
#' @param format The output format, one of `"html"`, `"pdf"` or `"docx"`.
#' @param quiet Whether to suppress rendering progress output.
#' @return The path to the rendered file, invisibly.
#' @rdname renderers
#' @keywords internal
renderRmd = function(input, outputDir, params = list(), format = "html", quiet = TRUE) {
  outputFile = outputPathFor(input, outputDir, format, label = params$runLabel)
  formatName = switch(format,
    html = "html_document",
    pdf = "pdf_document",
    docx = "word_document"
  )

  workDir = tempfile("rendershed-rmd-")
  dir.create(workDir)
  on.exit(unlink(workDir, recursive = TRUE), add = TRUE)

  rmarkdown::render(
    input,
    output_format = formatName,
    output_file = basename(outputFile),
    output_dir = outputDir,
    intermediates_dir = workDir,
    params = params,
    envir = new.env(parent = globalenv()),
    quiet = quiet
  )
  invisible(outputFile)
}

#' @title Render a Quarto document
#'
#' @description
#' Renders with the quarto CLI and moves the result into `outputDir`, since
#' quarto writes next to the source file by default.
#'
#' @param input Path to the source document.
#' @param outputDir Directory that receives the rendered file.
#' @param params A named list of document parameters.
#' @param format The output format, one of `"html"`, `"pdf"` or `"docx"`.
#' @param quiet Whether to suppress rendering progress output.
#' @rdname renderers
#' @keywords internal
renderQuarto = function(input, outputDir, params = list(), format = "html", quiet = TRUE) {
  outputFile = outputPathFor(input, outputDir, format, label = params$runLabel)
  quarto::quarto_render(
    input,
    output_format = format,
    output_file = basename(outputFile),
    execute_params = params,
    quiet = quiet
  )

  # file.rename() fails across file systems, e.g. a project on a network
  # share and a local output directory; fall back to copy and delete
  produced = file.path(dirname(input), basename(outputFile))
  if (!file.rename(produced, outputFile)) {
    if (!file.copy(produced, outputFile, overwrite = TRUE)) {
      stop("Could not move rendered file to '", outputFile, "'.", call. = FALSE)
    }
    unlink(produced)
  }
  invisible(outputFile)
}

#' @title Render a document with the matching engine
#'
#' @description
#' Dispatches to [renderRmd()] or [renderQuarto()] based on the file
#' extension and records how long the render took.
#'
#' @param input Path to the source document.
#' @param outputDir Directory that receives the rendered file.
#' @param params A named list of document parameters.
#' @param format The output format, one of `"html"`, `"pdf"` or `"docx"`.
#' @param quiet Whether to suppress rendering progress output.
#' @return A one-row data frame describing the render.
#' @export
renderDocument = function(input, outputDir, params = list(), format = "html", quiet = TRUE) {
  if (!file.exists(input)) {
    stop("Input document not found: ", input, call. = FALSE)
  }
  dir.create(outputDir, recursive = TRUE, showWarnings = FALSE)
  engine = detectEngine(input)

  # Render errors are recorded, not raised, so a scheduler can notify and
  # carry on with the next report
  started = Sys.time()
  result = tryCatch(
    {
      path = switch(engine,
        quarto = renderQuarto(input, outputDir, params, format, quiet),
        rmarkdown = renderRmd(input, outputDir, params, format, quiet)
      )
      list(output = path, status = "ok", message = NA_character_)
    },
    error = function(e) {
      list(output = NA_character_, status = "error", message = conditionMessage(e))
    }
  )

  data.frame(
    input = input,
    engine = engine,
    output = result$output,
    status = result$status,
    message = result$message,
    seconds = round(as.numeric(difftime(Sys.time(), started, units = "secs")), 2),
    stringsAsFactors = FALSE
  )
}

#' @title Render several documents in one go
#'
#' @description
#' Renders each input in turn with [renderDocument()] and binds the
#' per-document rows together. Failures do not stop the remaining renders.
#'
#' @param inputs A character vector of source document paths.
#' @param outputDir Directory that receives the rendered file.
#' @param params A named list of document parameters, shared by all inputs.
#' @param ... Passed to [renderDocument()].
#' @return A data frame with one row per input, or `NULL` when `inputs` is
#'   empty.
#' @export
renderMany = function(inputs, outputDir, params = list(), ...) {
  if (length(inputs) == 0L) {
    return(NULL)
  }

  rows = vector("list", length(inputs))
  for (i in seq_along(inputs)) {
    rows[[i]] = renderDocument(inputs[[i]], outputDir, params = params, ...)
  }
  out = do.call(rbind, rows)

  failed = out$status != "ok"
  if (any(failed)) {
    warning(
      sum(failed), " of ", nrow(out), " document(s) failed to render: ",
      paste(basename(out$input[failed]), collapse = ", "),
      call. = FALSE
    )
  }
  out
}

#' @title Remove rendering leftovers
#'
#' @description
#' Deletes knitr cache folders, `_files` directories and stray `.log` and
#' `.tex` files that renders leave next to the sources.
#'
#' @param dir Directory to clean.
#' @param patterns Regular expressions for files and folders to delete.
#' @param dryRun If `TRUE`, only report what would be removed.
#' @return The paths that were (or would be) removed, invisibly.
#' @export
cleanIntermediates = function(dir, patterns = c("_cache$", "_files$", "\\.log$", "\\.tex$"),
                              dryRun = FALSE) {
  candidates = list.files(dir, full.names = TRUE, include.dirs = TRUE)
  hits = candidates[Reduce(`|`, lapply(patterns, grepl, x = basename(candidates)))]
  if (length(hits) == 0L) {
    return(invisible(character()))
  }

  if (dryRun) {
    message("Would remove: ", paste(basename(hits), collapse = ", "))
  } else {
    unlink(hits, recursive = TRUE)
  }
  invisible(hits)
}
