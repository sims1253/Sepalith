#' Abbreviate a vector of names for display
#'
#' Shows the first and last few names separated by an ellipsis when the
#' vector is longer than `n_head + n_tail`.
#'
#' @param ids A character vector of names.
#' @param n_head Number of leading names to show.
#' @param n_tail Number of trailing names to show.
#' @return A single string.
#' @noRd
.abbrev_names <- function(ids, n_head = 2L, n_tail = 2L) {
  if (is.null(ids) || length(ids) == 0L) {
    return("(none)")
  }
  if (length(ids) <= n_head + n_tail) {
    return(paste(ids, collapse = " "))
  }

  paste(
    c(utils::head(ids, n_head), "...", utils::tail(ids, n_tail)),
    collapse = " "
  )
}

#' Format the feature line of the show output
#'
#' @param object An `AssayShelf` object.
#' @return A single string.
#' @noRd
.format_row_line <- function(object) {
  ids <- rownames(object@rowInfo)
  sprintf("rownames(%d): %s", length(ids), .abbrev_names(ids))
}

#' Format the sample line of the show output
#'
#' @param object An `AssayShelf` object.
#' @return A single string.
#' @noRd
.format_col_line <- function(object) {
  ids <- rownames(object@colInfo)
  sprintf("colnames(%d): %s", length(ids), .abbrev_names(ids))
}

#' Format an annotation-column line of the show output
#'
#' @param info A data frame of annotations.
#' @param label Prefix for the line, e.g. `"rowInfo"`.
#' @return A single string.
#' @noRd
.format_info_line <- function(info, label) {
  cols <- names(info)
  sprintf("%s names(%d): %s", label, length(cols), .abbrev_names(cols, 3L, 1L))
}

#' Display an AssayShelf
#'
#' Prints dimensions, assay names, dimnames and annotation columns.
#'
#' @param object An `AssayShelf` object.
#' @return `object`, invisibly.
#' @export
setMethod("show", "AssayShelf", function(object) {
  n_row <- nrow(object@rowInfo)
  n_col <- nrow(object@colInfo)

  cat("class: AssayShelf\n")
  cat(sprintf("dim: %d %d\n", n_row, n_col))
  cat(sprintf(
    "assays(%d): %s\n",
    length(object@assays),
    .abbrev_names(names(object@assays))
  ))

  # Feature lines first, then sample lines, mirroring the matrix layout.
  cat(.format_row_line(object), "\n", sep = "")
  cat(.format_info_line(object@rowInfo, "rowInfo"), "\n", sep = "")
  cat(.format_col_line(object), "\n", sep = "")
  cat(.format_info_line(object@colInfo, "colInfo"), "\n", sep = "")

  if (length(object@meta) > 0L) {
    cat(sprintf("meta(%d): %s\n", length(object@meta), .abbrev_names(names(object@meta))))
  }
  invisible(object)
})

#' Summarise an AssayShelf
#'
#' Collects per-assay value ranges, the share of zeros and missing values.
#'
#' @param object An `AssayShelf` object.
#' @param ... Ignored.
#' @return An object of class `summary_AssayShelf`.
#' @export
setMethod("summary", "AssayShelf", function(object, ...) {
  stats <- vector("list", length(object@assays))
  names(stats) <- names(object@assays)
  for (nm in names(object@assays)) {
    a <- object@assays[[nm]]
    # Ranges ignore NA/Inf; missing values are counted separately below.
    finite <- a[is.finite(a)]
    stats[[nm]] <- data.frame(
      assay = nm,
      min = if (length(finite) > 0L) min(finite) else NA_real_,
      median = if (length(finite) > 0L) stats::median(finite) else NA_real_,
      max = if (length(finite) > 0L) max(finite) else NA_real_,
      pct_zero = 100 * mean(a == 0, na.rm = TRUE),
      n_na = sum(is.na(a)),
      stringsAsFactors = FALSE
    )
  }

  out <- list(
    dim = c(nrow(object@rowInfo), nrow(object@colInfo)),
    assays = do.call(rbind, stats),
    qc_log = object@meta$qc_log
  )
  class(out) <- "summary_AssayShelf"
  out
})

#' Print an AssayShelf summary
#'
#' @param x An object of class `summary_AssayShelf`.
#' @param digits Number of significant digits for numeric columns.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.summary_AssayShelf <- function(x, digits = 3, ...) {
  cat(sprintf("AssayShelf with %d features and %d samples\n", x$dim[1L], x$dim[2L]))

  cat("\nAssays:\n")
  tab <- x$assays
  num_cols <- vapply(tab, is.numeric, logical(1))
  tab[num_cols] <- lapply(tab[num_cols], signif, digits = digits)
  print(tab, row.names = FALSE)

  if (!is.null(x$qc_log) && nrow(x$qc_log) > 0L) {
    cat("\nQC filters applied:\n")
    print(x$qc_log, row.names = FALSE)
  } else {
    cat("\nNo QC filters applied.\n")
  }
  invisible(x)
}
