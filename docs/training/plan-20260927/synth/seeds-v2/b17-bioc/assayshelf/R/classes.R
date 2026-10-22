#' @importFrom methods setClass setValidity new is validObject
NULL

#' Feature-by-sample assay container
#'
#' Holds one or more numeric matrices of identical shape together with
#' feature (row) and sample (column) annotations.
#'
#' @slot assays A named list of numeric matrices with identical dimensions.
#' @slot rowInfo A data frame of feature annotations, one row per feature.
#' @slot colInfo A data frame of sample annotations, one row per sample.
#' @slot meta A list of free-form metadata.
#' @export
setClass(
  "AssayShelf",
  representation(
    assays = "list",
    rowInfo = "data.frame",
    colInfo = "data.frame",
    meta = "list"
  ),
  prototype(
    assays = list(),
    rowInfo = data.frame(),
    colInfo = data.frame(),
    meta = list()
  )
)

#' Check that all assays share one shape
#'
#' Compares the class and dimensions of every assay against the first one.
#'
#' @param object An `AssayShelf` object.
#' @return `NULL` when valid, otherwise a character vector of problems.
#' @noRd
.check_assay_dims <- function(object) {
  assays <- object@assays
  if (length(assays) == 0L) {
    return(NULL)
  }
  if (is.null(names(assays)) || any(names(assays) == "")) {
    return("all assays must be named")
  }

  # The first assay defines the reference shape; collect every mismatch
  # rather than stopping at the first so the user sees them all at once.
  ref <- dim(assays[[1L]])
  problems <- character()
  for (nm in names(assays)) {
    a <- assays[[nm]]
    if (!is.matrix(a) || !is.numeric(a)) {
      problems <- c(problems, sprintf("assay '%s' is not a numeric matrix", nm))
    } else if (!identical(dim(a), ref)) {
      problems <- c(
        problems,
        sprintf(
          "assay '%s' has dim %s, expected %s",
          nm,
          paste(dim(a), collapse = "x"),
          paste(ref, collapse = "x")
        )
      )
    }
  }

  if (length(problems) == 0L) NULL else problems
}

#' Check feature annotations against the assays
#'
#' Ensures `rowInfo` has one row per feature and matching row names.
#'
#' @param object An `AssayShelf` object.
#' @noRd
.check_row_info <- function(object) {
  if (length(object@assays) == 0L) {
    return(NULL)
  }
  first <- object@assays[[1L]]
  info <- object@rowInfo
  if (nrow(info) != nrow(first)) {
    return(sprintf("rowInfo has %d rows, expected %d", nrow(info), nrow(first)))
  }

  # Assays without dimnames are allowed; only compare when names exist.
  ids <- rownames(first)
  if (!is.null(ids) && !identical(rownames(info), ids)) {
    return("rownames of rowInfo do not match assay rownames")
  }
  NULL
}

#' Check sample annotations against the assays
#'
#' Ensures `colInfo` has one row per sample and matching row names.
#'
#' @param object An `AssayShelf` object.
#' @noRd
.check_col_info <- function(object) {
  if (length(object@assays) == 0L) {
    return(NULL)
  }
  first <- object@assays[[1L]]
  info <- object@colInfo
  if (nrow(info) != ncol(first)) {
    return(sprintf("colInfo has %d rows, expected %d", nrow(info), ncol(first)))
  }

  ids <- colnames(first)
  if (!is.null(ids) && !identical(rownames(info), ids)) {
    return("rownames of colInfo do not match assay colnames")
  }
  NULL
}

setValidity("AssayShelf", function(object) {
  problems <- c(
    .check_assay_dims(object),
    .check_row_info(object),
    .check_col_info(object)
  )
  if (length(problems) == 0L) TRUE else problems
})

#' Build an empty annotation frame
#'
#' Creates a zero-column data frame with the requested number of rows and
#' optional row names, used when no annotation is supplied.
#'
#' @param n Number of rows.
#' @param ids Optional character vector of row names, length `n`.
#' @return A data frame with `n` rows and no columns.
#' @noRd
.empty_info <- function(n, ids = NULL) {
  out <- data.frame(row.names = seq_len(n))
  if (!is.null(ids)) {
    rownames(out) <- ids
  }
  out
}

#' Coerce an annotation argument to a data frame
#'
#' Accepts `NULL`, a data frame or anything coercible with
#' [as.data.frame()], and aligns row names to the assay dimnames.
#'
#' @param info The annotation supplied by the user, or `NULL`.
#' @param n Number of rows the annotation must have.
#' @param ids Optional character vector of row names, length `n`.
#' @param what Label used in error messages, e.g. `"rowInfo"`.
#' @return A data frame with `n` rows.
#' @noRd
.as_info <- function(info, n, ids, what) {
  if (is.null(info)) {
    return(.empty_info(n, ids))
  }

  info <- as.data.frame(info, stringsAsFactors = FALSE)
  if (nrow(info) != n) {
    stop(sprintf("`%s` must have %d rows, not %d", what, n, nrow(info)), call. = FALSE)
  }
  # Assay dimnames win over whatever row names the annotation came with.
  if (!is.null(ids)) {
    rownames(info) <- ids
  }
  info
}

#' Create an AssayShelf
#'
#' Bundles count or intensity matrices with feature and sample annotations.
#' A single matrix is accepted and stored under the name `"counts"`.
#'
#' @param assays A matrix, or a named list of matrices with identical
#'   dimensions.
#' @param rowInfo A data frame of feature annotations, one row per feature.
#' @param colInfo A data frame of sample annotations, one row per sample.
#' @param meta A list of free-form metadata.
#' @return An `AssayShelf` object.
#' @export
#' @examples
#' m <- matrix(rpois(20, 10), nrow = 5,
#'   dimnames = list(paste0("g", 1:5), paste0("s", 1:4)))
#' AssayShelf(m)
AssayShelf <- function(assays, rowInfo = NULL, colInfo = NULL, meta = list()) {
  if (is.matrix(assays)) {
    assays <- list(counts = assays)
  }
  if (!is.list(assays) || length(assays) == 0L) {
    stop("`assays` must be a matrix or a non-empty list of matrices", call. = FALSE)
  }
  if (is.null(names(assays))) {
    names(assays) <- paste0("assay", seq_along(assays))
  }

  first <- assays[[1L]]
  rowInfo <- .as_info(rowInfo, nrow(first), rownames(first), "rowInfo")
  colInfo <- .as_info(colInfo, ncol(first), colnames(first), "colInfo")

  new(
    "AssayShelf",
    assays = assays,
    rowInfo = rowInfo,
    colInfo = colInfo,
    meta = as.list(meta)
  )
}
