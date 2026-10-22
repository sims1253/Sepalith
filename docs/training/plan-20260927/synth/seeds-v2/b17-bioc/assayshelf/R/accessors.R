#' @importFrom methods setGeneric setMethod validObject
NULL

#' Access assay matrices
#'
#' Generic for extracting a single assay from a container.
#'
#' @param x An `AssayShelf` object.
#' @param i Assay name or index; defaults to the first assay.
#' @param ... Passed to methods.
#' @return A numeric matrix.
#' @export
setGeneric("assay", function(x, i, ...) standardGeneric("assay"))

#' Replace an assay matrix
#'
#' Generic for storing a single assay in a container.
#'
#' @param x An `AssayShelf` object.
#' @param i Assay name or index; defaults to the first assay.
#' @param ... Passed to methods.
#' @param value A numeric matrix with the same dimensions as `x`.
#' @return The modified object.
#' @export
setGeneric("assay<-", function(x, i, ..., value) standardGeneric("assay<-"))

#' @rdname assay
#' @param withDimnames Should the returned matrix carry feature and sample
#'   names?
#' @export
setMethod("assay", "AssayShelf", function(x, i, withDimnames = TRUE, ...) {
  if (length(x@assays) == 0L) {
    stop("object contains no assays", call. = FALSE)
  }
  if (missing(i)) {
    i <- 1L
  }
  if (is.character(i) && !i %in% names(x@assays)) {
    stop(sprintf("no assay named '%s'", i), call. = FALSE)
  }

  out <- x@assays[[i]]
  # Dimnames live in the annotation tables, not in the stored matrices.
  if (withDimnames) {
    dimnames(out) <- list(rownames(x@rowInfo), rownames(x@colInfo))
  }
  out
})

#' @rdname assay-set
#' @export
setMethod("assay<-", "AssayShelf", function(x, i, ..., value) {
  if (missing(i)) {
    i <- if (length(x@assays) > 0L) names(x@assays)[1L] else "counts"
  }
  # A numeric index can only replace; new assays need a name.
  if (is.numeric(i) && i > length(x@assays)) {
    stop("numeric index is out of range; use a name to add an assay", call. = FALSE)
  }

  value <- as.matrix(value)
  dimnames(value) <- NULL
  x@assays[[i]] <- value
  validObject(x)
  x
})

#' Get annotations
#'
#' Return the feature or sample annotation table, optionally restricted to
#' a subset of columns.
#'
#' @param x An `AssayShelf` object.
#' @param columns Optional character vector of columns to keep.
#' @return A data frame with one row per feature (`rowInfo()`) or per sample
#'   (`colInfo()`).
#' @export
rowInfo <- function(x, columns = NULL) {
  info <- x@rowInfo
  if (is.null(columns)) {
    return(info)
  }

  missing_cols <- setdiff(columns, names(info))
  if (length(missing_cols) > 0L) {
    stop("unknown rowInfo columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }
  info[, columns, drop = FALSE]
}

#' @rdname rowInfo
#' @export
colInfo <- function(x, columns = NULL) {
  info <- x@colInfo
  if (is.null(columns)) {
    return(info)
  }

  missing_cols <- setdiff(columns, names(info))
  if (length(missing_cols) > 0L) {
    stop("unknown colInfo columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }
  info[, columns, drop = FALSE]
}

#' Replace annotations
#'
#' Replace the feature or sample annotation table. Existing row names are
#' kept so the table stays aligned with the assays.
#'
#' @param x An `AssayShelf` object.
#' @param value A data frame with one row per feature or sample.
#' @return The modified object.
#' @rdname rowInfo-set
#' @export
`rowInfo<-`<- function(x, value) {
  value <- as.data.frame(value, stringsAsFactors = FALSE)
  if (nrow(value) != nrow(x@rowInfo)) {
    stop("replacement rowInfo has the wrong number of rows", call. = FALSE)
  }

  rownames(value) <- rownames(x@rowInfo)
  x@rowInfo <- value
  validObject(x)
  x
}

#' @rdname rowInfo-set
#' @export
`colInfo<-` <- function(x, value) {
  value <- as.data.frame(value, stringsAsFactors = FALSE)
  if (nrow(value) != nrow(x@colInfo)) {
    stop("replacement colInfo has the wrong number of rows", call. = FALSE)
  }

  rownames(value) <- rownames(x@colInfo)
  x@colInfo <- value
  validObject(x)
  x
}

#' Subset an AssayShelf
#'
#' Subsets features and samples consistently across all assays and both
#' annotation tables.
#'
#' @param x An `AssayShelf` object.
#' @param i Feature index: logical, numeric or character.
#' @param j Sample index: logical, numeric or character.
#' @param ... Ignored.
#' @param drop Ignored; the result is always an `AssayShelf`.
#' @return An `AssayShelf` object.
#' @export
setMethod("[", "AssayShelf", function(x, i, j, ..., drop = FALSE) {
  if (missing(i)) {
    i <- seq_len(nrow(x@rowInfo))
  }
  if (missing(j)) {
    j <- seq_len(nrow(x@colInfo))
  }

  # Translate names to positions up front so every slot is indexed the
  # same way, whatever dimnames the individual assays carry.
  if (is.character(i)) {
    i <- match(i, rownames(x@rowInfo))
    if (anyNA(i)) stop("unknown feature names in `i`", call. = FALSE)
  }
  if (is.character(j)) {
    j <- match(j, rownames(x@colInfo))
    if (anyNA(j)) stop("unknown sample names in `j`", call. = FALSE)
  }

  x@assays <- lapply(x@assays, function(a) a[i, j, drop = FALSE])
  x@rowInfo <- x@rowInfo[i, , drop = FALSE]
  x@colInfo <- x@colInfo[j, , drop = FALSE]
  validObject(x)
  x
})
