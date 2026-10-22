#' @title Most frequent value of a vector
#' @description Returns the modal value, breaking ties by first appearance.
#' @param x A vector.
#' @noRd
modeValue <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0L) {
    return(NA)
  }
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#' @title Learn imputation values
#' @description Computes one fill value per selected column using the chosen
#'   statistic. Character, factor and logical columns always use the mode.
#' @param data A data frame of training data.
#' @param cols Optional character vector of column names; defaults to all columns.
#' @param method Statistic for numeric columns: `"median"`, `"mean"` or `"mode"`.
#' @return An object of class `fp_imputer`.
#' @export
fitImputer <- function(data, cols = NULL, method = c("median", "mean", "mode")) {
  method <- match.arg(method)
  if (is.null(cols)) {
    cols <- names(data)
  }

  fills <- vector("list", length(cols))
  names(fills) <- cols
  for (col in cols) {
    x <- data[[col]]
    if (!is.numeric(x)) {
      # factors are stored as their label so the fill survives re-levelling
      fills[[col]] <- modeValue(if (is.factor(x)) as.character(x) else x)
      next
    }
    fills[[col]] <- switch(method,
      median = stats::median(x, na.rm = TRUE),
      mean = mean(x, na.rm = TRUE),
      mode = modeValue(x)
    )
  }

  structure(list(fills = fills, method = method), class = "fp_imputer")
}

#' @title Apply stored imputation values
#' @description Replaces missing entries in each recorded column with its
#'   learned fill value, preserving factor levels where possible. Columns
#'   whose fill value is itself missing are left alone.
#' @param imputer An object of class `fp_imputer`.
#' @param newdata A data frame to transform.
#' @return `newdata` with missing values filled.
#' @export
applyImputer <- function(imputer, newdata) {
  stopifnot(inherits(imputer, "fp_imputer"))

  for (col in intersect(names(imputer$fills), names(newdata))) {
    x <- newdata[[col]]
    gaps <- is.na(x)
    fill <- imputer$fills[[col]]
    # nothing to fill, or the training column was entirely missing
    if (!any(gaps) || is.na(fill)) {
      next
    }

    if (is.factor(x) && !(fill %in% levels(x))) {
      levels(x) <- c(levels(x), fill)
    }
    x[gaps] <- fill
    newdata[[col]] <- x
  }

  newdata
}

#' @title Add missingness indicator columns
#' @description Appends a 0/1 column `<col>_missing` for every selected column
#'   that has at least one missing value, so models can learn from
#'   missingness itself.
#' @param data A data frame of training data.
#' @param cols Optional character vector of column names; defaults to all columns.
#' @param suffix Suffix appended to indicator column names.
#' @return `data` with indicator columns added.
#' @export
addMissingIndicators <- function(data, cols = NULL, suffix = "_missing") {
  if (is.null(cols)) {
    cols <- names(data)
  }

  for (col in cols) {
    gaps <- is.na(data[[col]])
    if (any(gaps)) {
      data[[paste0(col, suffix)]] <- as.integer(gaps)
    }
  }
  data
}

#' @title Fill missing values by group
#' @description Imputes a numeric column with the group-wise median, falling
#'   back to the overall median for groups that are entirely missing and for
#'   rows whose group is itself missing.
#' @param data A data frame of training data.
#' @param col Name of the numeric column to fill.
#' @param by Name of the grouping column.
#' @return `data` with `col` filled.
#' @export
imputeByGroup <- function(data, col, by) {
  overall <- stats::median(data[[col]], na.rm = TRUE)
  groupMedians <- tapply(data[[col]], data[[by]], stats::median, na.rm = TRUE)
  groupMedians[is.na(groupMedians)] <- overall

  gaps <- is.na(data[[col]])
  fills <- unname(groupMedians[as.character(data[[by]][gaps])])
  # an NA group label has no entry in the lookup
  fills[is.na(fills)] <- overall
  data[[col]][gaps] <- fills
  data
}

#' @title Summarise missingness per column
#' @description Counts missing values in every column and reports the share
#'   missing, sorted from most to least incomplete.
#' @param data A data frame of training data.
#' @return A data frame with columns `column`, `nMissing` and `share`.
#' @export
missingProfile <- function(data) {
  nMissing <- vapply(data, function(x) sum(is.na(x)), integer(1))
  out <- data.frame(
    column = names(data),
    nMissing = unname(nMissing),
    share = unname(nMissing) / max(nrow(data), 1L),
    stringsAsFactors = FALSE
  )

  out <- out[order(-out$nMissing, out$column), , drop = FALSE]
  rownames(out) <- NULL
  out
}

#' @title Drop columns that are mostly missing
#' @description Removes columns whose share of missing values exceeds
#'   `threshold`.
#' @param data A data frame of training data.
#' @param threshold Maximum allowed share of missing values.
#' @param verbose If `TRUE`, report which columns were dropped.
#' @return `data` without the sparse columns.
#' @export
dropSparseColumns <- function(data, threshold = 0.5, verbose = FALSE) {
  if (threshold < 0 || threshold > 1) {
    stop("`threshold` must be between 0 and 1.", call. = FALSE)
  }

  # strictly greater, so threshold = 1 never drops anything
  profile <- missingProfile(data)
  drop <- profile$column[profile$share > threshold]
  if (verbose && length(drop) > 0) {
    message("Dropping ", length(drop), " column(s): ", paste(drop, collapse = ", "))
  }

  data[setdiff(names(data), drop)]
}

#' @title Print an imputer
#' @description Shows the imputation method and the fill value stored for
#'   each column.
#' @param x An object of class `fp_imputer`.
#' @param ... Ignored.
#' @export
print.fp_imputer <- function(x, ...) {
  cat("<fp_imputer: method =", x$method, ">\n")
  for (col in names(x$fills)) {
    cat(sprintf("  %-20s %s\n", col, format(x$fills[[col]])))
  }
  invisible(x)
}
