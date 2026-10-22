#' Learn the levels of categorical columns
#'
#' Records the observed levels of each selected column so that the same dummy
#' columns can be produced for new data. Rare levels can be pooled.
#'
#' @param data A data frame of training data.
#' @param cols Character vector of categorical column names.
#' @param minCount Levels seen fewer times than this are pooled into `other`.
#' @return An object of class `fp_onehot`.
#' @family encoders
#' @export
fitOneHot <- function(data, cols, minCount = 1L) {
  levelMap <- vector("list", length(cols))
  names(levelMap) <- cols
  for (col in cols) {
    counts <- table(as.character(data[[col]]), useNA = "no")
    keep <- names(counts)[counts >= minCount]
    if (length(keep) < length(counts)) {
      keep <- c(keep, "other")
    }
    levelMap[[col]] <- keep
  }
  structure(list(levels = levelMap, minCount = minCount), class = "fp_onehot")
}

#' Build dummy column names
#'
#' Combines a column name and its levels into syntactic dummy column names.
#'
#' @param col A single column name.
#' @param levels Character vector of levels for that column.
#' @return A character vector of the same length as `levels`.
#' @keywords internal
dummyNames <- function(col, levels) {
  make.names(paste(col, levels, sep = "_"), unique = TRUE)
}

#' Map raw values onto learned levels
#'
#' Replaces unseen or pooled values with `other` when that level exists, and
#' with `NA` otherwise.
#'
#' @param x A vector of raw categorical values.
#' @param levels Character vector of levels for that column.
#' @return A character vector.
#' @keywords internal
mapToLevels <- function(x, levels) {
  x <- as.character(x)
  unseen <- !is.na(x) & !(x %in% levels)
  if (any(unseen)) {
    x[unseen] <- if ("other" %in% levels) "other" else NA_character_
  }
  x
}

#' Apply a one-hot encoder
#'
#' Replaces each encoded column by one 0/1 indicator column per learned level.
#' Missing values produce rows of all zeros unless `keepNA` is `TRUE`.
#'
#' @param encoder An object of class `fp_onehot`.
#' @param newdata A data frame to transform.
#' @param keepNA If `TRUE`, missing inputs give `NA` in every dummy column.
#' @return A data frame with dummy columns in place of the originals.
#' @family encoders
#' @export
applyOneHot <- function(encoder, newdata, keepNA = FALSE) {
  stopifnot(inherits(encoder, "fp_onehot"))
  for (col in names(encoder$levels)) {
    levs <- encoder$levels[[col]]
    values <- mapToLevels(newdata[[col]], levs)
    dummies <- vapply(levs, function(l) as.integer(values == l), integer(nrow(newdata)))
    dummies <- matrix(dummies, nrow = nrow(newdata))
    if (!keepNA) {
      dummies[is.na(dummies)] <- 0L
    }
    colnames(dummies) <- dummyNames(col, levs)
    newdata[[col]] <- NULL
    newdata <- cbind(newdata, as.data.frame(dummies))
  }
  newdata
}

#' Learn integer codes for categorical columns
#'
#' Records an ordered set of levels per column for label encoding. The order
#' is either alphabetical or by descending frequency.
#'
#' @param data A data frame of training data.
#' @param cols Character vector of categorical column names.
#' @param order How to order levels: `"alpha"` or `"frequency"`.
#' @return An object of class `fp_label`.
#' @family encoders
#' @export
fitLabelEncoder <- function(data, cols, order = c("alpha", "frequency")) {
  order <- match.arg(order)
  levelMap <- lapply(cols, function(col) {
    counts <- table(as.character(data[[col]]))
    switch(order,
      alpha = sort(names(counts)),
      frequency = names(sort(counts, decreasing = TRUE))
    )
  })
  names(levelMap) <- cols
  structure(list(levels = levelMap, order = order), class = "fp_label")
}

#' Apply a label encoder
#'
#' Replaces each encoded column by its integer level code. Unseen values are
#' coded as `unknownCode`.
#'
#' @param encoder An object of class `fp_label`.
#' @param newdata A data frame to transform.
#' @param unknownCode Integer used for values not seen during fitting.
#' @return `newdata` with integer-coded columns.
#' @family encoders
#' @export
applyLabelEncoder <- function(encoder, newdata, unknownCode = 0L) {
  stopifnot(inherits(encoder, "fp_label"))
  for (col in names(encoder$levels)) {
    codes <- match(as.character(newdata[[col]]), encoder$levels[[col]])
    codes[is.na(codes) & !is.na(newdata[[col]])] <- unknownCode
    newdata[[col]] <- as.integer(codes)
  }
  newdata
}

#' Print a one-hot encoder
#'
#' Lists each encoded column with its number of dummy columns.
#'
#' @param x An object of class `fp_onehot`.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.fp_onehot <- function(x, ...) {
  nLevels <- lengths(x$levels)
  cat("<fp_onehot:", length(nLevels), "columns ->", sum(nLevels), "dummies>\n")
  for (col in names(nLevels)) {
    cat(sprintf("  %-20s %3d levels\n", col, nLevels[[col]]))
  }
  invisible(x)
}

#' Print a label encoder
#'
#' Lists each encoded column with its number of integer codes.
#'
#' @param x An object of class `fp_label`.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.fp_label <- function(x, ...) {
  nLevels <- lengths(x$levels)
  cat("<fp_label:", length(nLevels), "columns, order =", x$order, ">\n")
  for (col in names(nLevels)) {
    cat(sprintf("  %-20s %3d codes\n", col, nLevels[[col]]))
  }
  invisible(x)
}
