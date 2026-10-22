#' Select numeric columns to transform
#'
#' Resolves the `cols` argument against a data frame, defaulting to every
#' numeric column and failing on names that are absent or not numeric.
#'
#' @param data A data frame of training data.
#' @param cols Optional character vector of column names.
#' @return A character vector of column names.
#' @keywords internal
resolveNumericCols = function(data, cols = NULL) {
  if (is.null(cols)) {
    cols = names(data)[vapply(data, is.numeric, logical(1))]
  }
  missingCols = setdiff(cols, names(data))
  if (length(missingCols) > 0) {
    stop("Unknown columns: ", paste(missingCols, collapse = ", "), call. = FALSE)
  }
  notNumeric = cols[!vapply(data[cols], is.numeric, logical(1))]
  if (length(notNumeric) > 0) {
    stop("Columns are not numeric: ", paste(notNumeric, collapse = ", "), call. = FALSE)
  }
  return(cols)
}

#' Learn standardisation parameters
#'
#' Computes the mean and standard deviation of each selected column so that
#' the same centring and scaling can be applied to new data later.
#'
#' @param data A data frame of training data.
#' @param cols Optional character vector of column names.
#' @return An object of class `fp_scaler` with elements `center` and `scale`.
#' @family scalers
#' @export
#' @examples
#' sc = fitStandardScaler(mtcars, c("mpg", "hp"))
#' head(applyScaler(sc, mtcars))
fitStandardScaler = function(data, cols = NULL) {
  cols = resolveNumericCols(data, cols)
  center = vapply(data[cols], mean, numeric(1), na.rm = TRUE)
  scale = vapply(data[cols], stats::sd, numeric(1), na.rm = TRUE)
  # constant columns would divide by zero; leave them unscaled
  scale[is.na(scale) | scale == 0] = 1
  return(structure(list(cols = cols, center = center, scale = scale, type = "standard"),
    class = "fp_scaler"))
}

#' Learn min-max scaling parameters
#'
#' Records the minimum and range of each selected column so that values can be
#' mapped onto the unit interval.
#'
#' @param data A data frame of training data.
#' @param cols Optional character vector of column names.
#' @return An object of class `fp_scaler` with elements `center` and `scale`.
#' @family scalers
#' @export
fitMinMaxScaler = function(data, cols = NULL) {
  cols = resolveNumericCols(data, cols)
  center = vapply(data[cols], min, numeric(1), na.rm = TRUE)
  scale = vapply(data[cols], max, numeric(1), na.rm = TRUE) - center
  scale[is.na(scale) | scale == 0] = 1
  return(structure(list(cols = cols, center = center, scale = scale, type = "minmax"),
    class = "fp_scaler"))
}

#' Learn robust scaling parameters
#'
#' Uses the median and interquartile range of each selected column, which is
#' less sensitive to outliers than mean and standard deviation.
#'
#' @param data A data frame of training data.
#' @param cols Optional character vector of column names.
#' @return An object of class `fp_scaler` with elements `center` and `scale`.
#' @family scalers
#' @export
fitRobustScaler = function(data, cols = NULL) {
  cols = resolveNumericCols(data, cols)
  center = vapply(data[cols], stats::median, numeric(1), na.rm = TRUE)
  scale = vapply(data[cols], stats::IQR, numeric(1), na.rm = TRUE)
  scale[is.na(scale) | scale == 0] = 1
  return(structure(list(cols = cols, center = center, scale = scale, type = "robust"),
    class = "fp_scaler"))
}

#' Apply stored scaling parameters
#'
#' Centres and scales the columns recorded in a fitted scaler, leaving all
#' other columns untouched.
#'
#' @param scaler An object of class `fp_scaler`.
#' @param newdata A data frame to transform.
#' @return `newdata` with the scaled columns replaced.
#' @family scalers
#' @export
applyScaler = function(scaler, newdata) {
  stopifnot(inherits(scaler, "fp_scaler"))
  absent = setdiff(scaler$cols, names(newdata))
  if (length(absent) > 0) {
    stop("`newdata` is missing columns: ", paste(absent, collapse = ", "), call. = FALSE)
  }
  for (col in scaler$cols) {
    newdata[[col]] = (newdata[[col]] - scaler$center[[col]]) / scaler$scale[[col]]
  }
  return(newdata)
}

#' Reverse stored scaling parameters
#'
#' Maps scaled values back onto the original measurement scale.
#'
#' @param scaler An object of class `fp_scaler`.
#' @param newdata A data frame to transform.
#' @return `newdata` with the scaled columns restored.
#' @family scalers
#' @export
invertScaler = function(scaler, newdata) {
  stopifnot(inherits(scaler, "fp_scaler"))
  absent = setdiff(scaler$cols, names(newdata))
  if (length(absent) > 0) {
    stop("`newdata` is missing columns: ", paste(absent, collapse = ", "), call. = FALSE)
  }
  for (col in scaler$cols) {
    newdata[[col]] = newdata[[col]] * scaler$scale[[col]] + scaler$center[[col]]
  }
  return(newdata)
}

#' Print a fitted scaler
#'
#' Shows the scaling type and the stored parameters for each column.
#'
#' @param x An object of class `fp_scaler`.
#' @param digits Number of significant digits to display.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.fp_scaler = function(x, digits = 4, ...) {
  cat("<fp_scaler:", x$type, "on", length(x$cols), "columns>\n")
  params = data.frame(
    center = signif(x$center, digits),
    scale = signif(x$scale, digits),
    row.names = x$cols
  )
  print(params)
  return(invisible(x))
}

#' Save scaler parameters to a file
#'
#' Writes the scaler to a small CSV so that it can be versioned alongside a
#' model and reloaded without R serialisation.
#'
#' @param scaler An object of class `fp_scaler`.
#' @param path File path to write.
#' @return `path`, invisibly.
#' @export
writeScaler = function(scaler, path) {
  out = data.frame(
    column = scaler$cols,
    center = unname(scaler$center),
    scale = unname(scaler$scale),
    type = scaler$type,
    stringsAsFactors = FALSE
  )
  utils::write.csv(out, path, row.names = FALSE)
  return(invisible(path))
}

#' Read scaler parameters from a file
#'
#' Rebuilds an `fp_scaler` object from a CSV written by [writeScaler()].
#'
#' @param path File path to read.
#' @return An object of class `fp_scaler`.
#' @export
readScaler = function(path) {
  raw = utils::read.csv(path, stringsAsFactors = FALSE)
  required = c("column", "center", "scale", "type")
  if (!all(required %in% names(raw))) {
    stop("File does not look like a saved scaler: ", path, call. = FALSE)
  }
  types = unique(raw$type)
  if (length(types) != 1L) {
    stop("Saved scaler mixes several types.", call. = FALSE)
  }
  return(structure(list(
    cols = raw$column,
    center = stats::setNames(raw$center, raw$column),
    scale = stats::setNames(raw$scale, raw$column),
    type = types
  ), class = "fp_scaler"))
}
