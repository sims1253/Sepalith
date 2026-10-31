#' Field names of a spanwise class
#'
#' @param class One of `"span"` or `"gap"`.
#' @return A character vector of field names, in storage order.
#' @keywords internal
#' @noRd
span_fields <- function(class = c("span", "gap")) {
  class <- match.arg(class)
  fields <- switch(class,
    span = c("start", "end", "label"),
    gap = c("at", "width")
  )
  return(fields)
}

#' Low-level span constructor
#'
#' Builds the list without any checks. Use [span()] in user code.
#'
#' @param start A double vector of start positions.
#' @param end A double vector of end positions.
#' @param label A character vector of labels.
#' @return An object of class `spanwise_span`.
#' @keywords internal
new_span <- function(start = double(), end = double(), label = character()) {
  stopifnot(is.double(start), is.double(end), is.character(label))
  x <- list(start = start, end = end, label = label)
  return(structure(x, class = "spanwise_span"))
}

#' Validate a span object
#'
#' @details `start` and `end` must be finite doubles, `end` and `label` must
#'   have the same length as `start`, `label` must be a character vector
#'   without missing values, and no span may end before it starts.
#'
#' @param x An object of class `spanwise_span`.
#' @return `x`, invisibly, if it is valid.
#' @keywords internal
validate_span <- function(x) {
  fields <- span_fields("span")
  values <- unclass(x)
  if (!identical(names(values), fields)) {
    stop("A <spanwise_span> must have fields `start`, `end` and `label`.", call. = FALSE)
  }
  n <- length(values$start)

  # Validate `start`
  if (!is.double(values$start)) {
    stop(sprintf("`%s` must be a double vector.", "start"), call. = FALSE)
  }
  if (anyNA(values$start) || !all(is.finite(values$start))) {
    stop(sprintf("`%s` must contain finite values only.", "start"), call. = FALSE)
  }

  # Validate `end`
  if (!is.double(values$end)) {
    stop(sprintf("`%s` must be a double vector.", "end"), call. = FALSE)
  }
  if (anyNA(values$end) || !all(is.finite(values$end))) {
    stop(sprintf("`%s` must contain finite values only.", "end"), call. = FALSE)
  }
  if (length(values$end) != n) {
    stop(sprintf("`%s` must have the same length as `start`.", "end"), call. = FALSE)
  }

  # Validate `label`
  if (!is.character(values$label)) {
    stop(sprintf("`%s` must be a character vector.", "label"), call. = FALSE)
  }
  if (anyNA(values$label)) {
    stop(sprintf("`%s` must not contain missing values.", "label"), call. = FALSE)
  }
  if (length(values$label) != n) {
    stop(sprintf("`%s` must have the same length as `start`.", "label"), call. = FALSE)
  }

  # Every span must end at or after its start
  bad <- which(values$end < values$start)
  if (length(bad) > 0L) {
    stop(sprintf("Span %d ends before it starts.", bad[1L]), call. = FALSE)
  }
  return(invisible(x))
}

#' Create spans
#'
#' Length-one `start`, `end` or `label` are recycled to the common length.
#'
#' @param start A numeric vector of start positions.
#' @param end A numeric vector of end positions.
#' @param label An optional character vector of labels. Defaults to
#'   `"span1"`, `"span2"`, and so on.
#' @return A validated object of class `spanwise_span`.
#' @export
span <- function(start, end, label = NULL) {
  # Check `start`
  if (!is.numeric(start)) {
    stop(sprintf("`%s` must be numeric, not <%s>.", "start", class(start)[1L]), call. = FALSE)
  }
  if (length(start) == 0L) {
    stop(sprintf("`%s` must not be empty.", "start"), call. = FALSE)
  }
  # Check `end`
  if (!is.numeric(end)) {
    stop(sprintf("`%s` must be numeric, not <%s>.", "end", class(end)[1L]), call. = FALSE)
  }
  if (length(end) == 0L) {
    stop(sprintf("`%s` must not be empty.", "end"), call. = FALSE)
  }
  start <- as.double(start)
  end <- as.double(end)
  n <- max(length(start), length(end))

  # Recycle length-one inputs to the common length
  if (length(start) == 1L) {
    start <- rep(start, n)
  }
  if (length(end) == 1L) {
    end <- rep(end, n)
  }
  if (is.null(label)) {
    label <- paste0("span", seq_len(n))
  }
  label <- as.character(label)
  if (length(label) == 1L) {
    label <- rep(label, n)
  }

  x <- new_span(start, end, label)
  validate_span(x)
  return(x)
}

#' Low-level gap constructor
#'
#' Builds the list without any checks. Use [gap()] in user code.
#'
#' @param at A double vector of gap positions.
#' @param width A double vector of gap widths.
#' @return An object of class `spanwise_gap`.
#' @keywords internal
new_gap <- function(at = double(), width = double()) {
  stopifnot(is.double(at), is.double(width))
  x <- list(at = at, width = width)
  return(structure(x, class = "spanwise_gap"))
}

#' Validate a gap object
#'
#' @details `at` and `width` must be finite doubles, `width` must have the
#'   same length as `at`, and no gap may have a negative width.
#'
#' @param x An object of class `spanwise_gap`.
#' @return `x`, invisibly, if it is valid.
#' @keywords internal
validate_gap <- function(x) {
  fields <- span_fields("gap")
  values <- unclass(x)
  if (!identical(names(values), fields)) {
    stop("A <spanwise_gap> must have fields `at` and `width`.", call. = FALSE)
  }
  n <- length(values$at)

  # Validate `at`
  if (!is.double(values$at)) {
    stop(sprintf("`%s` must be a double vector.", "at"), call. = FALSE)
  }
  if (anyNA(values$at) || !all(is.finite(values$at))) {
    stop(sprintf("`%s` must contain finite values only.", "at"), call. = FALSE)
  }

  # Validate `width`
  if (!is.double(values$width)) {
    stop(sprintf("`%s` must be a double vector.", "width"), call. = FALSE)
  }
  if (anyNA(values$width) || !all(is.finite(values$width))) {
    stop(sprintf("`%s` must contain finite values only.", "width"), call. = FALSE)
  }
  if (length(values$width) != n) {
    stop(sprintf("`%s` must have the same length as `at`.", "width"), call. = FALSE)
  }

  # Every gap must have a non-negative width
  bad <- which(values$width < 0)
  if (length(bad) > 0L) {
    stop(sprintf("Gap %d has a negative width.", bad[1L]), call. = FALSE)
  }
  return(invisible(x))
}

#' Create gaps
#'
#' Length-one `at` or `width` are recycled to the common length.
#'
#' @param at A numeric vector of gap positions.
#' @param width A numeric vector of gap widths.
#' @return A validated object of class `spanwise_gap`.
#' @export
gap <- function(at, width) {
  # Check `at`
  if (!is.numeric(at)) {
    stop(sprintf("`%s` must be numeric, not <%s>.", "at", class(at)[1L]), call. = FALSE)
  }
  if (length(at) == 0L) {
    stop(sprintf("`%s` must not be empty.", "at"), call. = FALSE)
  }
  # Check `width`
  if (!is.numeric(width)) {
    stop(sprintf("`%s` must be numeric, not <%s>.", "width", class(width)[1L]), call. = FALSE)
  }
  if (length(width) == 0L) {
    stop(sprintf("`%s` must not be empty.", "width"), call. = FALSE)
  }
  at <- as.double(at)
  width <- as.double(width)
  n <- max(length(at), length(width))

  # Recycle length-one inputs to the common length
  if (length(at) == 1L) {
    at <- rep(at, n)
  }
  if (length(width) == 1L) {
    width <- rep(width, n)
  }

  x <- new_gap(at, width)
  validate_gap(x)
  return(x)
}
