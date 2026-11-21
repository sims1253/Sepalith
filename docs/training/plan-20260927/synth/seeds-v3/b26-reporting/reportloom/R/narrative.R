#' Direction of a change
#'
#' @export
#' @param before,after Numeric vectors of the earlier and later values.
#' @param tolerance Absolute differences up to this size count as flat.
#' @return A character vector of `"up"`, `"down"` or `"flat"`.
change.direction <- function(before, after, tolerance = 0) {
  delta <- after - before
  out <- rep("flat", length(delta))
  out[!is.na(delta) & delta > tolerance] <- "up"
  out[!is.na(delta) & delta < -tolerance] <- "down"
  out[is.na(delta)] <- NA_character_
  out
}

#' Sentence describing a change between two values
#'
#' @export
#' @param label A single string naming the measure.
#' @param before,after Single numbers: the earlier and later value.
#' @param digits Number of decimal places to keep.
#' @param unit Suffix appended to every number, such as `"%"`.
#' @return A single string.
change.sentence <- function(label, before, after, digits = 1L, unit = "") {
  if (is.na(before) || is.na(after)) {
    return(sprintf("%s cannot be compared.", label))
  }

  from <- paste0(formatC(before, format = "f", digits = digits), unit)
  to <- paste0(formatC(after, format = "f", digits = digits), unit)
  direction <- change.direction(before, after)
  if (direction == "up") {
    size <- paste0(formatC(after - before, format = "f", digits = digits), unit)
    sprintf("%s rose by %s, from %s to %s.", label, size, from, to)
  } else if (direction == "down") {
    size <- paste0(formatC(before - after, format = "f", digits = digits), unit)
    sprintf("%s fell by %s, from %s to %s.", label, size, from, to)
  } else {
    sprintf("%s was unchanged at %s.", label, to)
  }
}

#' Compare two periods measure by measure
#'
#' @export
#' @param current,previous Numeric vectors with one value per measure.
#' @param labels A character vector of measure names.
#' @param digits Number of decimal places to keep.
#' @param unit Suffix appended to every number, such as `"%"`.
#' @return A data frame with columns `label`, `current`, `previous`,
#'   `change`, `direction` and `sentence`.
period.comparison <- function(current, previous, labels, digits = 1L, unit = "") {
  # Check `current`
  if (!is.numeric(current)) {
    stop("`current` must be numeric.", call. = FALSE)
  }
  # Check `previous`
  if (!is.numeric(previous) || length(previous) != length(current)) {
    stop("`previous` must be numeric and match `current` in length.", call. = FALSE)
  }
  # Check `labels`
  if (length(labels) != length(current)) {
    stop("`labels` must match `current` in length.", call. = FALSE)
  }

  change <- current - previous
  direction <- change.direction(previous, current)
  sentence <- character(length(current))
  for (i in seq_along(current)) {
    sentence[i] <- change.sentence(labels[i], previous[i], current[i], digits, unit)
  }

  data.frame(
    label = labels,
    current = current,
    previous = previous,
    change = change,
    direction = direction,
    sentence = sentence
  )
}

#' Sentence naming the highest and lowest groups
#'
#' @export
#' @param values A numeric vector with one value per group.
#' @param labels A character vector of group names, the same length as
#'   `values`.
#' @param digits Number of decimal places to keep.
#' @return A single string.
extremes.sentence <- function(values, labels, digits = 1L) {
  if (length(values) != length(labels)) {
    stop("`values` and `labels` must have the same length.", call. = FALSE)
  }

  if (all(is.na(values))) {
    return("No group has an observed value.")
  }

  # Highest group
  top <- which.max(values)
  top.value <- formatC(values[top], format = "f", digits = digits)
  top.text <- sprintf("%s was highest (%s)", labels[top], top.value)
  # Lowest group
  bottom <- which.min(values)
  bottom.value <- formatC(values[bottom], format = "f", digits = digits)
  bottom.text <- sprintf("%s was lowest (%s)", labels[bottom], bottom.value)
  paste0(top.text, " and ", bottom.text, ".")
}

#' Sentence listing the leading groups
#'
#' @export
#' @param values A numeric vector with one value per group.
#' @param labels A character vector of group names, the same length as
#'   `values`.
#' @param n Number of groups to list.
#' @param decreasing Whether to list the largest values first.
#' @return A single string.
rank.sentence <- function(values, labels, n = 3L, decreasing = TRUE) {
  if (length(values) != length(labels)) {
    stop("`values` and `labels` must have the same length.", call. = FALSE)
  }
  ord <- order(values, decreasing = decreasing, na.last = NA)
  ord <- ord[seq_len(min(n, length(ord)))]
  items <- sprintf("%s (%s)", labels[ord], format(values[ord], trim = TRUE))
  which.end <- if (decreasing) "top" else "bottom"
  if (length(items) > 1L) {
    items <- c(paste(items[-length(items)], collapse = ", "), items[length(items)])
    items <- paste(items, collapse = " and ")
  }
  sprintf("The %s %d were %s.", which.end, length(ord), items)
}

#' Sentence comparing a value with its bounds
#'
#' @export
#' @param value A single number.
#' @param label A single string naming the measure.
#' @param lower,upper Bounds of the acceptable range; use `-Inf` or `Inf`
#'   for an open side.
#' @param digits Number of decimal places to keep.
#' @return A single string.
threshold.sentence <- function(value, label, lower = -Inf, upper = Inf, digits = 1L) {
  if (is.na(value)) {
    return(sprintf("%s is missing.", label))
  }

  shown <- formatC(value, format = "f", digits = digits)
  if (value < lower) {
    bound <- formatC(lower, format = "f", digits = digits)
    return(sprintf("%s (%s) is below the lower bound of %s.", label, shown, bound))
  }

  if (value > upper) {
    bound <- formatC(upper, format = "f", digits = digits)
    return(sprintf("%s (%s) is above the upper bound of %s.", label, shown, bound))
  }

  sprintf("%s (%s) is within range.", label, shown)
}

#' Phrase summarising a series of changes
#'
#' @export
#' @param x A numeric vector ordered in time.
#' @param period Noun for one step of the series.
#' @return A single string such as `"rose in 3 of 4 periods"`.
trend.phrase <- function(x, period = "period") {
  steps <- change.direction(x[-length(x)], x[-1L])
  steps <- steps[!is.na(steps)]
  if (length(steps) == 0L) {
    return("had too few values to show a trend")
  }

  n.up <- sum(steps == "up")
  n.down <- sum(steps == "down")
  total <- paste(length(steps), if (length(steps) == 1L) period else paste0(period, "s"))
  if (n.up > n.down) {
    sprintf("rose in %d of %s", n.up, total)
  } else if (n.down > n.up) {
    sprintf("fell in %d of %s", n.down, total)
  } else {
    sprintf("was mixed over %s", total)
  }
}

#' Numbered footnotes
#'
#' @description
#' Returns a small registry: `add()` stores a note and returns its marker,
#' `render()` returns the numbered notes, `count()` returns how many there
#' are. Adding the same text twice reuses the first marker.
#'
#' @export
#' @param style One of `"bracket"` or `"caret"` for markers `[1]` or `^1`.
#' @return A list of the functions `add`, `render` and `count`.
footnote.registry <- function(style = c("bracket", "caret")) {
  style <- match.arg(style)
  notes <- character()
  marker <- function(i) {
    if (style == "bracket") sprintf("[%d]", i) else sprintf("^%d", i)
  }

  add <- function(text) {
    i <- match(text, notes)
    if (is.na(i)) {
      notes <<- c(notes, text)
      i <- length(notes)
    }
    marker(i)
  }

  render <- function() {
    paste(marker(seq_along(notes)), notes)
  }

  count <- function() {
    length(notes)
  }

  list(add = add, render = render, count = count)
}
