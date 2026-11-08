#' Build a matching key for deduplication
#'
#' @param data A data frame to clean.
#' @param columns Columns that identify a record.
#' @param normalize Whether to lower-case and squish character values first.
#' @return A character vector with one key per row of `data`; rows with a
#'   missing key column get `NA`.
#' @keywords internal
dedupe.key <- function(data, columns, normalize = TRUE) {
  parts <- lapply(columns, function(col) {
    value <- as.character(data[[col]])
    if (normalize) {
      value <- tolower(gsub("[[:space:]]+", " ", trimws(value)))
    }
    value
  })
  key <- do.call(paste, c(parts, sep = "\r"))
  incomplete <- Reduce(`|`, lapply(parts, is.na))
  key[incomplete] <- NA
  key
}

#' Mark duplicate records
#'
#' @param data A data frame to clean.
#' @param columns Columns that identify a record.
#' @return `data` with two extra columns: `dup.group` (an integer id shared
#'   by matching rows, `NA` for incomplete keys) and `dup.size` (the number
#'   of rows in the group).
#' @export
mark.duplicates <- function(data, columns) {
  unknown <- setdiff(columns, names(data))
  if (length(unknown) > 0L) {
    stop("`data` lacks column(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  key <- dedupe.key(data, columns)
  group <- match(key, unique(key[!is.na(key)]))
  size <- as.integer(table(group)[as.character(group)])
  size[is.na(group)] <- 1L
  data$dup.group <- group
  data$dup.size <- size
  data
}

#' Remove duplicate records
#'
#' @param data A data frame to clean.
#' @param columns Columns that identify a record.
#' @param keep Which row of each duplicate group to keep: `"first"`,
#'   `"last"`, `"most.complete"`, `"newest"` or `"oldest"`.
#' @param time.col Name of a date or time column; required for `"newest"`
#'   and `"oldest"`.
#' @details Rows whose key columns contain `NA` are never treated as
#'   duplicates and are always kept.
#' @return `data` without the dropped rows, in the original row order.
#' @export
dedupe <- function(data, columns, keep = c("first", "last", "most.complete", "newest", "oldest"),
                   time.col = NULL) {
  keep <- match.arg(keep)
  unknown <- setdiff(c(columns, time.col), names(data))
  if (length(unknown) > 0L) {
    stop("`data` lacks column(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  if (keep %in% c("newest", "oldest") && is.null(time.col)) {
    stop("`time.col` is required when `keep` is \"", keep, "\".", call. = FALSE)
  }
  key <- dedupe.key(data, columns)
  rows <- seq_len(nrow(data))
  score <- switch(keep,
    first = {
      -rows
    },
    last = {
      rows
    },
    most.complete = {
      rowSums(!is.na(data)) - rows / (nrow(data) + 1)
    },
    newest = {
      as.numeric(data[[time.col]])
    },
    oldest = {
      -as.numeric(data[[time.col]])
    }
  )
  ord <- order(key, -score, rows)
  winner <- ord[!duplicated(key[ord])]
  kept <- sort(c(winner[!is.na(key[winner])], which(is.na(key))))
  data[kept, , drop = FALSE]
}

#' Collapse a group of duplicate values
#'
#' @param x Values of one column within one duplicate group.
#' @param how How to combine them: `"first"`, `"last"`, `"min"`, `"max"`
#'   or `"paste"`.
#' @return A single value.
#' @keywords internal
collapse.values <- function(x, how = c("first", "last", "min", "max", "paste")) {
  how <- match.arg(how)
  x <- x[!is.na(x)]
  if (length(x) == 0L) {
    return(NA)
  }
  switch(how,
    first = x[1],
    last = x[length(x)],
    min = min(x),
    max = max(x),
    paste = paste(unique(x), collapse = "; ")
  )
}

#' Merge duplicate records into one
#'
#' @param data A data frame to clean.
#' @param columns Columns that identify a record.
#' @param rules Named character vector giving the [collapse.values()] rule
#'   for non-key columns; unnamed columns use `"first"`.
#' @return A data frame with one row per duplicate group and one row per
#'   record with an incomplete key.
#' @export
coalesce.duplicates <- function(data, columns, rules = character()) {
  marked <- mark.duplicates(data, columns)
  group <- marked$dup.group
  group[is.na(group)] <- max(c(0L, group), na.rm = TRUE) + seq_len(sum(is.na(group)))
  others <- setdiff(names(data), columns)
  pieces <- lapply(split(seq_len(nrow(data)), factor(group, levels = unique(group))), function(i) {
    row <- data[i[1], , drop = FALSE]
    for (col in others) {
      how <- if (col %in% names(rules)) rules[[col]] else "first"
      row[[col]] <- collapse.values(data[[col]][i], how)
    }
    row
  })
  out <- do.call(rbind, pieces)
  rownames(out) <- NULL
  out
}

#' Summarise duplication
#'
#' @param data A data frame to clean.
#' @param columns Columns that identify a record.
#' @return A list with elements `n`, `n.unique`, `n.duplicated`,
#'   `n.incomplete` and `largest.group`.
#' @export
dedupe.summary <- function(data, columns) {
  marked <- mark.duplicates(data, columns)
  complete <- !is.na(marked$dup.group)
  list(
    n = nrow(data),
    n.unique = length(unique(marked$dup.group[complete])),
    n.duplicated = sum(duplicated(marked$dup.group[complete])),
    n.incomplete = sum(!complete),
    largest.group = if (any(complete)) max(marked$dup.size[complete]) else 0L
  )
}

#' Compare two person records field by field
#'
#' @description
#' Scores how well two records agree on `name`, `email`, `phone`,
#' `postcode` and `birth`, after normalising each field the way it is usually entered.
#'
#' @param a,b Named lists or one-row data frames holding a person record.
#' @param weights Named numeric vector of field weights.
#' @return A list with elements `fields` (a named logical vector, `NA`
#'   where either value is missing) and `score` (the weighted share of
#'   agreeing fields among the comparable ones).
#' @export
compare.records <- function(a, b,
                            weights = c(name = 1, email = 2, phone = 1, postcode = 1,
                                        birth = 1)) {
  agree <- c(name = NA, email = NA, phone = NA, postcode = NA, birth = NA)

  # Compare `name`: case and whitespace do not matter
  x <- tolower(gsub("[[:space:]]+", " ", trimws(a$name)))
  y <- tolower(gsub("[[:space:]]+", " ", trimws(b$name)))
  agree[["name"]] <- x == y

  # Compare `email`: case does not matter
  x <- tolower(trimws(a$email))
  y <- tolower(trimws(b$email))
  agree[["email"]] <- x == y

  # Compare `phone`: only digits matter
  x <- gsub("[^0-9]", "", a$phone)
  y <- gsub("[^0-9]", "", b$phone)
  agree[["phone"]] <- x == y

  # Compare `postcode`: case and spaces do not matter
  x <- toupper(gsub("[[:space:]]", "", a$postcode))
  y <- toupper(gsub("[[:space:]]", "", b$postcode))
  agree[["postcode"]] <- x == y

  # Compare `birth`: the parsed date matters
  x <- as.Date(as.character(a$birth))
  y <- as.Date(as.character(b$birth))
  agree[["birth"]] <- x == y

  comparable <- !is.na(agree)
  score <- if (any(comparable)) {
    sum(weights[names(agree)][comparable & agree]) / sum(weights[names(agree)][comparable])
  } else {
    NA_real_
  }
  list(fields = agree, score = score)
}
