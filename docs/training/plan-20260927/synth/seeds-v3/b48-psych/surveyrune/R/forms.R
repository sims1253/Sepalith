#' Validate a questionnaire form definition
#'
#' @description
#' A form is a data frame with one row per item and the columns `id`, `text`,
#' `scale`, `keyed`, `min` and `max`.
#'
#' @param form A data frame describing the questionnaire items.
#' @return `form`, invisibly, with `keyed` stored as integer.
#' @export
validate.form <- function(form) {
  columns <- c("id", "text", "scale", "keyed", "min", "max")
  absent <- setdiff(columns, names(form))
  if (length(absent) > 0) {
    stop(sprintf("`form` lacks columns: %s", toString(absent)), call. = FALSE)
  }

  # Column `id`
  if (!is.character(form$id) || anyNA(form$id) || anyDuplicated(form$id)) {
    stop("Column `id` must be unique non-missing strings.", call. = FALSE)
  }
  # Column `text`
  if (!is.character(form$text) || any(is.na(form$text) | form$text == "")) {
    stop("Column `text` must be non-empty strings.", call. = FALSE)
  }
  # Column `scale`
  if (!is.character(form$scale) || anyNA(form$scale)) {
    stop("Column `scale` must be non-missing strings.", call. = FALSE)
  }
  # Column `keyed`
  if (!all(form$keyed %in% c(-1, 1))) {
    stop("Column `keyed` must be -1 or 1.", call. = FALSE)
  }
  form$keyed <- as.integer(form$keyed)
  # Column `min`
  if (!is.numeric(form$min) || anyNA(form$min)) {
    stop("Column `min` must be non-missing numbers.", call. = FALSE)
  }
  # Column `max`
  if (!is.numeric(form$max) || anyNA(form$max)) {
    stop("Column `max` must be non-missing numbers.", call. = FALSE)
  }
  if (any(form$max <= form$min)) {
    stop("Column `max` must exceed column `min`.", call. = FALSE)
  }

  invisible(form)
}

#' Read a questionnaire form
#'
#' @description
#' Reads a form definition from disk in one of the supported formats.
#'
#' @param path Path of the form file.
#' @param format One of `"csv"`, `"tsv"`, `"rds"` or `"dcf"`.
#' @return A validated form data frame.
#' @export
read.form <- function(path, format = c("csv", "tsv", "rds", "dcf")) {
  format <- match.arg(format)
  if (!file.exists(path)) {
    stop(sprintf("Form file `%s` does not exist.", path), call. = FALSE)
  }

  form <- switch(format,
    csv = {
      utils::read.csv(path, stringsAsFactors = FALSE, colClasses = c(id = "character"))
    },
    tsv = {
      utils::read.delim(path, stringsAsFactors = FALSE, colClasses = c(id = "character"))
    },
    rds = {
      readRDS(path)
    },
    dcf = {
      raw <- as.data.frame(read.dcf(path), stringsAsFactors = FALSE)
      raw$keyed <- as.integer(raw$keyed)
      raw$min <- as.numeric(raw$min)
      raw$max <- as.numeric(raw$max)
      raw
    }
  )
  validate.form(form)
}

#' Write a questionnaire form
#'
#' @description
#' Writes a form definition to disk in one of the supported formats.
#'
#' @param form A data frame describing the questionnaire items.
#' @param path Path of the form file.
#' @param format One of `"csv"`, `"tsv"`, `"rds"` or `"dcf"`.
#' @return `path`, invisibly.
#' @export
write.form <- function(form, path, format = c("csv", "tsv", "rds", "dcf")) {
  format <- match.arg(format)
  form <- validate.form(form)

  switch(format,
    csv = {
      utils::write.csv(form, path, row.names = FALSE)
    },
    tsv = {
      utils::write.table(form, path, sep = "\t", quote = TRUE, row.names = FALSE)
    },
    rds = {
      saveRDS(form, path)
    },
    dcf = {
      write.dcf(form, path)
    }
  )
  invisible(path)
}

#' Select items for a short form
#'
#' @description
#' Keeps the best `n.per.scale` items of each scale by one criterion.
#'
#' @param form A data frame describing the questionnaire items.
#' @param stats A data frame with columns `id`, `item.rest`, `difficulty` and
#'   `variance`, one row per item.
#' @param n.per.scale Number of items to keep per scale.
#' @param criterion One of `"item.rest"`, `"difficulty"`, `"variance"` or
#'   `"first"`.
#' @return The rows of `form` that were selected, in form order.
#' @export
select.items <- function(form, stats, n.per.scale = 3,
                         criterion = c("item.rest", "difficulty", "variance", "first")) {
  criterion <- match.arg(criterion)
  stats <- stats[match(form$id, stats$id), , drop = FALSE]
  if (anyNA(stats$id)) {
    stop("`stats` must have a row for every item in `form`.", call. = FALSE)
  }

  # Higher merit is better under every criterion
  merit <- switch(criterion,
    item.rest = {
      stats$item.rest
    },
    difficulty = {
      # Prefer items of medium difficulty
      -abs(stats$difficulty - 0.5)
    },
    variance = {
      stats$variance
    },
    first = {
      -seq_len(nrow(form))
    }
  )

  keep <- logical(nrow(form))
  for (s in unique(form$scale)) {
    rows <- which(form$scale == s)
    best <- rows[order(merit[rows], decreasing = TRUE)]
    keep[utils::head(best, n.per.scale)] <- TRUE
  }
  form[keep, , drop = FALSE]
}

#' Order the items of a form for presentation
#'
#' @description
#' Arranges items grouped by scale, interleaved across scales, at random or in
#' their original order.
#'
#' @param form A data frame describing the questionnaire items.
#' @param layout One of `"original"`, `"grouped"`, `"interleaved"` or
#'   `"random"`.
#' @param seed Random seed used by `"random"`.
#' @return `form` reordered, with a `position` column.
#' @export
layout.form <- function(form, layout = c("original", "grouped", "interleaved", "random"),
                        seed = 1) {
  layout <- match.arg(layout)
  n <- nrow(form)

  ord <- switch(layout,
    original = {
      seq_len(n)
    },
    grouped = {
      order(match(form$scale, unique(form$scale)), seq_len(n))
    },
    interleaved = {
      # Rank of each item within its scale, then cycle through scales
      within <- stats::ave(seq_len(n), form$scale, FUN = seq_along)
      order(within, match(form$scale, unique(form$scale)))
    },
    random = {
      # Restore the caller's random state afterwards
      old.seed <- get0(".Random.seed", globalenv(), inherits = FALSE)
      on.exit(if (!is.null(old.seed)) assign(".Random.seed", old.seed, globalenv()))
      set.seed(seed)
      sample.int(n)
    }
  )
  form <- form[ord, , drop = FALSE]
  form$position <- seq_len(n)
  rownames(form) <- NULL
  form
}
