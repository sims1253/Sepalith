#' Validate a job table
#'
#' @param jobs A data frame with columns `id`, `release`, `proc_time`, `due`
#'   and `weight`.
#' @return `jobs`, invisibly, when every column passes its checks.
#' @keywords internal
validate_jobs <- function(jobs) {
  if (!is.data.frame(jobs)) {
    stop("`jobs` must be a data frame.", call. = FALSE)
  }
  required <- c("id", "release", "proc_time", "due", "weight")
  absent <- setdiff(required, names(jobs))
  if (length(absent) > 0) {
    stop("`jobs` is missing columns: ", paste(absent, collapse = ", "), call. = FALSE)
  }

  # id
  if (anyNA(jobs$id) || anyDuplicated(jobs$id) > 0) {
    stop(sprintf("Column `%s` must be unique and not missing.", "id"), call. = FALSE)
  }
  # release
  if (!is.numeric(jobs$release) || anyNA(jobs$release)) {
    stop(sprintf("Column `%s` must be numeric without missing values.", "release"),
      call. = FALSE)
  }
  if (any(jobs$release < 0)) {
    stop(sprintf("Column `%s` must be non-negative.", "release"), call. = FALSE)
  }
  # proc_time
  if (!is.numeric(jobs$proc_time) || anyNA(jobs$proc_time)) {
    stop(sprintf("Column `%s` must be numeric without missing values.", "proc_time"),
      call. = FALSE)
  }
  if (any(jobs$proc_time <= 0)) {
    stop(sprintf("Column `%s` must be positive.", "proc_time"), call. = FALSE)
  }
  # due
  if (!is.numeric(jobs$due) || anyNA(jobs$due)) {
    stop(sprintf("Column `%s` must be numeric without missing values.", "due"),
      call. = FALSE)
  }
  if (any(jobs$due < jobs$release)) {
    stop(sprintf("Column `%s` must not precede `%s`.", "due", "release"), call. = FALSE)
  }
  # weight
  if (!is.numeric(jobs$weight) || anyNA(jobs$weight)) {
    stop(sprintf("Column `%s` must be numeric without missing values.", "weight"),
      call. = FALSE)
  }
  if (any(jobs$weight <= 0)) {
    stop(sprintf("Column `%s` must be positive.", "weight"), call. = FALSE)
  }
  invisible(jobs)
}

#' Read a job table from disk
#'
#' @param path Path to the file.
#' @param format File format, one of `"csv"`, `"tsv"`, `"semicolon"` or `"rds"`.
#' @param id_col Name of the column holding the job identifier in the file.
#' @return A data frame with the identifier column renamed to `id`.
#' @export
read_job_table <- function(path, format = c("csv", "tsv", "semicolon", "rds"),
                           id_col = "id") {
  format <- match.arg(format)
  if (!file.exists(path)) {
    stop(sprintf("File '%s' does not exist.", path), call. = FALSE)
  }

  jobs <- switch(format,
    csv = {
      # Comma separated, dot decimal
      raw <- utils::read.csv(path, stringsAsFactors = FALSE)
      raw[!vapply(raw, function(col) all(is.na(col)), logical(1))]
    },
    tsv = {
      # Tab separated, dot decimal
      raw <- utils::read.delim(path, stringsAsFactors = FALSE)
      raw[!vapply(raw, function(col) all(is.na(col)), logical(1))]
    },
    semicolon = {
      # Semicolon separated, comma decimal
      raw <- utils::read.csv2(path, stringsAsFactors = FALSE)
      raw[!vapply(raw, function(col) all(is.na(col)), logical(1))]
    },
    rds = {
      # Serialised data frame, columns kept as stored
      raw <- readRDS(path)
      as.data.frame(raw, stringsAsFactors = FALSE)
    }
  )

  if (!id_col %in% names(jobs)) {
    stop(sprintf("Column '%s' not found in '%s'.", id_col, path), call. = FALSE)
  }
  names(jobs)[names(jobs) == id_col] <- "id"
  attr(jobs, "source") <- normalizePath(path)
  jobs
}

#' Coerce the columns of a raw job table
#'
#' @param jobs A data frame with columns `id`, `release`, `proc_time`, `due`
#'   and `weight`, possibly stored as text.
#' @param time_unit Unit of the time columns in the input, one of `"min"`,
#'   `"hour"` or `"shift"`; results are always in minutes.
#' @param shift_length Length of one shift in hours.
#' @return `jobs` with numeric time columns in minutes and default weights.
#' @export
coerce_job_columns <- function(jobs, time_unit = c("min", "hour", "shift"),
                               shift_length = 8) {
  time_unit <- match.arg(time_unit)
  to_minutes <- c(min = 1, hour = 60, shift = 60 * shift_length)[[time_unit]]
  to_number <- function(x) suppressWarnings(as.numeric(gsub(",", ".", trimws(x))))

  # id stays character
  jobs$id <- trimws(as.character(jobs$id))
  # release
  jobs$release <- to_number(jobs$release) * to_minutes
  if (anyNA(jobs$release)) {
    warning(sprintf("Column `%s` has %d value(s) that are not numbers.", "release",
      sum(is.na(jobs$release))), call. = FALSE)
  }
  # proc_time
  jobs$proc_time <- to_number(jobs$proc_time) * to_minutes
  if (anyNA(jobs$proc_time)) {
    warning(sprintf("Column `%s` has %d value(s) that are not numbers.", "proc_time",
      sum(is.na(jobs$proc_time))), call. = FALSE)
  }
  # due
  jobs$due <- to_number(jobs$due) * to_minutes
  if (anyNA(jobs$due)) {
    warning(sprintf("Column `%s` has %d value(s) that are not numbers.", "due",
      sum(is.na(jobs$due))), call. = FALSE)
  }
  # weight is unitless and defaults to one
  if (is.null(jobs$weight)) {
    jobs$weight <- 1
  } else {
    jobs$weight <- to_number(jobs$weight)
    jobs$weight[is.na(jobs$weight)] <- 1
  }
  jobs
}

#' Summarise the numeric columns of a job table
#'
#' @param jobs A data frame with columns `id`, `release`, `proc_time`, `due`
#'   and `weight`.
#' @return A data frame with one row per column among `release`, `proc_time`,
#'   `due`, `weight` and `slack`, and columns `min`, `median`, `mean`, `max`
#'   and `sd`.
#' @export
summarise_jobs <- function(jobs) {
  columns <- c("release", "proc_time", "due", "weight", "slack")
  jobs$slack <- jobs$due - jobs$release - jobs$proc_time
  out <- data.frame(
    column = columns,
    min = NA_real_,
    median = NA_real_,
    mean = NA_real_,
    max = NA_real_,
    sd = NA_real_
  )

  for (i in seq_along(columns)) {
    x <- jobs[[columns[[i]]]]
    out$min[i] <- min(x)
    out$median[i] <- stats::median(x)
    out$mean[i] <- mean(x)
    out$max[i] <- max(x)
    out$sd[i] <- if (length(x) > 1) stats::sd(x) else NA_real_
  }

  # Jobs with negative slack cannot finish on time even if started at release
  attr(out, "infeasible") <- jobs$id[jobs$slack < 0]
  out
}

#' Split a job table into dispatch groups
#'
#' @param jobs A data frame with columns `id`, `release`, `proc_time`, `due`,
#'   `weight` and `family`.
#' @param by Grouping, one of `"family"`, `"due_week"`, `"priority"` or
#'   `"size"`.
#' @param week_length Length of a week in the time unit of `due`.
#' @return A named list of data frames, one per group.
#' @export
split_jobs <- function(jobs, by = c("family", "due_week", "priority", "size"),
                       week_length = 7 * 24 * 60) {
  by <- match.arg(by)

  group <- switch(by,
    family = {
      # Product family as recorded on the job
      key <- as.character(jobs$family)
      ifelse(is.na(key), "unknown", key)
    },
    due_week = {
      # Week in which the job is due
      key <- floor(jobs$due / week_length) + 1
      paste0("week_", key)
    },
    priority = {
      # Customer weight split at its median
      key <- jobs$weight > stats::median(jobs$weight)
      ifelse(key, "high", "normal")
    },
    size = {
      # Processing time split at its median
      key <- jobs$proc_time > stats::median(jobs$proc_time)
      ifelse(key, "long", "short")
    }
  )

  out <- split(jobs, group)
  lapply(out, function(part) {
    rownames(part) <- NULL
    part
  })
}
