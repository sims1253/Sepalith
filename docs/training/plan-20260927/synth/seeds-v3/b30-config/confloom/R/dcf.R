#' Coerce a DCF field value
#'
#' @description
#' DCF files store every value as text. This converts one value to the
#' requested type; `"auto"` guesses from the text itself. Dates use the ISO
#' format and lists are comma-separated, as in `Depends: a, b`.
#'
#' @param x A single string read from a DCF file.
#' @param type Target type.
#' @return The converted value, or `NA` of the target type when `x` is `NA`.
#' @family dcf
#' @keywords internal
coerceDcfValue = function(x, type = c("auto", "logical", "integer", "numeric",
                                      "date", "list", "character")) {
  type = match.arg(type)
  if (type == "auto") {
    type = if (grepl("^(yes|no|true|false)$", x, ignore.case = TRUE)) {
      "logical"
    } else if (grepl("^-?[0-9]+$", x)) {
      "integer"
    } else if (!is.na(suppressWarnings(as.numeric(x)))) {
      "numeric"
    } else if (grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", x)) {
      "date"
    } else if (grepl(",", x, fixed = TRUE)) {
      "list"
    } else {
      "character"
    }
  }

  switch(type,
    logical = {
      tolower(x) %in% c("yes", "true")
    },
    integer = {
      as.integer(x)
    },
    numeric = {
      as.numeric(x)
    },
    date = {
      as.Date(x)
    },
    list = {
      parts = trimws(strsplit(x, ",", fixed = TRUE)[[1]])
      parts[nzchar(parts)]
    },
    character = {
      trimws(x)
    }
  )
}

#' Convert one DCF record to a list
#'
#' @description
#' Empty fields are dropped; the rest are coerced with [coerceDcfValue()].
#'
#' @param record A named character vector, one row of [read.dcf()] output.
#' @param types A named character vector of target types; fields not listed
#'   use `"auto"`.
#' @family dcf
#' @keywords internal
dcfRecordToList = function(record, types = character()) {
  record = record[!is.na(record)]
  out = vector("list", length(record))
  names(out) = names(record)
  for (field in names(record)) {
    type = if (field %in% names(types)) types[[field]] else "auto"
    out[[field]] = coerceDcfValue(record[[field]], type)
  }
  out
}

#' Read a DCF configuration file
#'
#' @description
#' Each record becomes one element of the result, named by the field given
#' in `key`. Records without that field are an error.
#'
#' @param path Path to the DCF file.
#' @param key Field whose value names each record.
#' @param types A named character vector of target types; fields not listed
#'   use `"auto"`.
#' @return A named list of records, each a named list of fields.
#' @family dcf
#' @export
readDcfConfig = function(path, key = "Name", types = character()) {
  if (!file.exists(path)) {
    stop(sprintf("Config file '%s' does not exist.", path), call. = FALSE)
  }
  records = read.dcf(path, keep.white = character())
  if (!key %in% colnames(records)) {
    stop(sprintf("Config file '%s' has no '%s' field.", path, key), call. = FALSE)
  }

  out = vector("list", nrow(records))
  for (i in seq_len(nrow(records))) {
    if (is.na(records[i, key])) {
      stop(sprintf("Record %d in '%s' has no '%s' field.", i, path, key),
           call. = FALSE)
    }
    out[[i]] = dcfRecordToList(records[i, ], types)
  }
  names(out) = records[, key]
  out
}

#' Write a DCF configuration file
#'
#' @description
#' The inverse of [readDcfConfig()]: one record per element of `config`,
#' with logical fields written as `yes`/`no` and vectors joined by commas.
#'
#' @param config A named list of records, each a named list of fields.
#' @inheritParams readDcfConfig
#' @family dcf
#' @export
writeDcfConfig = function(config, path, key = "Name") {
  fields = unique(c(key, unlist(lapply(config, names), use.names = FALSE)))
  records = matrix(NA_character_, nrow = length(config), ncol = length(fields),
                   dimnames = list(NULL, fields))

  for (i in seq_along(config)) {
    record = config[[i]]
    record[[key]] = names(config)[[i]]
    for (field in names(record)) {
      value = record[[field]]
      records[i, field] = if (is.logical(value)) {
        if (value) "yes" else "no"
      } else {
        paste(format(value), collapse = ", ")
      }
    }
  }
  write.dcf(records, file = path, keep.white = character())
  invisible(path)
}

#' Validate a job description
#'
#' @description
#' Checks one record read by [readDcfConfig()] field by field.
#'
#' @param job A named list of fields for one job.
#' @return A character vector of problems, empty when the job is valid. The
#'   fields checked are `Name`, `Command`, `Schedule`, `Retries`, `Timeout`
#'   and `Enabled`.
#' @family dcf
#' @keywords internal
validateJobRecord = function(job) {
  problems = character()

  # Name: required, a single identifier
  if (is.null(job$Name)) {
    problems = c(problems, "Field 'Name' is missing.")
  } else if (!grepl("^[A-Za-z][A-Za-z0-9_.-]*$", job$Name)) {
    problems = c(problems, "Field 'Name' is not a valid identifier.")
  }

  # Command: required, non-empty text
  if (is.null(job$Command)) {
    problems = c(problems, "Field 'Command' is missing.")
  } else if (!nzchar(trimws(job$Command))) {
    problems = c(problems, "Field 'Command' is empty.")
  }

  # Schedule: required, one of the known keywords
  if (is.null(job$Schedule)) {
    problems = c(problems, "Field 'Schedule' is missing.")
  } else if (!job$Schedule %in% c("hourly", "daily", "weekly", "monthly")) {
    problems = c(problems, "Field 'Schedule' is not a known schedule.")
  }

  # Retries: optional, a non-negative integer
  if (!is.null(job$Retries) && (!is.numeric(job$Retries) || job$Retries < 0)) {
    problems = c(problems, "Field 'Retries' must be a non-negative number.")
  }

  # Timeout: optional, a positive number
  if (!is.null(job$Timeout) && (!is.numeric(job$Timeout) || job$Timeout <= 0)) {
    problems = c(problems, "Field 'Timeout' must be a positive number.")
  }

  # Enabled: optional, yes or no
  if (!is.null(job$Enabled) && !is.logical(job$Enabled)) {
    problems = c(problems, "Field 'Enabled' must be yes or no.")
  }
  problems
}

#' Tabulate jobs
#'
#' @param config A named list of records, each a named list of fields.
#' @return A data frame with one row per job and columns `name`,
#'   `schedule`, `retries`, `timeout`, `enabled` and `after`. Missing fields
#'   become `NA`, except `enabled`, which defaults to `TRUE`.
#' @family dcf
#' @export
summariseJobs = function(config) {
  field = function(job, key, missing) {
    value = job[[key]]
    if (is.null(value)) missing else value
  }
  name = names(config)
  schedule = vapply(config, field, character(1), key = "Schedule",
                    missing = NA_character_, USE.NAMES = FALSE)
  retries = vapply(config, field, integer(1), key = "Retries",
                   missing = NA_integer_, USE.NAMES = FALSE)
  timeout = vapply(config, function(job) as.numeric(field(job, "Timeout", NA)),
                   numeric(1), USE.NAMES = FALSE)
  enabled = vapply(config, field, logical(1), key = "Enabled",
                   missing = TRUE, USE.NAMES = FALSE)
  after = vapply(config, function(job) paste(field(job, "After", ""), collapse = ", "),
                 character(1), USE.NAMES = FALSE)

  data.frame(name = name, schedule = schedule, retries = retries,
             timeout = timeout, enabled = enabled, after = after,
             stringsAsFactors = FALSE)
}
