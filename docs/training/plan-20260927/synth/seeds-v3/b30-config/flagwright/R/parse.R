#' Split one command-line token into a flag name and value
#'
#' Long flags may carry an inline value (`--keep=3`). Short flags are
#' single letters and are mapped through `aliases`. Anything else, including
#' negative numbers, is reported as a positional argument.
#'
#' @param token A single command-line token.
#' @param aliases A named character vector mapping short letters to long
#'   flag names, e.g. `c(v = "verbose")`.
#' @return A list with elements `name` and `value`; `name` is `NA` for
#'   positional arguments and `value` is `NA` when no inline value was given.
#' @keywords internal
split_flag_token <- function(token, aliases = character()) {
  # Long flag, with or without an inline value
  if (startsWith(token, "--") && nchar(token) > 2L) {
    body <- substring(token, 3L)
    eq <- regexpr("=", body, fixed = TRUE)
    if (eq > 0L) {
      return(list(
        name = substring(body, 1L, eq - 1L),
        value = substring(body, eq + 1L)
      ))
    }
    return(list(name = body, value = NA_character_))
  }

  # Short flag, looked up in the aliases
  if (grepl("^-[A-Za-z]$", token)) {
    key <- substring(token, 2L)
    name <- if (key %in% names(aliases)) aliases[[key]] else key
    return(list(name = name, value = NA_character_))
  }
  return(list(name = NA_character_, value = NA_character_))
}

#' Expand bundled short flags
#'
#' Turns `-vn` into `-v -n` so that each short flag can be looked up on
#' its own. Long flags and positionals pass through unchanged.
#'
#' @param args A character vector of command-line arguments.
#' @keywords internal
expand_short_flags <- function(args) {
  out <- vector("list", length(args))
  for (i in seq_along(args)) {
    arg <- args[[i]]
    if (grepl("^-[A-Za-z]{2,}$", arg)) {
      letters_used <- strsplit(substring(arg, 2L), "")[[1]]
      out[[i]] <- paste0("-", letters_used)
    } else {
      out[[i]] <- arg
    }
  }

  return(unlist(out, use.names = FALSE))
}

#' Convert a raw flag value to its declared type
#'
#' @param value A single string taken from the command line.
#' @param type The declared type of the flag.
#' @param flag The long name of the flag, used in error messages.
#' @return The converted value. Errors when `value` cannot be converted.
#'   Dates use the ISO format; lists are comma-separated.
#' @export
parse_flag_value <- function(value,
                             type = c("logical", "integer", "numeric",
                                      "character", "path", "date", "list"),
                             flag = "value") {
  type <- match.arg(type)
  msg <- sprintf("Flag `--%s` expects a %s value, not \"%s\".", flag, type, value)

  switch(type,
    logical = {
      out <- switch(tolower(value),
        true = , yes = , `1` = TRUE,
        false = , no = , `0` = FALSE,
        NA
      )
      if (is.na(out)) stop(msg, call. = FALSE)
    },
    integer = {
      out <- suppressWarnings(as.integer(value))
      if (is.na(out)) stop(msg, call. = FALSE)
    },
    numeric = {
      out <- suppressWarnings(as.numeric(value))
      if (is.na(out)) stop(msg, call. = FALSE)
    },
    character = {
      out <- value
      if (!nzchar(out)) stop(msg, call. = FALSE)
    },
    path = {
      out <- path.expand(value)
      if (!nzchar(out)) stop(msg, call. = FALSE)
    },
    date = {
      out <- as.Date(value, optional = TRUE)
      if (is.na(out)) stop(msg, call. = FALSE)
    },
    list = {
      out <- trimws(strsplit(value, ",", fixed = TRUE)[[1]])
      if (length(out) == 0L || !all(nzchar(out))) stop(msg, call. = FALSE)
    }
  )
  return(out)
}

#' Look up the value of one flag
#'
#' Reads `--name=value` or `--name value`. When the flag is given more than
#' once the last occurrence wins, as in most shells.
#'
#' @param args A character vector of command-line arguments.
#' @param name The long name of the flag, without dashes.
#' @param default Value returned when the flag is absent.
#' @return The raw string value, or `default`.
#' @keywords internal
flag_value <- function(args, name, default = NULL) {
  long <- paste0("--", name)
  value <- default

  i <- 1L
  while (i <= length(args)) {
    if (startsWith(args[[i]], paste0(long, "="))) {
      value <- substring(args[[i]], nchar(long) + 2L)
    } else if (args[[i]] == long) {
      if (i == length(args)) {
        stop(sprintf("Flag `--%s` needs a value.", name), call. = FALSE)
      }
      i <- i + 1L
      value <- args[[i]]
    }
    i <- i + 1L
  }
  return(value)
}

#' Parse command-line arguments for the backup script
#'
#' Walks `args` once and fills one slot per known flag. Unknown flags are an
#' error; anything that does not start with a dash is kept as a positional.
#'
#' @param args A character vector of command-line arguments.
#' @return A list with elements `source`, `dest`, `keep`, `compress`,
#'   `dry_run`, `verbose` and `positional`.
#' @keywords internal
parse_backup_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  out <- list(
    source = NULL,
    dest = NULL,
    keep = 7L,
    compress = "gzip",
    dry_run = FALSE,
    verbose = FALSE,
    positional = character()
  )
  args <- expand_short_flags(args)
  aliases <- c(n = "dry-run", v = "verbose")

  i <- 1L
  while (i <= length(args)) {
    tok <- split_flag_token(args[[i]], aliases)
    if (is.na(tok$name)) {
      out$positional <- c(out$positional, args[[i]])
      i <- i + 1L
      next
    }

    # Flags that take a value read it inline or from the next argument
    if (tok$name %in% c("source", "dest", "keep", "compress") && is.na(tok$value)) {
      if (i == length(args)) {
        stop(sprintf("Flag `--%s` needs a value.", tok$name), call. = FALSE)
      }
      i <- i + 1L
      tok$value <- args[[i]]
    }
    switch(tok$name,
      source = {
        out$source <- parse_flag_value(tok$value, "path", "source")
      },
      dest = {
        out$dest <- parse_flag_value(tok$value, "path", "dest")
      },
      keep = {
        out$keep <- parse_flag_value(tok$value, "integer", "keep")
      },
      compress = {
        out$compress <- parse_flag_value(tok$value, "character", "compress")
      },
      `dry-run` = {
        out$dry_run <- TRUE
      },
      verbose = {
        out$verbose <- TRUE
      },
      stop(sprintf("Unknown flag `--%s`.", tok$name), call. = FALSE)
    )
    i <- i + 1L
  }

  return(out)
}

#' Parse command-line arguments for the report script
#'
#' Walks `args` once and fills one slot per known flag. Unknown flags are an
#' error; anything that does not start with a dash is kept as a positional.
#'
#' @param args A character vector of command-line arguments.
#' @return A list with elements `input`, `output`, `format`, `from`, `to`,
#'   `quiet` and `positional`.
#' @keywords internal
parse_report_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  out <- list(
    input = NULL,
    output = NULL,
    format = "html",
    from = NULL,
    to = NULL,
    quiet = FALSE,
    positional = character()
  )
  args <- expand_short_flags(args)
  aliases <- c(o = "output", q = "quiet")

  i <- 1L
  while (i <= length(args)) {
    tok <- split_flag_token(args[[i]], aliases)
    if (is.na(tok$name)) {
      out$positional <- c(out$positional, args[[i]])
      i <- i + 1L
      next
    }

    # Flags that take a value read it inline or from the next argument
    if (tok$name %in% c("input", "output", "format", "from", "to") && is.na(tok$value)) {
      if (i == length(args)) {
        stop(sprintf("Flag `--%s` needs a value.", tok$name), call. = FALSE)
      }
      i <- i + 1L
      tok$value <- args[[i]]
    }
    switch(tok$name,
      input = {
        out$input <- parse_flag_value(tok$value, "path", "input")
      },
      output = {
        out$output <- parse_flag_value(tok$value, "path", "output")
      },
      format = {
        out$format <- parse_flag_value(tok$value, "character", "format")
      },
      from = {
        out$from <- parse_flag_value(tok$value, "date", "from")
      },
      to = {
        out$to <- parse_flag_value(tok$value, "date", "to")
      },
      quiet = {
        out$quiet <- TRUE
      },
      stop(sprintf("Unknown flag `--%s`.", tok$name), call. = FALSE)
    )
    i <- i + 1L
  }

  return(out)
}

#' Parse arguments against a flag specification
#'
#' The specification is a data frame with one row per flag and columns
#' `name`, `short`, `type` and `default`. Every flag starts at its default;
#' flags of type `"logical"` are switches and never consume a value.
#'
#' @param args A character vector of command-line arguments.
#' @param spec A data frame describing the accepted flags.
#' @return A named list with one element per row of `spec`, plus
#'   `positional`.
#' @export
parse_args_with_spec <- function(args, spec) {
  # Every flag starts at its default; switches start off
  out <- vector("list", nrow(spec))
  names(out) <- spec$name
  for (j in seq_len(nrow(spec))) {
    if (spec$type[[j]] == "logical") {
      out[j] <- list(FALSE)
    } else if (!is.na(spec$default[[j]])) {
      out[j] <- list(parse_flag_value(spec$default[[j]], spec$type[[j]], spec$name[[j]]))
    }
  }
  aliases <- stats::setNames(spec$name, spec$short)
  aliases <- aliases[!is.na(names(aliases))]

  args <- expand_short_flags(args)
  positional <- character()

  i <- 1L
  while (i <= length(args)) {
    tok <- split_flag_token(args[[i]], aliases)
    if (is.na(tok$name)) {
      positional <- c(positional, args[[i]])
      i <- i + 1L
      next
    }
    row <- match(tok$name, spec$name)
    if (is.na(row)) {
      stop(sprintf("Unknown flag `--%s`.", tok$name), call. = FALSE)
    }
    if (spec$type[[row]] == "logical") {
      out[[tok$name]] <- is.na(tok$value) ||
        parse_flag_value(tok$value, "logical", tok$name)
    } else {
      if (is.na(tok$value)) {
        if (i == length(args)) {
          stop(sprintf("Flag `--%s` needs a value.", tok$name), call. = FALSE)
        }
        i <- i + 1L
        tok$value <- args[[i]]
      }
      out[[tok$name]] <- parse_flag_value(tok$value, spec$type[[row]], tok$name)
    }
    i <- i + 1L
  }

  out$positional <- positional
  return(out)
}
