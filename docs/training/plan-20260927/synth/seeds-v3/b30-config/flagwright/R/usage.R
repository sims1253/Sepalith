#' Create a flag specification
#'
#' Each column is checked in turn before the data frame is built.
#'
#' @param name Long flag names, without dashes.
#' @param short Single-letter aliases, or `NA` for flags without one.
#' @param type Declared types, as accepted by [parse_flag_value()].
#' @param default Default values as strings, or `NA`.
#' @param help One-line help texts.
#' @return A data frame of class `flag_spec` with columns `name`, `short`,
#'   `type`, `default` and `help`.
#' @export
new_flag_spec <- function(name, short = NA, type = "character", default = NA,
                          help = "") {
  n <- length(name)
  types <- c("logical", "integer", "numeric", "character", "path", "date", "list")
  # Check `name`
  if (!is.character(name) || anyNA(name) || !all(grepl("^[a-z][a-z0-9-]*$", name))) {
    stop("`name` must contain lower-case flag names.", call. = FALSE)
  }
  if (anyDuplicated(name)) {
    stop("`name` must not contain duplicates.", call. = FALSE)
  }

  # Check `short`
  short <- rep_len(as.character(short), n)
  if (!all(is.na(short) | grepl("^[A-Za-z]$", short))) {
    stop("`short` must contain single letters or NA.", call. = FALSE)
  }
  if (anyDuplicated(short, incomparables = NA)) {
    stop("`short` must not contain duplicates.", call. = FALSE)
  }

  # Check `type`
  type <- rep_len(as.character(type), n)
  if (!all(type %in% types)) {
    stop("`type` must contain known flag types.", call. = FALSE)
  }

  # Check `default`
  default <- rep_len(as.character(default), n)
  if (any(type == "logical" & !is.na(default))) {
    stop("`default` must be NA for logical flags.", call. = FALSE)
  }

  # Check `help`
  help <- rep_len(as.character(help), n)
  if (anyNA(help)) {
    stop("`help` must not contain NA.", call. = FALSE)
  }

  spec <- data.frame(name = name, short = short, type = type, default = default,
                     help = help, stringsAsFactors = FALSE)
  class(spec) <- c("flag_spec", class(spec))
  return(spec)
}

#' Build a flag specification from a nested list
#'
#' @param flags A named list; each element is a list with optional fields
#'   `short`, `type`, `default` and `help`.
#' @keywords internal
flag_spec_from_list <- function(flags) {
  n <- length(flags)
  short <- rep(NA_character_, n)
  type <- rep("character", n)
  default <- rep(NA_character_, n)
  help <- rep("", n)

  for (i in seq_len(n)) {
    entry <- flags[[i]]
    if (!is.null(entry$short)) short[[i]] <- entry$short
    if (!is.null(entry$type)) type[[i]] <- entry$type
    if (!is.null(entry$default)) default[[i]] <- as.character(entry$default)
    if (!is.null(entry$help)) help[[i]] <- entry$help
  }
  return(new_flag_spec(names(flags), short = short, type = type,
                       default = default, help = help))
}

#' Wrap help text to a width
#'
#' @param text A single string.
#' @param width Total line width.
#' @param indent Number of spaces before every line after the first.
#' @keywords internal
wrap_help <- function(text, width = 80L, indent = 0L) {
  if (!nzchar(text)) return("")
  lines <- strwrap(text, width = width - indent)
  pad <- strrep(" ", indent)
  if (length(lines) > 1L) {
    lines[-1L] <- paste0(pad, lines[-1L])
  }
  return(paste(lines, collapse = "\n"))
}

#' Format a flag specification
#'
#' @param x A `flag_spec` object.
#' @param width Total line width.
#' @param ... Unused.
#' @return A character vector with one element per flag: the flag names on
#'   the left, padded to 30 columns, then the wrapped help text.
#' @export
format.flag_spec <- function(x, width = 80L, ...) {
  out <- character(nrow(x))
  for (i in seq_len(nrow(x))) {
    left <- if (is.na(x$short[[i]])) {
      sprintf("    --%s", x$name[[i]])
    } else {
      sprintf("-%s, --%s", x$short[[i]], x$name[[i]])
    }
    if (x$type[[i]] != "logical") {
      left <- sprintf("%s <%s>", left, x$type[[i]])
    }
    left <- formatC(paste0("  ", left), width = -30L)

    help <- x$help[[i]]
    if (!is.na(x$default[[i]])) {
      help <- sprintf("%s [default: %s]", help, x$default[[i]])
    }
    out[[i]] <- paste0(left, wrap_help(help, width, 30L))
  }
  return(out)
}

#' Print a flag specification
#'
#' @param x A `flag_spec` object.
#' @param width Total line width.
#' @param ... Unused.
#' @export
print.flag_spec <- function(x, width = 80L, ...) {
  cat(sprintf("<flag_spec> %d flag(s)\n", nrow(x)))
  lines <- format(x, width = width)
  if (length(lines) > 0L) {
    cat(lines, sep = "\n")
  }
  return(invisible(x))
}

#' Summarise a flag specification
#'
#' @param object A `flag_spec` object.
#' @param ... Unused.
#' @return A list with elements `n_flags`, `n_short`, `n_switches`,
#'   `n_defaults`, `required`, `undocumented` and `types`.
#' @export
summary.flag_spec <- function(object, ...) {
  n_flags <- nrow(object)
  n_short <- sum(!is.na(object$short))
  n_switches <- sum(object$type == "logical")
  n_defaults <- sum(!is.na(object$default))
  # Flags that take a value but have no default must be given by the user
  required <- object$name[object$type != "logical" & is.na(object$default)]
  undocumented <- object$name[!nzchar(object$help)]
  types <- table(factor(object$type, levels = c("logical", "integer", "numeric",
                                                "character", "path", "date",
                                                "list")))

  return(list(
    n_flags = n_flags,
    n_short = n_short,
    n_switches = n_switches,
    n_defaults = n_defaults,
    required = required,
    undocumented = undocumented,
    types = types
  ))
}

#' Build the full usage text for a script
#'
#' The text has up to five parts, in order: the `Usage:` line, the
#' description, the `Options:` block with one line per flag, the
#' `Arguments:` block listing positionals and the `Environment:` block
#' listing environment variables. Empty parts are left out.
#'
#' @param spec A `flag_spec` object.
#' @param prog Name of the script as typed by the user.
#' @param description Optional paragraph describing the script.
#' @param positional Named character vector of positional arguments and
#'   their help texts.
#' @param env Named character vector of environment variables and their
#'   help texts.
#' @param width Total line width.
#' @return A single string.
#' @export
format_usage <- function(spec, prog, description = NULL,
                         positional = character(), env = character(),
                         width = 80L) {
  parts <- character()

  # 1. Usage line
  usage <- sprintf("Usage: %s [options]", prog)
  if (length(positional) > 0L) {
    usage <- paste(usage, paste0("<", names(positional), ">", collapse = " "))
  }
  parts <- c(parts, usage)

  # 2. Description
  if (!is.null(description) && nzchar(description)) {
    parts <- c(parts, wrap_help(description, width, 0L))
  }

  # 3. Options block
  if (nrow(spec) > 0L) {
    lines <- format(spec, width = width)
    parts <- c(parts, paste(c("Options:", lines), collapse = "\n"))
  }

  # 4. Arguments block
  if (length(positional) > 0L) {
    left <- formatC(paste0("  ", names(positional)), width = -30L)
    lines <- paste0(left, vapply(unname(positional), wrap_help, character(1),
                                 width = width, indent = 30L))
    parts <- c(parts, paste(c("Arguments:", lines), collapse = "\n"))
  }

  # 5. Environment block
  if (length(env) > 0L) {
    left <- formatC(paste0("  ", names(env)), width = -30L)
    lines <- paste0(left, vapply(unname(env), wrap_help, character(1),
                                 width = width, indent = 30L))
    parts <- c(parts, paste(c("Environment:", lines), collapse = "\n"))
  }
  return(paste(parts, collapse = "\n\n"))
}
