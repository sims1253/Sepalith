#' Define a single typed flag
#'
#' Each flag object carries its name, a default and, for choice flags, the
#' accepted values. The class is `flag_<type>` so that [coerce_flag()] can
#' dispatch on it.
#'
#' @param name Long flag name, without dashes.
#' @param type One of `"switch"`, `"count"`, `"number"`, `"choice"` or
#'   `"text"`.
#' @param default Default value, already of the right type.
#' @param choices Accepted values for choice flags.
#' @param help One-line help text.
#' @param multiple Whether the flag may be given more than once; its value
#'   is then a vector with one element per occurrence.
#' @return An object of class `flag_<type>` and `flag`.
#' @export
flag <- function(name, type = c("switch", "count", "number", "choice", "text"),
                 default = NULL, choices = NULL, help = "", multiple = FALSE) {
  type <- match.arg(type)
  if (!is.character(name) || length(name) != 1L || !nzchar(name)) {
    stop("`name` must be a single non-empty string.", call. = FALSE)
  }

  # Multi-valued flags may have several defaults
  ok_length <- function(x) length(x) == 1L || (multiple && length(x) >= 1L)
  switch(type,
    switch = {
      if (is.null(default)) default <- FALSE
      stopifnot(is.logical(default), ok_length(default))
    },
    count = {
      if (is.null(default)) default <- 0L
      stopifnot(is.numeric(default), ok_length(default), all(default >= 0))
    },
    number = {
      if (is.null(default)) default <- NA_real_
      stopifnot(is.numeric(default), ok_length(default))
    },
    choice = {
      if (is.null(choices)) stop("Choice flags need `choices`.", call. = FALSE)
      if (is.null(default)) default <- choices[[1]]
      stopifnot(is.character(default), ok_length(default), all(default %in% choices))
    },
    text = {
      if (is.null(default)) default <- NA_character_
      stopifnot(is.character(default), ok_length(default))
    }
  )

  structure(
    list(name = name, default = default, choices = choices, help = help,
         multiple = multiple),
    class = c(paste0("flag_", type), "flag")
  )
}

#' Coerce a raw command-line value for a flag
#'
#' @param flag A flag object created by [flag()].
#' @param value A string from the command line, or `NULL` when the flag was
#'   given without a value. Several strings are accepted only for flags
#'   created with `multiple = TRUE` and are coerced one by one.
#' @return The coerced value.
#' @export
coerce_flag <- function(flag, value) {
  UseMethod("coerce_flag")
}

#' @rdname coerce_flag
#' @export
coerce_flag.flag_switch <- function(flag, value) {
  # Repeated flags are coerced one occurrence at a time
  if (length(value) > 1L) {
    if (!isTRUE(flag$multiple)) {
      stop(sprintf("Flag `--%s` can only be given once.", flag$name), call. = FALSE)
    }
    return(vapply(value, function(v) coerce_flag(flag, v), logical(1), USE.NAMES = FALSE))
  }

  if (is.null(value)) {
    return(TRUE)
  }
  out <- switch(tolower(value),
    true = , yes = , on = TRUE,
    false = , no = , off = FALSE,
    NA
  )
  if (is.na(out)) {
    stop(sprintf("Flag `--%s` got invalid value \"%s\".", flag$name, value),
         call. = FALSE)
  }
  return(out)
}

#' @rdname coerce_flag
#' @export
coerce_flag.flag_count <- function(flag, value) {
  # Repeated flags are coerced one occurrence at a time
  if (length(value) > 1L) {
    if (!isTRUE(flag$multiple)) {
      stop(sprintf("Flag `--%s` can only be given once.", flag$name), call. = FALSE)
    }
    return(vapply(value, function(v) coerce_flag(flag, v), integer(1), USE.NAMES = FALSE))
  }

  if (is.null(value)) {
    return(as.integer(flag$default) + 1L)
  }
  out <- suppressWarnings(as.integer(value))
  if (is.na(out) || out < 0L) {
    stop(sprintf("Flag `--%s` got invalid value \"%s\".", flag$name, value),
         call. = FALSE)
  }
  return(out)
}

#' @rdname coerce_flag
#' @export
coerce_flag.flag_number <- function(flag, value) {
  # Repeated flags are coerced one occurrence at a time
  if (length(value) > 1L) {
    if (!isTRUE(flag$multiple)) {
      stop(sprintf("Flag `--%s` can only be given once.", flag$name), call. = FALSE)
    }
    return(vapply(value, function(v) coerce_flag(flag, v), numeric(1), USE.NAMES = FALSE))
  }

  if (is.null(value)) {
    stop(sprintf("Flag `--%s` needs a value.", flag$name), call. = FALSE)
  }
  out <- suppressWarnings(as.numeric(value))
  if (is.na(out)) {
    stop(sprintf("Flag `--%s` got invalid value \"%s\".", flag$name, value),
         call. = FALSE)
  }
  return(out)
}

#' @rdname coerce_flag
#' @export
coerce_flag.flag_choice <- function(flag, value) {
  # Repeated flags are coerced one occurrence at a time
  if (length(value) > 1L) {
    if (!isTRUE(flag$multiple)) {
      stop(sprintf("Flag `--%s` can only be given once.", flag$name), call. = FALSE)
    }
    return(vapply(value, function(v) coerce_flag(flag, v), character(1), USE.NAMES = FALSE))
  }

  if (is.null(value)) {
    stop(sprintf("Flag `--%s` needs a value.", flag$name), call. = FALSE)
  }
  out <- flag$choices[pmatch(value, flag$choices)]
  if (is.na(out)) {
    stop(sprintf("Flag `--%s` got invalid value \"%s\".", flag$name, value),
         call. = FALSE)
  }
  return(out)
}

#' @rdname coerce_flag
#' @export
coerce_flag.flag_text <- function(flag, value) {
  # Repeated flags are coerced one occurrence at a time
  if (length(value) > 1L) {
    if (!isTRUE(flag$multiple)) {
      stop(sprintf("Flag `--%s` can only be given once.", flag$name), call. = FALSE)
    }
    return(vapply(value, function(v) coerce_flag(flag, v), character(1), USE.NAMES = FALSE))
  }

  if (is.null(value)) {
    stop(sprintf("Flag `--%s` needs a value.", flag$name), call. = FALSE)
  }
  out <- trimws(value)
  if (!nzchar(out)) {
    stop(sprintf("Flag `--%s` got invalid value \"%s\".", flag$name, value),
         call. = FALSE)
  }
  return(out)
}

#' Describe a set of flags on the console
#'
#' @param flags A list of flag objects created by [flag()].
#' @importFrom cli cli_h2 cli_dl
#' @export
describe_flags <- function(flags) {
  cli_h2("Flags")

  items <- vapply(flags, function(f) {
    type <- sub("^flag_", "", class(f)[[1]])
    default <- if (length(f$default) == 1L && !is.na(f$default)) {
      format(f$default)
    } else {
      "none"
    }
    sprintf("{.emph %s} %s (default: {.val %s})", type, f$help, default)
  }, character(1))
  names(items) <- paste0("--", vapply(flags, `[[`, character(1), "name"))
  cli_dl(items)
  return(invisible(flags))
}
