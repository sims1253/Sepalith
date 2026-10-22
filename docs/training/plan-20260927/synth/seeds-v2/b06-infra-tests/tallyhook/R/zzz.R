#' Default values for tallyhook options
#'
#' @description Single source of truth for the options set in `.onLoad()`.
#'   Every option name is prefixed with `tallyhook.`.
#' @keywords internal
tallyhook_default_options <- function() {
  list(
    tallyhook.verbose = TRUE,
    tallyhook.cache_dir = tools::R_user_dir("tallyhook", which = "cache"),
    tallyhook.max_retries = 3L,
    tallyhook.currency = "EUR"
  )
}

#' Set default options when the package loads
#'
#' @description Only options the user has not already set are touched.
#' @param libname The library path the package was loaded from.
#' @param pkgname The name of the package.
#' @noRd
.onLoad <- function(libname, pkgname) {
  defaults <- tallyhook_default_options()
  unset <- !(names(defaults) %in% names(options()))
  if (any(unset)) {
    options(defaults[unset])
  }
  invisible()
}

#' Show a startup message in interactive sessions
#'
#' @description The message is skipped when `tallyhook.verbose` is `FALSE`.
#' @inheritParams .onLoad
#' @noRd
.onAttach <- function(libname, pkgname) {
  if (!interactive() || !isTRUE(getOption("tallyhook.verbose"))) {
    return(invisible())
  }

  version <- utils::packageVersion(pkgname)
  currency <- getOption("tallyhook.currency", "EUR")
  packageStartupMessage(
    "tallyhook ", format(version), " (default currency: ", currency, ").\n",
    "Silence this message with options(tallyhook.verbose = FALSE)."
  )
  invisible()
}

#' Read a tallyhook option
#'
#' @description Looks up `tallyhook.<name>`, falling back to the package
#'   default and then to `default` when the option is unset.
#' @param name The option name without the `tallyhook.` prefix.
#' @param default The value returned when neither the option nor a package
#'   default exists.
#' @return The option value.
#' @export
#' @examples
#' tallyhook_opt("currency")
tallyhook_opt <- function(name, default = NULL) {
  check_string(name)
  full_name <- paste0("tallyhook.", name)
  value <- getOption(full_name)
  if (!is.null(value)) {
    return(value)
  }

  # The package defaults cover sessions where .onLoad() options were unset
  defaults <- tallyhook_default_options()
  if (full_name %in% names(defaults)) {
    return(defaults[[full_name]])
  }
  return(default)
}

#' Validated option getters
#'
#' @description `tallyhook_is_verbose()` returns the `verbose` flag,
#'   `tallyhook_max_retries()` the retry count, and `tallyhook_cache_dir()`
#'   the cache directory, created on first use when `create` is `TRUE`.
#' @return A flag, a non-negative integer or a directory path.
#' @rdname option-getters
#' @keywords internal
tallyhook_is_verbose <- function() {
  value <- tallyhook_opt("verbose")
  check_flag(value, arg = "tallyhook.verbose")
  return(value)
}

#' @rdname option-getters
tallyhook_max_retries <- function() {
  value <- tallyhook_opt("max_retries")
  value <- check_count(value, arg = "tallyhook.max_retries")
  return(value)
}

#' @rdname option-getters
#' @param create Whether to create the directory if it does not exist.
tallyhook_cache_dir <- function(create = TRUE) {
  value <- tallyhook_opt("cache_dir")
  check_string(value, arg = "tallyhook.cache_dir")
  if (create && !dir.exists(value)) {
    dir.create(value, recursive = TRUE, showWarnings = FALSE)
  }
  return(value)
}

#' Temporarily set tallyhook options
#'
#' @description Prefixes each name with `tallyhook.` and sets the options
#'   until the calling frame exits. Intended for tests and short-lived
#'   overrides.
#' @param ... Name-value pairs of options, without the `tallyhook.` prefix.
#' @param .local_envir The environment whose exit restores the old values.
#' @return The previous option values, invisibly.
#' @export
#' @examples
#' f <- function() {
#'   local_tallyhook_options(currency = "USD")
#'   tallyhook_opt("currency")
#' }
#' f()
local_tallyhook_options <- function(..., .local_envir = parent.frame()) {
  new_values <- rlang::list2(...)
  if (length(new_values) == 0L) {
    return(invisible(list()))
  }
  if (!rlang::is_named(new_values)) {
    cli::cli_abort("All options passed to {.fn local_tallyhook_options} must be named.")
  }

  names(new_values) <- paste0("tallyhook.", names(new_values))
  old <- withr::local_options(new_values, .local_envir = .local_envir)
  return(invisible(old))
}
