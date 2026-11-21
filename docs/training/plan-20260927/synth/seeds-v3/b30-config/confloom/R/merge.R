#' @title Merge a configuration list over defaults
#'
#' @description
#' Recursively fills in every key of `defaults` that `config` lacks. Values
#' present in `config` always win, except that nested lists are merged
#' rather than replaced. A nested list replacing a scalar default, or the
#' other way round, triggers a warning.
#'
#' @param config A named list of user settings.
#' @param defaults A named list of default settings.
#' @param path Dotted path of the current level, used in warnings.
#' @return A named list with the keys of `defaults` first, followed by any
#'   extra keys of `config`.
#' @keywords internal
mergeDefaults = function(config, defaults, path = NULL) {
  if (is.null(config)) {
    return(defaults)
  }

  out = defaults
  for (key in names(config)) {
    where = paste(c(path, key), collapse = ".")
    if (is.list(config[[key]]) && is.list(defaults[[key]])) {
      out[[key]] = mergeDefaults(config[[key]], defaults[[key]], where)
    } else {
      if (!is.null(defaults[[key]]) && is.list(config[[key]]) != is.list(defaults[[key]])) {
        warning(sprintf("Setting '%s' changes shape from its default.", where),
                call. = FALSE)
      }
      out[key] = list(config[[key]])
    }
  }
  out
}

#' @title Merge several configurations
#'
#' @param ... Named lists of settings, from lowest to highest precedence.
#' @param policy What to do when two configurations set the same top-level
#'   key: `"last"` keeps the later value, `"first"` keeps the earlier one,
#'   `"error"` stops, and `"deep"` merges nested lists with
#'   [mergeDefaults()].
#' @return A single named list.
#' @export
mergeConfigs = function(..., policy = c("last", "first", "error", "deep")) {
  policy = match.arg(policy)
  configs = list(...)

  out = list()
  for (config in configs) {
    clash = intersect(names(out), names(config))
    switch(policy,
      last = {
        out[names(config)] = config
      },
      first = {
        fresh = setdiff(names(config), clash)
        out[fresh] = config[fresh]
      },
      error = {
        if (length(clash) > 0) {
          stop(sprintf("Key(s) set more than once: %s.",
                       paste(clash, collapse = ", ")), call. = FALSE)
        }
        out[names(config)] = config
      },
      deep = {
        out = mergeDefaults(config, out)
      }
    )
  }
  out
}

#' @title Default settings for every section
#'
#' @return A named list with sections `server`, `database`, `logging` and
#'   `cache`.
#' @keywords internal
configDefaults = function() {
  list(
    server = list(host = "127.0.0.1", port = 8080L, workers = 2L),
    database = list(url = "sqlite::memory:", pool = 5L, timeout = 30),
    logging = list(level = "info", file = "", rotate = FALSE),
    cache = list(enabled = TRUE, dir = "", ttl = 3600)
  )
}

#' @title Apply defaults section by section
#'
#' @description
#' Each of the sections `server`, `database`, `logging` and `cache` is
#' handled the same way: warn about unknown keys, merge over the section's
#' defaults, then coerce the section's typed keys. Sections not listed in
#' `defaults` are kept as they are.
#'
#' @param config A named list of sections read from a file.
#' @param defaults A named list of default sections, see [configDefaults()].
#' @return The completed configuration.
#' @keywords internal
applySectionDefaults = function(config, defaults = configDefaults()) {
  # server
  unknown = setdiff(names(config$server), names(defaults$server))
  if (length(unknown) > 0) {
    warning(sprintf("Unknown key(s) in [%s]: %s", "server",
                    paste(unknown, collapse = ", ")), call. = FALSE)
  }
  config$server = mergeDefaults(config$server, defaults$server)
  config$server$port = as.integer(config$server$port)
  config$server$workers = as.integer(config$server$workers)

  # database
  unknown = setdiff(names(config$database), names(defaults$database))
  if (length(unknown) > 0) {
    warning(sprintf("Unknown key(s) in [%s]: %s", "database",
                    paste(unknown, collapse = ", ")), call. = FALSE)
  }
  config$database = mergeDefaults(config$database, defaults$database)
  config$database$pool = as.integer(config$database$pool)
  config$database$timeout = as.numeric(config$database$timeout)

  # logging
  unknown = setdiff(names(config$logging), names(defaults$logging))
  if (length(unknown) > 0) {
    warning(sprintf("Unknown key(s) in [%s]: %s", "logging",
                    paste(unknown, collapse = ", ")), call. = FALSE)
  }
  config$logging = mergeDefaults(config$logging, defaults$logging)
  config$logging$rotate = as.logical(config$logging$rotate)

  # cache
  unknown = setdiff(names(config$cache), names(defaults$cache))
  if (length(unknown) > 0) {
    warning(sprintf("Unknown key(s) in [%s]: %s", "cache",
                    paste(unknown, collapse = ", ")), call. = FALSE)
  }
  config$cache = mergeDefaults(config$cache, defaults$cache)
  config$cache$enabled = as.logical(config$cache$enabled)
  config$cache$ttl = as.numeric(config$cache$ttl)
  config
}

#' @title Flatten and unflatten nested configuration lists
#'
#' @description
#' `flattenConfig()` turns `list(a = list(b = 1))` into `list(a.b = 1)`;
#' `unflattenConfig()` does the reverse.
#'
#' @param config A nested named list.
#' @param sep Separator placed between the parts of a key.
#' @return A flat or nested named list.
#' @rdname flattenConfig
#' @keywords internal
flattenConfig = function(config, sep = ".") {
  out = list()
  for (key in names(config)) {
    value = config[[key]]
    if (is.list(value) && length(value) > 0) {
      inner = flattenConfig(value, sep)
      names(inner) = paste(key, names(inner), sep = sep)
      out = c(out, inner)
    } else {
      out[key] = list(value)
    }
  }
  out
}

#' @param flat A flat named list as returned by `flattenConfig()`.
#' @rdname flattenConfig
#' @keywords internal
unflattenConfig = function(flat, sep = ".") {
  heads = vapply(strsplit(names(flat), sep, fixed = TRUE), `[[`, character(1), 1)
  out = list()
  for (head in unique(heads)) {
    members = flat[heads == head]
    if (length(members) == 1 && names(members) == head) {
      out[head] = list(members[[1]])
    } else {
      names(members) = substring(names(members), nchar(head) + nchar(sep) + 1)
      out[[head]] = unflattenConfig(members, sep)
    }
  }
  out
}

#' @title Compare two configurations
#'
#' @description
#' Both configurations are flattened with [flattenConfig()] first, so keys
#' are reported in dotted form.
#'
#' @param old,new Named lists to compare.
#' @return A data frame with columns `key`, `change` (`"added"`,
#'   `"removed"` or `"changed"`), `old` and `new`, sorted by `key`. Values
#'   are formatted as strings; absent values are `NA`.
#' @export
diffConfig = function(old, new) {
  oldFlat = flattenConfig(old)
  newFlat = flattenConfig(new)
  show = function(x) paste(format(x), collapse = ", ")

  added = setdiff(names(newFlat), names(oldFlat))
  removed = setdiff(names(oldFlat), names(newFlat))
  shared = intersect(names(oldFlat), names(newFlat))
  changed = shared[!vapply(shared, function(k) {
    identical(oldFlat[[k]], newFlat[[k]])
  }, logical(1))]

  out = data.frame(
    key = c(added, removed, changed),
    change = rep(c("added", "removed", "changed"),
                 c(length(added), length(removed), length(changed))),
    old = c(rep(NA_character_, length(added)),
            vapply(oldFlat[c(removed, changed)], show, character(1))),
    new = c(vapply(newFlat[added], show, character(1)),
            rep(NA_character_, length(removed)),
            vapply(newFlat[changed], show, character(1))),
    stringsAsFactors = FALSE
  )
  out = out[order(out$key), , drop = FALSE]
  rownames(out) = NULL
  out
}
