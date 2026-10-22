#' @title Parse a dependency field into a data frame
#' @description Splits a comma-separated dependency field such as `Imports`
#'   into package names and optional version constraints.
#' @param x A single string holding the raw field value.
#' @param type The dependency type recorded in the result, e.g. `"Imports"`.
#' @return A data frame with columns `type`, `package`, `op` and `version`.
#' @export
deps_parse <- function(x, type) {
  empty <- data.frame(
    type = character(), package = character(), op = character(),
    version = character(), stringsAsFactors = FALSE
  )
  if (is.na(x) || !nzchar(trimws(x))) {
    return(empty)
  }

  entries <- trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
  entries <- entries[nzchar(entries)]
  pkg <- trimws(sub("\\(.*$", "", entries))

  # A constraint looks like "(>= 1.2.0)"; the operator may lack spaces.
  has_ver <- grepl("\\(", entries)
  constraint <- ifelse(has_ver, sub("^[^(]*\\(\\s*(.*?)\\s*\\)$", "\\1", entries), "")
  has_op <- has_ver & grepl("^[<>=]+", constraint)
  op <- ifelse(has_op, sub("^([<>=]+).*$", "\\1", constraint), NA_character_)
  ver <- ifelse(has_ver, trimws(sub("^[<>=]+", "", constraint)), NA_character_)
  out <- data.frame(
    type = rep(type, length(pkg)), package = pkg, op = op,
    version = ver, stringsAsFactors = FALSE
  )

  return(out)
}

#' @title Format a dependency data frame back into a field value
#' @param deps A data frame of dependencies as returned by [deps_parse()].
#' @return A single string suitable for a DESCRIPTION field.
#' @export
deps_format <- function(deps) {
  if (nrow(deps) == 0L) {
    return(NA_character_)
  }
  # R itself always comes first, the rest alphabetically regardless of case.
  deps <- deps[order(deps$package != "R", tolower(deps$package)), , drop = FALSE]
  entries <- ifelse(
    is.na(deps$version),
    deps$package,
    paste0(deps$package, " (", deps$op, " ", deps$version, ")")
  )

  return(paste0("\n    ", paste(entries, collapse = ",\n    ")))
}

#' @title List all dependencies declared in a DESCRIPTION
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return A data frame of dependencies as returned by [deps_parse()].
#' @export
deps_list <- function(path = ".") {
  types <- c("Depends", "Imports", "Suggests", "LinkingTo")
  parts <- vector("list", length(types))

  for (i in seq_along(types)) {
    raw <- desc_get_field(types[[i]], path = path)
    parts[[i]] <- deps_parse(raw, types[[i]])
  }

  out <- do.call(rbind, parts)
  rownames(out) <- NULL
  return(out)
}

#' @title Add or update a dependency in one DESCRIPTION field
#' @param package Name of the package to depend on.
#' @param type The DESCRIPTION field to modify.
#' @param min_version Optional minimum version, recorded as `>=`.
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return `TRUE` if the DESCRIPTION changed, invisibly.
#' @keywords internal
deps_add <- function(package, type, min_version = NULL, path = ".") {
  current <- deps_parse(desc_get_field(type, path = path), type)
  op <- if (is.null(min_version)) NA_character_ else ">="
  ver <- if (is.null(min_version)) NA_character_ else as.character(min_version)

  hit <- match(package, current$package)
  if (!is.na(hit)) {
    same <- identical(current$op[[hit]], op) && identical(current$version[[hit]], ver)
    # Without a new minimum version an existing entry is left alone.
    if (same || is.null(min_version)) {
      return(invisible(FALSE))
    }
    current$op[[hit]] <- op
    current$version[[hit]] <- ver
  } else {
    row <- data.frame(
      type = type, package = package, op = op,
      version = ver, stringsAsFactors = FALSE
    )
    current <- rbind(current, row)
  }

  desc_set_field(type, deps_format(current), path = path)
  return(invisible(TRUE))
}

#' @title Declare a package in Imports, Suggests or LinkingTo
#' @description Adds the package to the field, or raises its minimum version
#'   when the package is already listed with a lower one.
#' @param package Name of the package to depend on.
#' @param min_version Optional minimum version, recorded as `>=`.
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return `TRUE` if the DESCRIPTION changed, invisibly.
#' @export
use_import <- function(package, min_version = NULL, path = ".") {
  changed <- deps_add(package, "Imports", min_version = min_version, path = path)
  if (changed) {
    message("Added '", package, "' to Imports.")
  }
  return(invisible(changed))
}

#' @rdname use_import
#' @export
use_suggest <- function(package, min_version = NULL, path = ".") {
  changed <- deps_add(package, "Suggests", min_version = min_version, path = path)
  if (changed) {
    message("Added '", package, "' to Suggests.")
  }
  return(invisible(changed))
}

#' @rdname use_import
#' @export
use_linking_to <- function(package, min_version = NULL, path = ".") {
  changed <- deps_add(package, "LinkingTo", min_version = min_version, path = path)
  if (changed) {
    message("Added '", package, "' to LinkingTo.")
  }
  return(invisible(changed))
}

#' @title Find packages declared in more than one dependency field
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return A data frame with columns `package` and `types`, one row per
#'   duplicated package.
#' @export
deps_duplicates <- function(path = ".") {
  deps <- deps_list(path)
  deps <- deps[deps$package != "R", , drop = FALSE]
  counts <- table(deps$package)
  dupes <- names(counts)[counts > 1L]
  types <- character(length(dupes))

  for (i in seq_along(dupes)) {
    types[[i]] <- paste(deps$type[deps$package == dupes[[i]]], collapse = ", ")
  }

  return(data.frame(package = dupes, types = types, stringsAsFactors = FALSE))
}
