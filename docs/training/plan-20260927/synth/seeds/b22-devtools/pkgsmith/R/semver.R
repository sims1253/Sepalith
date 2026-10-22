#' Parse a semantic version string
#'
#' Accepts `MAJOR.MINOR.PATCH` with an optional fourth development component
#' (`1.2.3.9000`) or a `-prerelease` suffix (`1.2.3-rc.1`).
#'
#' @param x A single version string.
#' @return A list with integer elements `major`, `minor`, `patch`, `dev` and
#'   a character element `pre`.
#' @family semver
#' @export
semver_parse = function(x) {
  x = trimws(x)
  pre = NA_character_
  if (grepl("-", x, fixed = TRUE)) {
    pre = sub("^[^-]*-", "", x)
    x = sub("-.*$", "", x)
  }
  parts = strsplit(x, "[.]")[[1L]]
  if (length(parts) < 2L || length(parts) > 4L || any(!grepl("^[0-9]+$", parts))) {
    stop("'", x, "' is not a valid version string.", call. = FALSE)
  }
  nums = as.integer(parts)
  length(nums) = 4L
  nums[is.na(nums)] = 0L
  list(major = nums[[1L]], minor = nums[[2L]], patch = nums[[3L]], dev = nums[[4L]], pre = pre)
}

#' Format a parsed semantic version
#'
#' @param v A parsed version as returned by [semver_parse()].
#' @return A single version string.
#' @family semver
#' @export
semver_format = function(v) {
  out = paste(v$major, v$minor, v$patch, sep = ".")
  if (v$dev > 0L) {
    out = paste(out, v$dev, sep = ".")
  }
  if (!is.na(v$pre)) {
    out = paste(out, v$pre, sep = "-")
  }
  out
}

#' Compare two semantic versions
#'
#' Pre-release versions sort before their release; development versions sort
#' after it.
#'
#' @param a,b Version strings to compare.
#' @return `-1L`, `0L` or `1L`.
#' @family semver
#' @export
semver_compare = function(a, b) {
  va = semver_parse(a)
  vb = semver_parse(b)
  for (field in c("major", "minor", "patch")) {
    if (va[[field]] != vb[[field]]) {
      return(if (va[[field]] < vb[[field]]) -1L else 1L)
    }
  }
  pre_a = is.na(va$pre)
  pre_b = is.na(vb$pre)
  if (pre_a != pre_b) {
    return(if (pre_a) 1L else -1L)
  }
  if (!pre_a && va$pre != vb$pre) {
    return(if (va$pre < vb$pre) -1L else 1L)
  }
  if (va$dev != vb$dev) {
    return(if (va$dev < vb$dev) -1L else 1L)
  }
  0L
}

#' Bump the major component of a version
#'
#' @param x A single version string.
#' @return The bumped version string.
#' @family version bumps
#' @export
bump_major = function(x) {
  v = semver_parse(x)
  v$major = v$major + 1L
  v$minor = 0L
  v$patch = 0L
  v$dev = 0L
  v$pre = NA_character_
  semver_format(v)
}

#' Bump the minor component of a version
#'
#' @param x A single version string.
#' @return The bumped version string.
#' @family version bumps
#' @export
bump_minor = function(x) {
  v = semver_parse(x)
  v$minor = v$minor + 1L
  v$patch = 0L
  v$dev = 0L
  v$pre = NA_character_
  semver_format(v)
}

#' Bump the patch component of a version
#'
#' @param x A single version string.
#' @return The bumped version string.
#' @family version bumps
#' @export
bump_patch = function(x) {
  v = semver_parse(x)
  v$patch = v$patch + 1L
  v$dev = 0L
  v$pre = NA_character_
  semver_format(v)
}

#' Bump the development component of a version
#'
#' Starts the development series at `9000` when the version has none.
#'
#' @param x A single version string.
#' @return The bumped version string.
#' @family version bumps
#' @export
bump_dev = function(x) {
  v = semver_parse(x)
  v$dev = if (v$dev == 0L) 9000L else v$dev + 1L
  v$pre = NA_character_
  semver_format(v)
}

#' Bump the version recorded in DESCRIPTION
#'
#' @param which Which component to bump.
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return The new version string, invisibly.
#' @export
use_version = function(which = c("patch", "minor", "major", "dev"), path = ".") {
  which = match.arg(which)
  old = desc_get_field("Version", path = path)
  if (is.na(old)) {
    stop("DESCRIPTION has no Version field.", call. = FALSE)
  }
  new = switch(which,
    patch = bump_patch(old),
    minor = bump_minor(old),
    major = bump_major(old),
    dev = bump_dev(old)
  )
  desc_set_field("Version", new, path = path)
  message("Version: ", old, " -> ", new)
  invisible(new)
}
