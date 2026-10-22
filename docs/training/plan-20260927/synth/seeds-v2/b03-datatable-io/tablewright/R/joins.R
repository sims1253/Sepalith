#' Look up attributes from a dimension table by key
#'
#' Performs an update join: columns from `dim` are added to `fact` by
#' reference, matching on `on`. Rows without a match get `NA`.
#'
#' @param fact A data.table of fact records to enrich.
#' @param dim A data.table of dimension records with a unique key.
#' @param on Character vector of join columns present in both tables.
#' @param cols Character vector of columns to copy from `dim`.
#' @return `fact`, modified by reference and returned invisibly.
#' @export
lookup_join <- function(fact, dim, on, cols) {
  check_has_cols(fact, on, "lookup_join")
  check_unique_key(dim, on, "lookup_join")
  check_has_cols(dim, cols, "lookup_join")

  # Refuse to overwrite: an update join would silently replace existing values
  clash <- intersect(cols, names(fact))
  if (length(clash) > 0L) {
    stop("Columns already exist in fact: ", paste(clash, collapse = ", "), call. = FALSE)
  }

  src <- paste0("i.", cols)
  fact[dim, (cols) := mget(src), on = on]
  return(invisible(fact))
}

#' Keep fact rows that have a match in a dimension table
#'
#' A semi join: only rows of `fact` whose key appears in `dim` are returned,
#' and no columns from `dim` are added.
#'
#' @inheritParams lookup_join
#' @export
semi_join_dt <- function(fact, dim, on) {
  check_has_cols(fact, on, "semi_join")
  check_has_cols(dim, on, "semi_join")
  keys <- unique(dim[, on, with = FALSE])
  # deduped keys first, so the join cannot multiply fact rows
  out <- fact[keys, on = on, nomatch = NULL]
  return(out)
}

#' Keep fact rows that have no match in a dimension table
#'
#' An anti join, the complement of [semi_join_dt()]. Useful for finding
#' orphaned records before a load.
#'
#' @inheritParams lookup_join
#' @return A data.table with a subset of the rows of `fact`.
#' @export
anti_join_dt <- function(fact, dim, on) {
  check_has_cols(fact, on, "anti_join")
  check_has_cols(dim, on, "anti_join")
  keys <- unique(dim[, on, with = FALSE])
  # `!` turns the join into "rows of fact with no match in keys"
  out <- fact[!keys, on = on]
  return(out)
}

#' Attach the most recent dimension row at or before each fact date
#'
#' Uses a rolling join so that slowly changing dimensions (prices, rates,
#' assignments) are matched to the version valid at the fact's timestamp.
#'
#' @param fact A data.table of fact records to enrich.
#' @param dim A data.table of dimension records with a unique key.
#' @param by Character vector of exact-match columns.
#' @param time_col Name of the date or time column present in both tables.
#' @param max_gap Optional maximum allowed distance, in units of `time_col`.
#' @return A new data.table with all rows of `fact` and the matched columns.
#' @export
rolling_join <- function(fact, dim, by, time_col, max_gap = NULL) {
  check_has_cols(fact, c(by, time_col), "rolling_join")
  check_has_cols(dim, c(by, time_col), "rolling_join")

  # Rolling on Date against POSIXct would compare days with seconds
  if (!identical(class(fact[[time_col]]), class(dim[[time_col]]))) {
    stop("`", time_col, "` must have the same class in both tables.", call. = FALSE)
  }

  roll <- if (is.null(max_gap)) TRUE else as.numeric(max_gap)
  dim <- data.table::copy(dim)
  # Keep the dimension's own timestamp; the join column takes the fact's value
  match_col <- paste0(time_col, "_matched")
  dim[, (match_col) := get(time_col)]
  data.table::setkeyv(dim, c(by, time_col))
  out <- dim[fact, on = c(by, time_col), roll = roll]
  data.table::setcolorder(out, names(fact))
  return(out)
}

#' Summarise how well two tables match on a key
#'
#' Reports counts of matched and unmatched keys on each side, which is a quick
#' sanity check before choosing a join type.
#'
#' @param x,y Data.tables to compare.
#' @param on Character vector of join columns present in both tables.
#' @return A one-row data.table with match counts and the match rate of `x`.
#' @export
join_coverage <- function(x, y, on) {
  check_has_cols(x, on, "join_coverage")
  check_has_cols(y, on, "join_coverage")

  kx <- unique(x[, on, with = FALSE])
  ky <- unique(y[, on, with = FALSE])
  # inner join of the distinct keys on both sides
  both <- nrow(kx[ky, on = on, nomatch = NULL])

  out <- data.table::data.table(
    keys_x = nrow(kx),
    keys_y = nrow(ky),
    matched = both,
    only_x = nrow(kx) - both,
    only_y = nrow(ky) - both
  )
  out[, match_rate_x := if (keys_x > 0L) matched / keys_x else NA_real_]
  return(out)
}

#' Stack several tables that share a key into one long table
#'
#' Tables are bound by column name; a `source` column records which element
#' of `tables` each row came from. Missing columns are filled with `NA`.
#'
#' @param tables A named list of data.tables.
#' @param key Character vector of columns that together form the key.
#' @param dedupe If `TRUE`, keep only the last row for each key.
#' @return A keyed data.table.
#' @export
stack_keyed <- function(tables, key, dedupe = FALSE) {
  if (is.null(names(tables)) || anyDuplicated(names(tables))) {
    stop("`tables` must be a list with unique names.", call. = FALSE)
  }
  for (nm in names(tables)) {
    check_has_cols(tables[[nm]], key, paste0("stack:", nm))
  }

  # idcol takes the list names, so `source` says where each row came from
  out <- data.table::rbindlist(tables, use.names = TRUE, fill = TRUE, idcol = "source")
  if (dedupe) {
    # rbindlist keeps list order, so the last row per key is the newest
    out <- out[, .SD[.N], by = key]
  }
  data.table::setkeyv(out, key)
  return(out)
}

#' Compare two versions of a keyed table
#'
#' Classifies every key as added, removed, changed or unchanged, comparing the
#' columns in `compare`. Numeric columns are compared with a tolerance.
#'
#' @param old,new Data.tables holding the previous and current versions.
#' @param on Character vector of join columns present in both tables.
#' @param compare Columns to compare; defaults to all shared non-key columns.
#' @param tolerance Absolute tolerance used for numeric columns.
#' @return A data.table with the key columns, a `status` column and, for
#'   changed rows, a `changed_cols` column listing what differs.
#' @export
reconcile_tables <- function(old, new, on, compare = NULL, tolerance = 1e-8) {
  check_unique_key(old, on, "reconcile:old")
  check_unique_key(new, on, "reconcile:new")
  if (is.null(compare)) {
    compare <- setdiff(intersect(names(old), names(new)), on)
  }
  check_has_cols(old, compare, "reconcile:old")
  check_has_cols(new, compare, "reconcile:new")

  merged <- merge(old, new, by = on, all = TRUE, suffixes = c(".old", ".new"))
  # flag which side each key came from; NA columns alone cannot tell
  merged[, `:=`(.in_old = FALSE, .in_new = FALSE)]
  merged[old, .in_old := TRUE, on = on]
  merged[new, .in_new := TRUE, on = on]

  diffs <- vector("list", length(compare))
  names(diffs) <- compare
  for (col in compare) {
    a <- merged[[paste0(col, ".old")]]
    b <- merged[[paste0(col, ".new")]]
    differs <- if (is.numeric(a) && is.numeric(b)) {
      abs(a - b) > tolerance
    } else {
      as.character(a) != as.character(b)
    }
    # NA on one side only is a change; NA on both sides is not
    differs[is.na(differs)] <- xor(is.na(a), is.na(b))[is.na(differs)]
    diffs[[col]] <- differs
  }

  changed <- rep("", nrow(merged))
  if (length(compare) > 0L && nrow(merged) > 0L) {
    diff_mat <- matrix(unlist(diffs), nrow = nrow(merged))
    changed <- apply(diff_mat, 1L, function(r) paste(compare[r], collapse = ","))
  }

  out <- merged[, on, with = FALSE]
  out[, status := data.table::fcase(
    !merged$.in_old, "added",
    !merged$.in_new, "removed",
    nzchar(changed), "changed",
    default = "unchanged"
  )]
  out[, changed_cols := ifelse(status == "changed", changed, NA_character_)]
  out[]
}
