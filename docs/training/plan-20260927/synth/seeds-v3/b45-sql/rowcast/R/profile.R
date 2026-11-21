#' Profile one result column
#'
#' Computes the requested statistics; `min`, `max` and `mean` are `NA` when they
#' do not apply to the column type.
#'
#' @param x A result column.
#' @param stats Any of `"n"`, `"n_missing"`, `"n_distinct"`, `"min"`, `"max"` and `"mean"`.
#' @return A named list with one element per requested statistic.
#' @export
column.profile <- function(x, stats = c("n", "n_missing", "n_distinct", "min", "max", "mean")) {
  stats <- match.arg(stats, several.ok = TRUE)
  ordered <- is.numeric(x) || inherits(x, c("Date", "POSIXct")) || is.character(x)
  present <- x[!is.na(x)]
  out <- list()
  # Statistic `n`
  if ("n" %in% stats) {
    out$n <- length(x)
  }
  # Statistic `n_missing`
  if ("n_missing" %in% stats) {
    out$n_missing <- sum(is.na(x))
  }
  # Statistic `n_distinct`
  if ("n_distinct" %in% stats) {
    out$n_distinct <- length(unique(present))
  }
  # Statistic `min`
  if ("min" %in% stats) {
    out$min <- if (ordered && length(present) > 0) min(present) else NA
  }
  # Statistic `max`
  if ("max" %in% stats) {
    out$max <- if (ordered && length(present) > 0) max(present) else NA
  }
  # Statistic `mean`
  if ("mean" %in% stats) {
    out$mean <- if (is.numeric(x) && length(present) > 0) mean(present) else NA
  }
  out
}

#' Profile every column of a result set
#'
#' Classifies each column as `integer`, `numeric`, `logical`, `date`, `timestamp`
#' or `character` and summarises it with type-appropriate text.
#'
#' @param df A data frame of query results.
#' @return A data frame with columns `column`, `kind`, `n_missing`, `low`, `high`.
#' @export
result.profile <- function(df) {
  if (!is.data.frame(df)) {
    stop("`df` must be a data frame.", call. = FALSE)
  }
  rows <- vector("list", ncol(df))
  for (i in seq_along(df)) {
    x <- df[[i]]
    present <- x[!is.na(x)]
    kind <- if (is.integer(x)) {
      "integer"
    } else if (is.numeric(x)) {
      "numeric"
    } else if (is.logical(x)) {
      "logical"
    } else if (inherits(x, "Date")) {
      "date"
    } else if (inherits(x, "POSIXct")) {
      "timestamp"
    } else {
      "character"
    }
    has <- length(present) > 0
    range <- switch(kind,
      integer = {
        if (has) as.character(range(present)) else c(NA, NA)
      },
      numeric = {
        if (has) format(range(present), digits = 6) else c(NA, NA)
      },
      logical = {
        c(as.character(sum(!present)), as.character(sum(present)))
      },
      date = {
        if (has) format(range(present), "%Y-%m-%d") else c(NA, NA)
      },
      timestamp = {
        if (has) format(range(present), "%Y-%m-%d %H:%M:%S", tz = "UTC") else c(NA, NA)
      },
      character = {
        if (has) as.character(range(nchar(present))) else c(NA, NA)
      }
    )
    rows[[i]] <- data.frame(
      column = names(df)[i],
      kind = kind,
      n_missing = sum(is.na(x)),
      low = range[1],
      high = range[2]
    )
  }
  do.call(rbind, rows)
}

#' Compare two result sets by key
#'
#' Splits the rows into those only in `old`, only in `new`, and those whose
#' non-key values differ.
#'
#' @param old,new Data frames with the same columns.
#' @param key The name of the key column.
#' @return A list with elements `removed`, `added` and `changed`, each a data frame.
#' @export
compare.result.sets <- function(old, new, key) {
  if (!identical(sort(names(old)), sort(names(new)))) {
    stop("`old` and `new` must have the same columns.", call. = FALSE)
  }
  if (!key %in% names(old)) {
    stop(sprintf("`key` column `%s` is not in the result sets.", key), call. = FALSE)
  }
  new <- new[names(old)]
  # Rows only in `old`
  removed <- old[!old[[key]] %in% new[[key]], , drop = FALSE]
  rownames(removed) <- NULL
  # Rows only in `new`
  added <- new[!new[[key]] %in% old[[key]], , drop = FALSE]
  rownames(added) <- NULL
  # Rows in both with different values
  shared <- intersect(old[[key]], new[[key]])
  before <- old[match(shared, old[[key]]), , drop = FALSE]
  after <- new[match(shared, new[[key]]), , drop = FALSE]
  differs <- vapply(seq_along(shared), function(i) {
    !identical(as.list(before[i, ]), as.list(after[i, ]))
  }, logical(1))
  changed <- after[differs, , drop = FALSE]
  rownames(changed) <- NULL
  list(removed = removed, added = added, changed = changed)
}

#' Derive a table schema from a result set
#'
#' Maps the R class of each column to a column type in the target dialect.
#'
#' @param df A data frame of query results.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A data frame with columns `name`, `r_class` and `sql_type`.
#' @export
result.set.schema <- function(df, dialect = c("postgres", "mysql", "sqlite")) {
  dialect <- match.arg(dialect)
  out <- data.frame(name = names(df), r_class = character(ncol(df)),
                    sql_type = character(ncol(df)))
  for (i in seq_along(df)) {
    x <- df[[i]]
    r.class <- class(x)[1]
    sql.type <- switch(r.class,
      integer = {
        c(postgres = "INTEGER", mysql = "INT", sqlite = "INTEGER")[[dialect]]
      },
      numeric = {
        c(postgres = "DOUBLE PRECISION", mysql = "DOUBLE", sqlite = "REAL")[[dialect]]
      },
      logical = {
        c(postgres = "BOOLEAN", mysql = "TINYINT(1)", sqlite = "INTEGER")[[dialect]]
      },
      Date = {
        c(postgres = "DATE", mysql = "DATE", sqlite = "TEXT")[[dialect]]
      },
      POSIXct = {
        c(postgres = "TIMESTAMPTZ", mysql = "DATETIME", sqlite = "TEXT")[[dialect]]
      },
      {
        c(postgres = "TEXT", mysql = "TEXT", sqlite = "TEXT")[[dialect]]
      }
    )
    out$r_class[i] <- r.class
    out$sql_type[i] <- sql.type
  }
  out
}

#' Count result columns by kind
#'
#' Tallies how many columns of `df` fall in each kind and how many values are missing.
#'
#' @param df A data frame of query results.
#' @return A data frame with columns `kind`, `columns` and `missing`, one row per kind.
#' @export
summarize.kinds <- function(df) {
  kinds <- c("integer", "numeric", "logical", "date", "timestamp", "character")
  columns <- setNames(integer(length(kinds)), kinds)
  missing <- setNames(integer(length(kinds)), kinds)
  for (col in names(df)) {
    x <- df[[col]]
    kind <- if (is.integer(x)) {
      "integer"
    } else if (is.numeric(x)) {
      "numeric"
    } else if (is.logical(x)) {
      "logical"
    } else if (inherits(x, "Date")) {
      "date"
    } else if (inherits(x, "POSIXct")) {
      "timestamp"
    } else {
      "character"
    }
    columns[[kind]] <- columns[[kind]] + 1L
    missing[[kind]] <- missing[[kind]] + sum(is.na(x))
  }
  keep <- columns > 0L
  data.frame(kind = kinds[keep], columns = unname(columns[keep]), missing = unname(missing[keep]))
}
