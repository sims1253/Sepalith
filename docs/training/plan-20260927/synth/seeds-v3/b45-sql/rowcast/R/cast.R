#' Cast one result column to an R type
#'
#' Converts the raw driver values of a column and warns about values that
#' could not be converted.
#'
#' @param x A vector of raw values, usually character.
#' @param type One of `"integer"`, `"numeric"`, `"logical"`, `"date"`, `"timestamp"`
#'   or `"character"`.
#' @param name The column name used in warnings.
#' @return A vector of the requested type, the same length as `x`.
#' @export
cast.column <- function(x, type = c("integer", "numeric", "logical", "date", "timestamp",
                                    "character"),
                        name = "x") {
  type <- match.arg(type)
  raw <- trimws(as.character(x))
  raw[raw == ""] <- NA
  values <- switch(type,
    integer = {
      out <- suppressWarnings(as.integer(raw))
      out
    },
    numeric = {
      out <- suppressWarnings(as.numeric(raw))
      out
    },
    logical = {
      out <- rep(NA, length(raw))
      out[tolower(raw) %in% c("t", "true", "1", "yes")] <- TRUE
      out[tolower(raw) %in% c("f", "false", "0", "no")] <- FALSE
      out
    },
    date = {
      out <- as.Date(substr(raw, 1L, 10L), format = "%Y-%m-%d")
      out
    },
    timestamp = {
      out <- as.POSIXct(substr(raw, 1L, 19L), format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
      out
    },
    character = {
      out <- raw
      out
    }
  )
  lost <- !is.na(raw) & is.na(values)
  if (any(lost)) {
    warning(sprintf("Column `%s`: %d value(s) could not be cast to %s.", name, sum(lost), type),
            call. = FALSE)
  }
  values
}

#' Cast every column of a result set
#'
#' Applies [cast.column()] to each column named in `types`; other columns are kept as is.
#'
#' @param df A data frame of raw query results.
#' @param types A named character vector mapping column names to types.
#' @return `df` with the listed columns converted.
#' @export
cast.result.set <- function(df, types) {
  if (!is.data.frame(df)) {
    stop("`df` must be a data frame.", call. = FALSE)
  }
  if (length(types) > 0 && is.null(names(types))) {
    stop("`types` must be a named character vector.", call. = FALSE)
  }
  unknown <- setdiff(names(types), names(df))
  if (length(unknown) > 0) {
    stop(sprintf("Unknown column(s) in `types`: %s.", toString(unknown)), call. = FALSE)
  }
  for (col in names(types)) {
    df[[col]] <- cast.column(df[[col]], types[[col]], name = col)
  }
  df
}

#' Normalise boolean values from a driver
#'
#' Each dialect returns booleans differently; map its spellings to `TRUE`/`FALSE`.
#'
#' @param x A vector of raw boolean values.
#' @param dialect One of `"postgres"`, `"mysql"`, `"sqlite"` or `"mssql"`.
#' @return A logical vector the same length as `x`.
#' @export
normalize.booleans <- function(x, dialect = c("postgres", "mysql", "sqlite", "mssql")) {
  dialect <- match.arg(dialect)
  if (is.logical(x)) {
    return(x)
  }
  raw <- tolower(trimws(as.character(x)))
  spellings <- switch(dialect,
    postgres = {
      list(true = c("t", "true"), false = c("f", "false"))
    },
    mysql = {
      list(true = c("1"), false = c("0"))
    },
    sqlite = {
      list(true = c("1", "true"), false = c("0", "false"))
    },
    mssql = {
      list(true = c("1", "true"), false = c("0", "false"))
    }
  )
  out <- rep(NA, length(raw))
  out[raw %in% spellings$true] <- TRUE
  out[raw %in% spellings$false] <- FALSE
  unknown <- !is.na(raw) & is.na(out)
  if (any(unknown)) {
    stop(sprintf("Unrecognised boolean value(s): %s.", toString(unique(raw[unknown]))),
         call. = FALSE)
  }
  out
}

#' Parse timestamps returned by a driver
#'
#' Strips fractional seconds and offsets in the dialect's text format, then parses as UTC.
#'
#' @param x A character vector of timestamps.
#' @param dialect One of `"postgres"`, `"mysql"`, `"sqlite"` or `"mssql"`.
#' @return A `POSIXct` vector in UTC.
#' @export
parse.db.timestamp <- function(x, dialect = c("postgres", "mysql", "sqlite", "mssql")) {
  dialect <- match.arg(dialect)
  raw <- trimws(as.character(x))
  raw <- switch(dialect,
    postgres = {
      # 2024-01-02 03:04:05.678+00
      raw <- sub("[+-][0-9]{2}(:?[0-9]{2})?$", "", raw)
      sub("\\.[0-9]+$", "", raw)
    },
    mysql = {
      # 2024-01-02 03:04:05.678
      sub("\\.[0-9]+$", "", raw)
    },
    sqlite = {
      # 2024-01-02T03:04:05.678Z
      raw <- sub("Z$", "", sub("T", " ", raw, fixed = TRUE))
      sub("\\.[0-9]+$", "", raw)
    },
    mssql = {
      # 2024-01-02 03:04:05.6780000
      sub("\\.[0-9]+$", "", raw)
    }
  )
  as.POSIXct(raw, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
}

#' Map database type names to cast types
#'
#' Classifies each declared database type into one of the types understood by
#' [cast.column()]: `integer`, `numeric`, `logical`, `date`, `timestamp`, `character`.
#'
#' @param db.types A named character vector of database type names, keyed by column.
#' @return A named character vector of cast types.
#' @export
cast.types.from.meta <- function(db.types) {
  upper <- toupper(db.types)
  out <- setNames(rep("character", length(upper)), names(db.types))
  # Type `integer`
  hits <- grepl("^(SMALLINT|INTEGER|INT|BIGINT|SERIAL|BIGSERIAL|TINYINT)\\b", upper)
  out[hits] <- "integer"
  # Type `numeric`
  hits <- grepl("^(NUMERIC|DECIMAL|REAL|DOUBLE|FLOAT)\\b", upper)
  out[hits] <- "numeric"
  # Type `logical`
  hits <- grepl("^(BOOLEAN|BOOL|BIT)\\b", upper) | upper == "TINYINT(1)"
  out[hits] <- "logical"
  # Type `date`
  hits <- grepl("^DATE$", upper)
  out[hits] <- "date"
  # Type `timestamp`
  hits <- grepl("^(TIMESTAMP|TIMESTAMPTZ|DATETIME|DATETIME2)\\b", upper)
  out[hits] <- "timestamp"
  out
}

#' Run a query and cast its result
#'
#' Fetches with DBI, then applies [cast.result.set()] with the given types.
#'
#' @param con A DBI connection.
#' @param sql A single SQL string.
#' @param types A named character vector mapping column names to types.
#' @param params Optional list of bind parameters.
#' @return A data frame with the listed columns converted.
#' @export
fetch.typed <- function(con, sql, types, params = NULL) {
  df <- if (is.null(params)) {
    DBI::dbGetQuery(con, sql)
  } else {
    DBI::dbGetQuery(con, sql, params = params)
  }
  missing <- setdiff(names(types), names(df))
  if (length(missing) > 0) {
    warning(sprintf("Column(s) not in result: %s.", toString(missing)), call. = FALSE)
    types <- types[setdiff(names(types), missing)]
  }
  cast.result.set(df, types)
}
