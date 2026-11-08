#' @title Does a value need quoting?
#' @param x A character vector.
#' @param style The quoting convention.
#' @param sep The field separator, used by `"csv"`.
#' @return A logical vector the same length as `x`; `NA` values never need
#'   quoting.
#' @keywords internal
needs.quoting <- function(x, style = c("sql", "csv", "shell", "regex"), sep = ",") {
  style <- match.arg(style)
  needs <- switch(style,
    sql = {
      rep(TRUE, length(x))
    },
    csv = {
      special <- paste0("[\"\r\n", sep, "]")
      grepl(special, x) | grepl("^ | $", x)
    },
    shell = {
      !grepl("^[[:alnum:]_./=:,+-]+$", x)
    },
    regex = {
      grepl("[][{}()+*^$|\\\\?.]", x)
    }
  )

  needs & !is.na(x)
}

#' @title Quote values
#' @description Wraps values in the quoting convention of `style` and escapes
#'   the quote character inside them. [dequote()] is the inverse.
#' @param x A character vector.
#' @param style The quoting convention: SQL string literals, CSV fields,
#'   POSIX shell words or literal regular expressions.
#' @param always Quote every element, not only those that need it? SQL
#'   literals are always quoted.
#' @return A character vector the same length as `x`. Missing values become
#'   `NULL` for `"sql"` and an empty string otherwise.
#' @export
enquote <- function(x, style = c("sql", "csv", "shell", "regex"), always = FALSE) {
  style <- match.arg(style)
  if (!is.character(x)) {
    stop("`x` must be a character vector.", call. = FALSE)
  }

  # Escape the quote character, then wrap the value in quotes
  needs <- !is.na(x) & (always | needs.quoting(x, style))
  out <- x
  switch(style,
    sql = {
      inner <- gsub("'", "''", x[needs], fixed = TRUE)
      out[needs] <- paste0("'", inner, "'")
      out[is.na(x)] <- "NULL"
    },
    csv = {
      inner <- gsub("\"", "\"\"", x[needs], fixed = TRUE)
      out[needs] <- paste0("\"", inner, "\"")
      out[is.na(x)] <- ""
    },
    shell = {
      inner <- gsub("'", "'\\''", x[needs], fixed = TRUE)
      out[needs] <- paste0("'", inner, "'")
      out[is.na(x)] <- ""
    },
    regex = {
      inner <- gsub("\\E", "\\E\\\\E\\Q", x[needs], fixed = TRUE)
      out[needs] <- paste0("\\Q", inner, "\\E")
      out[is.na(x)] <- ""
    }
  )

  out
}

#' @title Remove quoting from values
#' @description The inverse of [enquote()]. Values that are not quoted in
#'   the convention of `style` are returned unchanged.
#' @param x A character vector.
#' @param style The quoting convention: SQL string literals, CSV fields,
#'   POSIX shell words or literal regular expressions.
#' @return A character vector the same length as `x`. `NULL` for `"sql"` and
#'   empty strings otherwise become `NA`.
#' @export
dequote <- function(x, style = c("sql", "csv", "shell", "regex")) {
  style <- match.arg(style)
  if (!is.character(x)) {
    stop("`x` must be a character vector.", call. = FALSE)
  }

  # Strip the wrapping quotes, then unescape the quote character
  out <- x
  switch(style,
    sql = {
      quoted <- grepl("^'.*'$", x)
      inner <- sub("^'(.*)'$", "\\1", x[quoted])
      out[quoted] <- gsub("''", "'", inner, fixed = TRUE)
      out[x %in% "NULL"] <- NA_character_
    },
    csv = {
      quoted <- grepl("^\".*\"$", x)
      inner <- sub("^\"(.*)\"$", "\\1", x[quoted])
      out[quoted] <- gsub("\"\"", "\"", inner, fixed = TRUE)
      out[x %in% ""] <- NA_character_
    },
    shell = {
      quoted <- grepl("^'.*'$", x)
      inner <- sub("^'(.*)'$", "\\1", x[quoted])
      out[quoted] <- gsub("'\\''", "'", inner, fixed = TRUE)
      out[x %in% ""] <- NA_character_
    },
    regex = {
      quoted <- grepl("^\\\\Q.*\\\\E$", x)
      inner <- sub("^\\\\Q(.*)\\\\E$", "\\1", x[quoted])
      out[quoted] <- gsub("\\E\\\\E\\Q", "\\E", inner, fixed = TRUE)
      out[x %in% ""] <- NA_character_
    }
  )

  out
}

#' @title Split a delimited line, honouring quotes
#' @description A character-level scanner: separators inside quotes do not
#'   split, and a doubled quote inside quotes stands for one quote character.
#' @param line A single string.
#' @param sep The field separator, a single character.
#' @param quote The quote character, a single character.
#' @return A character vector of unquoted fields.
#' @export
quoted.split <- function(line, sep = ",", quote = "\"") {
  stopifnot(nchar(sep) == 1L, nchar(quote) == 1L)
  chars <- strsplit(line, "", fixed = TRUE)[[1]]
  fields <- character()
  current <- character()
  in.quotes <- FALSE
  i <- 1L

  while (i <= length(chars)) {
    ch <- chars[i]
    if (in.quotes) {
      if (ch == quote && i < length(chars) && chars[i + 1L] == quote) {
        # Doubled quote: keep one, skip the other
        current <- c(current, quote)
        i <- i + 1L
      } else if (ch == quote) {
        in.quotes <- FALSE
      } else {
        current <- c(current, ch)
      }
    } else {
      if (ch == quote) {
        in.quotes <- TRUE
      } else if (ch == sep) {
        fields <- c(fields, paste(current, collapse = ""))
        current <- character()
      } else {
        current <- c(current, ch)
      }
    }
    i <- i + 1L
  }

  if (in.quotes) {
    warning("Unterminated quote in `line`.", call. = FALSE)
  }
  c(fields, paste(current, collapse = ""))
}

#' @title Join fields into a delimited line
#' @param fields A character vector of field values.
#' @param sep The field separator.
#' @return A single string, with fields quoted where [needs.quoting()] says so.
#' @export
quoted.join <- function(fields, sep = ",") {
  needs <- needs.quoting(fields, "csv", sep = sep)
  quoted <- fields
  quoted[needs] <- paste0("\"", gsub("\"", "\"\"", fields[needs], fixed = TRUE), "\"")
  quoted[is.na(fields)] <- ""
  paste(quoted, collapse = sep)
}
