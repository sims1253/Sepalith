#' @title Diff two versions of R code
#' @description Aligns the lines of `before` and `after` along their longest
#'   common subsequence; deletions come before insertions.
#' @param before,after Character vectors of source lines.
#' @return A data frame with columns `op` (`" "`, `"-"` or `"+"`), `old`
#'   and `new` (line numbers, `NA` where absent) and `text`.
#' @export
style.diff <- function(before, after) {
  n <- length(before)
  m <- length(after)
  # Lengths of the longest common subsequences of all suffixes
  lcs <- matrix(0L, n + 1L, m + 1L)
  for (i in rev(seq_len(n))) {
    for (j in rev(seq_len(m))) {
      if (before[i] == after[j]) {
        lcs[i, j] <- lcs[i + 1L, j + 1L] + 1L
      } else {
        lcs[i, j] <- max(lcs[i + 1L, j], lcs[i, j + 1L])
      }
    }
  }

  op <- text <- character()
  old <- new <- integer()
  i <- 1L
  j <- 1L
  while (i <= n || j <= m) {
    if (i <= n && j <= m && before[i] == after[j]) {
      op <- c(op, " ")
      text <- c(text, before[i])
      old <- c(old, i)
      new <- c(new, j)
      i <- i + 1L
      j <- j + 1L
    } else if (i <= n && (j > m || lcs[i + 1L, j] >= lcs[i, j + 1L])) {
      op <- c(op, "-")
      text <- c(text, before[i])
      old <- c(old, i)
      new <- c(new, NA_integer_)
      i <- i + 1L
    } else {
      op <- c(op, "+")
      text <- c(text, after[j])
      old <- c(old, NA_integer_)
      new <- c(new, j)
      j <- j + 1L
    }
  }
  data.frame(op = op, old = old, new = new, text = text, stringsAsFactors = FALSE)
}

#' @title Render a diff of two versions of R code
#' @description Shows the result of [style.diff()] in one of four layouts.
#' @inheritParams style.diff
#' @param format The layout: `unified` (every line with its marker), `side`
#'   (old and new next to each other), `inline` (changed lines with their
#'   line numbers) or `stat` (a one-line count).
#' @return A character vector of output lines.
#' @export
render.diff <- function(before, after, format = c("unified", "side", "inline", "stat")) {
  format <- match.arg(format)
  diff <- style.diff(before, after)
  changed <- diff$op != " "

  switch(format,
    unified = {
      body <- paste0(diff$op, diff$text)
      c("--- before", "+++ after", body)
    },
    side = {
      left <- ifelse(diff$op == "+", "", diff$text)
      right <- ifelse(diff$op == "-", "", diff$text)
      width <- max(0L, nchar(left))
      sprintf("%-*s %s %s", width, left, ifelse(changed, diff$op, "|"), right)
    },
    inline = {
      line <- ifelse(diff$op == "+", diff$new, diff$old)
      body <- sprintf("%4d %s %s", line[changed], diff$op[changed], diff$text[changed])
      if (any(changed)) body else "no changes"
    },
    stat = {
      added <- sum(diff$op == "+")
      removed <- sum(diff$op == "-")
      sprintf("%d insertions(+), %d deletions(-)", added, removed)
    }
  )
}

#' @title Summarise the layout of R code before and after a fix
#' @description Measures the same layout properties of both versions.
#' @inheritParams style.diff
#' @param width The line width above which a line counts as long.
#' @return A data frame with columns `measure` (`lines`, `blank`,
#'   `trailing`, `long`, `semicolons` and `chars`), `before`, `after` and
#'   `delta`.
#' @export
change.summary <- function(before, after, width = 80L) {
  measures <- c("lines", "blank", "trailing", "long", "semicolons", "chars")
  old <- stats::setNames(integer(length(measures)), measures)
  new <- old

  # Lines
  old[["lines"]] <- length(before)
  new[["lines"]] <- length(after)
  # Blank lines
  old[["blank"]] <- sum(!nzchar(trimws(before)))
  new[["blank"]] <- sum(!nzchar(trimws(after)))
  # Lines with trailing white space
  old[["trailing"]] <- sum(grepl("[ ]+$", before))
  new[["trailing"]] <- sum(grepl("[ ]+$", after))
  # Lines wider than `width`
  old[["long"]] <- sum(nchar(before) > width)
  new[["long"]] <- sum(nchar(after) > width)
  # Lines with a semicolon outside of comments
  old[["semicolons"]] <- sum(grepl(";", sub("#.*$", "", before), fixed = TRUE))
  new[["semicolons"]] <- sum(grepl(";", sub("#.*$", "", after), fixed = TRUE))
  # Characters
  old[["chars"]] <- sum(nchar(before))
  new[["chars"]] <- sum(nchar(after))

  data.frame(
    measure = measures, before = unname(old), after = unname(new),
    delta = unname(new - old), stringsAsFactors = FALSE
  )
}

#' @title Check that a fixer is safe
#' @description Runs a fixer on R code and checks the properties every fixer
#'   should have.
#' @param fixer A function that takes source lines as its first argument and
#'   returns a list with an `after` element.
#' @param text A character vector of R source lines.
#' @param ... Passed to `fixer`.
#' @return A list with elements `parses`, `same.code` (the parsed code is
#'   unchanged), `idempotent` (a second run changes nothing) and `changed`
#'   (the number of diff lines).
#' @export
check.fixer <- function(fixer, text, ...) {
  # 1. Fix the code once and check that the result still parses
  once <- fixer(text, ...)$after
  parsed <- tryCatch(parse(text = once, keep.source = FALSE), error = function(e) NULL)
  parses <- !is.null(parsed)
  # 2. Compare the code before and after, ignoring the layout
  original <- parse(text = text, keep.source = FALSE)
  same.code <- parses && identical(lapply(original, deparse), lapply(parsed, deparse))
  # 3. Fix the result again: the second run must change nothing
  idempotent <- parses && identical(fixer(once, ...)$after, once)
  # 4. Count the diff lines
  changed <- sum(style.diff(text, once)$op != " ")
  list(parses = parses, same.code = same.code, idempotent = idempotent, changed = changed)
}

#' @title Run a pair of opposite fixers
#' @description Applies `forward` and then `backward`, which should undo it,
#'   and reports what the round trip does not restore.
#' @param text A character vector of R source lines.
#' @param forward,backward Fixers that take source lines and return a list
#'   with an `after` element.
#' @return A list with elements `forward` and `backward` (the lines after
#'   each pass), `edits.forward` and `edits.backward` (diff line counts),
#'   `restored` and `lost` (lines of `text` missing after the round trip).
#' @export
roundtrip.fixer <- function(text, forward, backward) {
  # Forward pass
  there <- forward(text)$after
  diff.there <- style.diff(text, there)
  # Backward pass
  back <- backward(there)$after
  diff.back <- style.diff(there, back)

  lost <- style.diff(text, back)
  list(
    forward = there, backward = back,
    edits.forward = sum(diff.there$op != " "), edits.backward = sum(diff.back$op != " "),
    restored = identical(back, text), lost = lost$old[lost$op == "-"]
  )
}
