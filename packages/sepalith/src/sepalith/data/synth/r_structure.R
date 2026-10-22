# Parse-structure dump for the SYNTH-v1 generator (parse only, never evaluates).
# Usage: Rscript --vanilla r_structure.R FILE...   -> one JSON object per line.
# Each object: file, ok, error, strings (multi-line string spans), comments
# (lines holding a comment token), units (top-level expressions with their
# braced blocks, statement starts and function formals).
js <- function(x) {
  if (is.null(x)) return("null")
  if (is.list(x)) {
    if (!is.null(names(x))) {
      parts <- vapply(names(x), function(n) paste0("\"", n, "\":", js(x[[n]])), "")
      return(paste0("{", paste(parts, collapse = ","), "}"))
    }
    return(paste0("[", paste(vapply(x, js, ""), collapse = ","), "]"))
  }
  if (is.character(x)) {
    x <- gsub("\\\\", "\\\\\\\\", x)
    x <- gsub("\"", "\\\\\"", x)
    x <- gsub("\n", "\\\\n", x)
    x <- gsub("\t", "\\\\t", x)
    v <- paste0("\"", x, "\"")
  } else if (is.logical(x)) {
    v <- ifelse(x, "true", "false")
  } else {
    v <- format(x, scientific = FALSE, trim = TRUE)
  }
  if (length(x) == 1L && is.null(attr(x, "array"))) v else paste0("[", paste(v, collapse = ","), "]")
}
arr <- function(x) structure(x, array = TRUE)

describe <- function(path) {
  exprs <- tryCatch(parse(path, keep.source = TRUE), error = function(e) e)
  if (inherits(exprs, "error")) {
    return(list(file = path, ok = FALSE, error = conditionMessage(exprs)))
  }
  pd <- getParseData(exprs, includeText = TRUE)
  if (is.null(pd) || nrow(pd) == 0L) {
    return(list(file = path, ok = TRUE, error = NULL, strings = list(), comments = arr(integer()), units = list()))
  }
  pd <- pd[order(pd$line1, pd$col1, -pd$line2, -pd$col2), ]
  kids <- split(seq_len(nrow(pd)), pd$parent)
  child_rows <- function(id) {
    k <- kids[[as.character(id)]]
    if (is.null(k)) integer() else k
  }
  str_rows <- which(pd$token == "STR_CONST" & pd$line2 > pd$line1)
  strings <- lapply(str_rows, function(i) arr(c(pd$line1[i], pd$line2[i])))
  comments <- arr(sort(unique(pd$line1[pd$token == "COMMENT"])))
  # all nodes below id (inclusive)
  subtree <- function(id) {
    out <- integer()
    stack <- id
    while (length(stack)) {
      cur <- stack[1L]
      stack <- stack[-1L]
      r <- which(pd$id == cur)
      out <- c(out, r)
      stack <- c(pd$id[child_rows(cur)], stack)
    }
    out
  }
  top <- which(pd$parent == 0L & !pd$terminal)
  units <- lapply(top, function(i) {
    rows <- subtree(pd$id[i])
    sub <- pd[rows, ]
    # braced blocks
    blocks <- list()
    for (r in rows) {
      if (pd$terminal[r]) next
      ch <- child_rows(pd$id[r])
      if (length(ch) && pd$token[ch[1L]] == "'{'") {
        close <- ch[pd$token[ch] == "'}'"]
        stm <- ch[!pd$terminal[ch]]
        blocks[[length(blocks) + 1L]] <- list(
          open_line = pd$line1[ch[1L]], open_col = pd$col1[ch[1L]],
          close_line = pd$line1[close[1L]], close_col = pd$col1[close[1L]],
          stmt_lines = arr(pd$line1[stm]), stmt_cols = arr(pd$col1[stm]),
          stmt_end_lines = arr(pd$line2[stm])
        )
      }
    }
    # functions and their formals
    funs <- list()
    for (r in rows) {
      if (pd$token[r] != "FUNCTION") next
      fexpr <- pd$parent[r]
      ch <- child_rows(fexpr)
      formals <- pd$text[ch[pd$token[ch] == "SYMBOL_FORMALS"]]
      body <- ch[!pd$terminal[ch]]
      body <- body[length(body)]
      fr <- which(pd$id == fexpr)
      funs[[length(funs) + 1L]] <- list(
        line = pd$line1[fr], end_line = pd$line2[fr],
        formals = arr(formals),
        body_line = if (length(body)) pd$line1[body] else -1L,
        body_braced = length(body) > 0L && pd$token[child_rows(pd$id[body])[1L]] == "'{'"
      )
    }
    # assignment target name for `name <- function` / `name = function`
    ch <- child_rows(pd$id[i])
    name <- ""
    kind <- "other"
    if (length(ch) >= 3L && pd$token[ch[2L]] %in% c("LEFT_ASSIGN", "EQ_ASSIGN")) {
      lhs <- child_rows(pd$id[ch[1L]])
      if (length(lhs) == 1L && pd$token[lhs] %in% c("SYMBOL", "STR_CONST")) name <- pd$text[lhs]
      rhs <- child_rows(pd$id[ch[3L]])
      if (length(rhs) && pd$token[rhs[1L]] == "FUNCTION") kind <- "assign_function"
      else kind <- "assign_other"
    } else if (length(ch) && !pd$terminal[ch[1L]]) {
      head <- child_rows(pd$id[ch[1L]])
      if (length(head) == 1L && pd$token[head] == "SYMBOL_FUNCTION_CALL") {
        kind <- "call"
        name <- pd$text[head]
      } else if (length(head) == 3L && pd$token[head[3L]] == "SYMBOL_FUNCTION_CALL") {
        kind <- "call"
        name <- paste0(pd$text[head[1L]], pd$text[head[2L]], pd$text[head[3L]])
      }
    }
    sym <- unique(sub$text[sub$token %in% c("SYMBOL", "SYMBOL_FUNCTION_CALL", "SYMBOL_FORMALS", "SYMBOL_SUB", "SLOT")])
    list(start = pd$line1[i], end = pd$line2[i], end_col = pd$col2[i], kind = kind, name = name,
         blocks = blocks, funs = funs, symbols = arr(sym))
  })
  list(file = path, ok = TRUE, error = NULL, strings = strings, comments = comments, units = units)
}

for (path in commandArgs(trailingOnly = TRUE)) {
  cat(js(describe(path)), "\n", sep = "")
}
