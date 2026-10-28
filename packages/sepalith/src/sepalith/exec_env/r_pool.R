# Static analysis of top-level R functions for the exec-env function pool.
# Usage: Rscript --vanilla r_pool.R LISTFILE OUT.jsonl
# LISTFILE has one R file path per line. Never evaluates package code: the only
# evaluation is of the `function(...)` expression itself (which creates a closure
# without running its body or default arguments) so codetools can list globals.
# One JSON line per top-level `name <- function(...)` / `name = function(...)`;
# files that do not parse get one line with "file_error".
suppressPackageStartupMessages(library(codetools))
args <- commandArgs(trailingOnly = TRUE)
files <- readLines(args[[1]], warn = FALSE)
out <- file(args[[2]], open = "w", encoding = "UTF-8")

esc <- function(x) {
  x <- gsub("\\\\", "\\\\\\\\", x)
  x <- gsub('"', '\\\\"', x)
  x <- gsub("[[:cntrl:]]", " ", x)
  x
}
js <- function(x) {
  if (is.list(x)) {
    if (length(x) == 0) return(if (is.null(names(x))) "[]" else "{}")
    if (!is.null(names(x))) {
      return(paste0("{", paste0('"', esc(names(x)), '":', vapply(x, js, ""), collapse = ","), "}"))
    }
    return(paste0("[", paste(vapply(x, js, ""), collapse = ","), "]"))
  }
  if (length(x) == 0) return("[]")
  if (is.logical(x) && length(x) == 1) return(if (isTRUE(x)) "true" else "false")
  if (is.numeric(x) && length(x) == 1) return(format(x, scientific = FALSE))
  if (is.character(x) && length(x) == 1 && !is.null(attr(x, "scalar"))) return(paste0('"', esc(x), '"'))
  paste0("[", paste0('"', esc(as.character(x)), '"', collapse = ","), "]")
}
scalar <- function(x) structure(as.character(x), scalar = TRUE)

search_pkgs <- setdiff(search(), c(".GlobalEnv", "Autoloads"))
where_base <- function(g) {
  for (s in search_pkgs) if (exists(g, envir = as.environment(s), inherits = FALSE)) return(sub("^package:", "", s))
  NA_character_
}
allowed_ns <- rownames(installed.packages(priority = c("base", "recommended")))
ns_exists <- function(pkg, fn) {
  if (!pkg %in% allowed_ns) return(FALSE)
  ok <- tryCatch(exists(fn, envir = asNamespace(pkg), inherits = FALSE), error = function(e) FALSE)
  isTRUE(ok)
}

analyze_file <- function(f) {
  exprs <- tryCatch(parse(f, keep.source = TRUE, encoding = "UTF-8"), error = function(e) e)
  if (inherits(exprs, "error")) {
    writeLines(paste0('{"file":"', esc(f), '","file_error":"parse"}'), out)
    return(invisible())
  }
  pd <- getParseData(exprs, includeText = TRUE)
  srcrefs <- attr(exprs, "srcref")
  lines <- readLines(f, warn = FALSE, encoding = "UTF-8")
  top <- lapply(seq_along(exprs), function(i) {
    e <- exprs[[i]]
    nm <- if (is.call(e) && as.character(e[[1]])[1] %in% c("<-", "=") && is.name(e[[2]])) as.character(e[[2]]) else NA_character_
    isfun <- !is.na(nm) && is.call(e[[3]]) && identical(as.character(e[[3]][[1]])[1], "function")
    list(name = nm, fun = isfun, l1 = srcrefs[[i]][1], l2 = srcrefs[[i]][3])
  })
  top_funs <- unique(na.omit(vapply(top, function(t) if (t$fun) t$name else NA_character_, "")))
  top_vars <- unique(na.omit(vapply(top, function(t) if (!t$fun) t$name else NA_character_, "")))
  for (i in seq_along(exprs)) {
    t <- top[[i]]
    if (!t$fun) next
    rec <- tryCatch({
      e <- exprs[[i]]
      tok <- pd[pd$terminal & pd$line1 >= t$l1 & pd$line2 <= t$l2, ]
      fun_tok <- tok[tok$token == "FUNCTION", ][1, ]
      kids <- pd[pd$parent == fun_tok$parent & pd$token == "expr", ]
      body <- kids[nrow(kids), ]
      first <- pd[pd$terminal & pd$line1 == body$line1 & pd$col1 == body$col1, ]
      braced <- nrow(first) > 0 && first$token[1] == "'{'"
      inner <- if (braced && body$line2 - body$line1 >= 2) lines[(body$line1 + 1):(body$line2 - 1)] else character()
      nonblank <- sum(nzchar(trimws(inner)))
      code <- sum(nzchar(trimws(inner)) & !grepl("^\\s*#", inner))
      stmts <- if (braced) {
        brace_id <- body$id
        nrow(pd[pd$parent == brace_id & pd$token %in% c("expr", "expr_or_assign_or_help", "equal_assign"), ])
      } else 0
      fobj <- eval(e[[3]], envir = new.env(parent = baseenv()))
      g <- findGlobals(fobj, merge = FALSE)
      classify <- function(names) {
        lapply(setNames(as.list(names), names), function(n) {
          if (n %in% top_funs) return(scalar("file_fun"))
          if (n %in% top_vars) return(scalar("file_var"))
          w <- where_base(n)
          if (!is.na(w)) return(scalar(paste0("pkg:", w)))
          scalar("unresolved")
        })
      }
      pkgs <- which(tok$token == "SYMBOL_PACKAGE")
      ns <- character()
      for (k in pkgs) {
        nxt <- tok[tok$line1 == tok$line1[k] & tok$col1 > tok$col1[k] & tok$token %in% c("SYMBOL_FUNCTION_CALL", "SYMBOL"), ]
        nxt <- nxt[order(nxt$col1), ]
        op <- tok[tok$line1 == tok$line1[k] & tok$col1 > tok$col1[k] & tok$token %in% c("NS_GET", "NS_GET_INT"), ]
        op <- op[order(op$col1), ]
        if (nrow(nxt)) {
          kind <- if (nrow(op) && op$token[1] == "NS_GET_INT") ":::" else "::"
          ns <- c(ns, paste0(tok$text[k], kind, nxt$text[1], if (kind == "::" && ns_exists(tok$text[k], nxt$text[1])) "" else "!"))
        }
      }
      calls <- unique(tok$text[tok$token == "SYMBOL_FUNCTION_CALL"])
      list(file = scalar(f), name = scalar(t$name), start = t$l1, end = t$l2,
           body_line1 = body$line1, body_line2 = body$line2, braced = braced,
           nonblank = nonblank, code = code, stmts = stmts,
           formals = names(formals(fobj)) %||% character(),
           fun_globals = classify(g$functions), var_globals = classify(g$variables),
           ns = unique(ns), calls = calls,
           super_assign = any(tok$text == "<<-"),
           loops = sum(tok$token %in% c("FOR", "WHILE", "REPEAT")),
           ifs = sum(tok$token == "IF"), returns = sum(tok$text == "return" & tok$token == "SYMBOL_FUNCTION_CALL"),
           nested_funs = sum(tok$token == "FUNCTION") - 1L,
           strings = sum(tok$token == "STR_CONST"), numbers = sum(tok$token == "NUM_CONST"))
    }, error = function(err) list(file = scalar(f), name = scalar(t$name), error = scalar(conditionMessage(err))))
    writeLines(js(rec), out, useBytes = TRUE)
  }
}
for (f in files) {
  r <- tryCatch(analyze_file(f), error = function(e) e)
  if (inherits(r, "error")) writeLines(paste0('{"file":"', esc(f), '","file_error":"analysis"}'), out)
}
close(out)
