#' Read the parse data of an R source file
#'
#' Parses the file with source references kept and returns the token table
#' produced by [utils::getParseData()], with the file path attached.
#'
#' @param path Path to an R source file.
#' @return A data frame of parse data, or `NULL` if the file does not parse.
#' @export
readSourceTokens <- function(path) {
  exprs <- tryCatch(
    parse(path, keep.source = TRUE, encoding = "UTF-8"),
    error = function(e) {
      warning("Could not parse '", path, "': ", conditionMessage(e), call. = FALSE)
      NULL
    }
  )
  if (is.null(exprs)) {
    return(NULL)
  }
  pd <- utils::getParseData(exprs, includeText = TRUE)
  if (is.null(pd) || nrow(pd) == 0L) {
    return(NULL)
  }
  pd$file <- path
  pd[order(pd$line1, pd$col1), , drop = FALSE]
}

#' Get the direct children of a parse node
#'
#' @param pd A data frame of parse data as returned by [readSourceTokens()].
#' @param id The id of the parent node.
#' @return The rows of `pd` whose parent is `id`, in source order.
#' @family parse tree helpers
#' @keywords internal
tokenChildren <- function(pd, id) {
  kids <- pd[pd$parent == id, , drop = FALSE]
  kids[order(kids$line1, kids$col1), , drop = FALSE]
}

#' Get all descendants of a parse node
#'
#' @param pd A data frame of parse data as returned by [readSourceTokens()].
#' @param id The id of the ancestor node.
#' @return The rows of `pd` below `id` at any depth, in source order.
#' @family parse tree helpers
#' @keywords internal
tokenDescendants <- function(pd, id) {
  found <- integer()
  frontier <- id
  while (length(frontier) > 0L) {
    kids <- pd$id[pd$parent %in% frontier]
    found <- c(found, kids)
    frontier <- kids
  }
  out <- pd[pd$id %in% found, , drop = FALSE]
  out[order(out$line1, out$col1), , drop = FALSE]
}

#' Get the parent node of a parse node
#'
#' @param pd A data frame of parse data as returned by [readSourceTokens()].
#' @param id The id of the child node.
#' @return The single row of `pd` that is the parent of `id`, or `NULL` at the
#'   top level.
#' @family parse tree helpers
#' @keywords internal
tokenParent <- function(pd, id) {
  parentId <- pd$parent[pd$id == id]
  if (length(parentId) == 0L || parentId <= 0L) {
    return(NULL)
  }
  pd[pd$id == parentId, , drop = FALSE]
}

#' Find function definitions in parse data
#'
#' Locates every `function` keyword and resolves the name it is assigned to,
#' if any. Anonymous functions get the name `NA`.
#'
#' @param pd A data frame of parse data as returned by [readSourceTokens()].
#' @return A data frame with columns `name`, `id`, `line1` and `line2`, where
#'   `id` is the expression node of the function definition.
#' @export
findFunctionDefs <- function(pd) {
  keywords <- pd[pd$token == "FUNCTION", , drop = FALSE]
  n <- nrow(keywords)
  out <- data.frame(
    name = rep(NA_character_, n), id = integer(n),
    line1 = integer(n), line2 = integer(n), stringsAsFactors = FALSE
  )
  for (i in seq_len(n)) {
    fnExpr <- tokenParent(pd, keywords$id[[i]])
    out$id[[i]] <- fnExpr$id
    out$line1[[i]] <- fnExpr$line1
    out$line2[[i]] <- fnExpr$line2
    assign <- tokenParent(pd, fnExpr$id)
    if (is.null(assign)) {
      next
    }
    siblings <- tokenChildren(pd, assign$id)
    ops <- c("LEFT_ASSIGN", "EQ_ASSIGN")
    if (nrow(siblings) == 3L && siblings$token[[2L]] %in% ops) {
      target <- tokenDescendants(pd, siblings$id[[1L]])
      sym <- target$text[target$token %in% c("SYMBOL", "STR_CONST")]
      if (length(sym) == 1L) {
        out$name[[i]] <- gsub("^[\"'`]|[\"'`]$", "", sym)
      }
    }
  }
  out
}

#' List the formal arguments of a function definition
#'
#' @param pd A data frame of parse data as returned by [readSourceTokens()].
#' @param fnId The id of a function definition node from [findFunctionDefs()].
#' @return A data frame of the `SYMBOL_FORMALS` tokens.
#' @family function inspectors
#' @export
functionFormals <- function(pd, fnId) {
  kids <- tokenChildren(pd, fnId)
  kids[kids$token == "SYMBOL_FORMALS", c("text", "line1", "col1"), drop = FALSE]
}

#' List the symbols used in a function body
#'
#' @param pd A data frame of parse data as returned by [readSourceTokens()].
#' @param fnId The id of a function definition node from [findFunctionDefs()].
#' @return A character vector of distinct symbol names.
#' @family function inspectors
#' @export
functionBodySymbols <- function(pd, fnId) {
  kids <- tokenChildren(pd, fnId)
  body <- kids[kids$token == "expr", , drop = FALSE]
  if (nrow(body) == 0L) {
    return(character())
  }
  bodyId <- body$id[[nrow(body)]]
  inner <- tokenDescendants(pd, bodyId)
  syms <- inner$text[inner$token %in% c("SYMBOL", "SYMBOL_FUNCTION_CALL")]
  unique(syms)
}

#' Count the body lines of a function definition
#'
#' Blank lines and comment-only lines are excluded from the count.
#'
#' @param pd A data frame of parse data as returned by [readSourceTokens()].
#' @param fnId The id of a function definition node from [findFunctionDefs()].
#' @return A single integer.
#' @family function inspectors
#' @export
functionCodeLines <- function(pd, fnId) {
  inner <- tokenDescendants(pd, fnId)
  code <- inner[inner$terminal & inner$token != "COMMENT", , drop = FALSE]
  length(unique(code$line1))
}
