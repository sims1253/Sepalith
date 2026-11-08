#' Profile the control-flow constructs of R code
#'
#' Counts each construct and how deeply it nests inside itself.
#' @param text A character vector of R source lines.
#' @return A data frame with one row per construct (`if`, `for`, `while`,
#'   `repeat`, `function`) and columns `construct`, `n`, `max_depth` and
#'   `lines`.
#' @export
construct_profile <- function(text) {
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  key <- as.character(pd$id)
  parent <- stats::setNames(as.character(pd$parent), key)
  span <- stats::setNames(pd$line2 - pd$line1 + 1L, key)
  # The node a keyword belongs to is the parent of the keyword token
  owner <- stats::setNames(rep("", nrow(pd)), key)
  is_keyword <- pd$token %in% c("IF", "FOR", "WHILE", "REPEAT", "FUNCTION")
  owner[as.character(pd$parent[is_keyword])] <- pd$token[is_keyword]
  nesting <- function(node, token) {
    depth <- 0L
    while (node %in% key) {
      if (owner[[node]] == token) depth <- depth + 1L
      node <- parent[[node]]
    }
    depth
  }

  # `if` statements
  nodes <- as.character(pd$parent[pd$token == "IF"])
  depth <- vapply(nodes, nesting, integer(1), token = "IF")
  if_row <- c(n = length(nodes), max_depth = max(0L, depth), lines = sum(span[nodes]))

  # `for` loops
  nodes <- as.character(pd$parent[pd$token == "FOR"])
  depth <- vapply(nodes, nesting, integer(1), token = "FOR")
  for_row <- c(n = length(nodes), max_depth = max(0L, depth), lines = sum(span[nodes]))

  # `while` loops
  nodes <- as.character(pd$parent[pd$token == "WHILE"])
  depth <- vapply(nodes, nesting, integer(1), token = "WHILE")
  while_row <- c(n = length(nodes), max_depth = max(0L, depth), lines = sum(span[nodes]))

  # `repeat` loops
  nodes <- as.character(pd$parent[pd$token == "REPEAT"])
  depth <- vapply(nodes, nesting, integer(1), token = "REPEAT")
  repeat_row <- c(n = length(nodes), max_depth = max(0L, depth), lines = sum(span[nodes]))

  # `function` definitions
  nodes <- as.character(pd$parent[pd$token == "FUNCTION"])
  depth <- vapply(nodes, nesting, integer(1), token = "FUNCTION")
  function_row <- c(n = length(nodes), max_depth = max(0L, depth), lines = sum(span[nodes]))

  rows <- rbind(if_row, for_row, while_row, repeat_row, function_row)
  data.frame(
    construct = c("if", "for", "while", "repeat", "function"),
    n = unname(rows[, "n"]), max_depth = unname(rows[, "max_depth"]),
    lines = unname(rows[, "lines"]), stringsAsFactors = FALSE
  )
}

#' Count the node types of parsed R code
#'
#' Walks the language objects recursively; no parse data is needed.
#' @param text A character vector of R source lines.
#' @return A named integer vector with the counts of `call`, `symbol`,
#'   `character`, `numeric`, `logical`, `null` and `missing` nodes.
#' @export
expr_kinds <- function(text) {
  counts <- c(call = 0L, symbol = 0L, character = 0L, numeric = 0L, logical = 0L,
    null = 0L, missing = 0L)
  is_empty <- function(x) is.symbol(x) && identical(as.character(x), "")

  walk <- function(node) {
    switch(typeof(node),
      language = {
        counts[["call"]] <<- counts[["call"]] + 1L
        parts <- as.list(node)
        for (i in seq_along(parts)) {
          if (is_empty(parts[[i]])) {
            counts[["missing"]] <<- counts[["missing"]] + 1L
          } else {
            walk(parts[[i]])
          }
        }
      },
      pairlist = {
        parts <- as.list(node)
        for (i in seq_along(parts)) {
          if (is_empty(parts[[i]])) {
            counts[["missing"]] <<- counts[["missing"]] + 1L
          } else {
            walk(parts[[i]])
          }
        }
      },
      symbol = counts[["symbol"]] <<- counts[["symbol"]] + 1L,
      character = counts[["character"]] <<- counts[["character"]] + 1L,
      double = ,
      integer = ,
      complex = counts[["numeric"]] <<- counts[["numeric"]] + 1L,
      logical = counts[["logical"]] <<- counts[["logical"]] + 1L,
      NULL = counts[["null"]] <<- counts[["null"]] + 1L
    )
    invisible(NULL)
  }

  for (e in as.list(parse(text = text, keep.source = FALSE))) walk(e)
  counts
}

#' Tabulate the functions defined in R code
#'
#' Finds every `name <- function(...)` definition and measures it. Counts
#' include any functions nested in the definition.
#' @param text A character vector of R source lines.
#' @return A data frame with one row per function and columns `name`,
#'   `line`, `formals`, `lines`, `calls` and `returns`.
#' @export
function_table <- function(text) {
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[order(pd$line1, pd$col1), , drop = FALSE]
  terminal <- pd[pd$terminal, , drop = FALSE]
  tok <- terminal$token
  prev <- c("", tok)[seq_along(tok)]
  prev2 <- c("", "", tok)[seq_along(tok)]
  defs <- which(tok == "FUNCTION" & prev %in% c("LEFT_ASSIGN", "EQ_ASSIGN") &
    prev2 == "SYMBOL")
  inside <- function(node) {
    after_start <- terminal$line1 > node$line1 |
      (terminal$line1 == node$line1 & terminal$col1 >= node$col1)
    before_end <- terminal$line2 < node$line2 |
      (terminal$line2 == node$line2 & terminal$col2 <= node$col2)
    after_start & before_end
  }

  n <- length(defs)
  name <- character(n)
  line <- formals <- lines <- calls <- returns <- integer(n)
  for (k in seq_len(n)) {
    node <- pd[pd$id == terminal$parent[defs[k]], , drop = FALSE]
    within <- terminal[inside(node), , drop = FALSE]
    # Name and first line
    name[k] <- gsub("^`|`$", "", terminal$text[defs[k] - 2L])
    line[k] <- node$line1
    # Formal arguments
    formals[k] <- sum(within$token == "SYMBOL_FORMALS")
    # Lines spanned
    lines[k] <- node$line2 - node$line1 + 1L
    # Calls made
    calls[k] <- sum(within$token == "SYMBOL_FUNCTION_CALL")
    # Explicit returns
    returns[k] <- sum(within$token == "SYMBOL_FUNCTION_CALL" & within$text == "return")
  }

  data.frame(
    name = name, line = line, formals = formals, lines = lines,
    calls = calls, returns = returns, stringsAsFactors = FALSE
  )
}

#' Build the call graph of the functions defined in R code
#'
#' Only calls between functions defined in `text` become edges.
#' @param text A character vector of R source lines.
#' @return A list with elements `nodes`, `edges` (a data frame with columns
#'   `from` and `to`), `roots`, `leaves` and `recursive`.
#' @export
call_graph <- function(text) {
  # 1. Find the defined functions
  # 2. Collect the calls inside each definition
  # 3. Keep calls to defined functions as edges
  # 4. Derive roots, leaves and self-recursive functions
  defs <- function_table(text)
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal & pd$token == "SYMBOL_FUNCTION_CALL", , drop = FALSE]

  edges <- data.frame(from = character(), to = character(), stringsAsFactors = FALSE)
  for (k in seq_len(nrow(defs))) {
    end <- defs$line[k] + defs$lines[k] - 1L
    in_body <- pd$line1 >= defs$line[k] & pd$line1 <= end
    called <- intersect(unique(pd$text[in_body]), defs$name)
    if (length(called) > 0L) {
      edges <- rbind(edges, data.frame(from = defs$name[k], to = called,
        stringsAsFactors = FALSE))
    }
  }

  roots <- setdiff(defs$name, edges$to)
  leaves <- setdiff(defs$name, edges$from[edges$from != edges$to])
  recursive <- edges$from[edges$from == edges$to]
  list(nodes = defs$name, edges = edges, roots = roots, leaves = leaves, recursive = recursive)
}

#' Classify the top-level expressions of R code
#'
#' Looks at the head of each top-level call to tell definitions, package
#' attachments, control flow and plain calls apart.
#' @param text A character vector of R source lines.
#' @return A data frame with columns `line`, `kind` and `name`, where `kind`
#'   is one of `function`, `assignment`, `attach`, `control`, `block`,
#'   `call` or `value`.
#' @export
top_level_kinds <- function(text) {
  exprs <- parse(text = text, keep.source = TRUE)
  line <- vapply(attr(exprs, "srcref"), function(s) as.integer(s[1L]), integer(1))
  n <- length(exprs)
  kind <- character(n)
  name <- character(n)

  for (i in seq_len(n)) {
    e <- exprs[[i]]
    if (!is.call(e)) {
      kind[i] <- "value"
      name[i] <- if (is.symbol(e)) as.character(e) else typeof(e)
      next
    }
    head <- paste(deparse(e[[1L]]), collapse = "")
    switch(head,
      "<-" = ,
      "<<-" = ,
      "=" = {
        is_fun <- is.call(e[[3L]]) && identical(e[[3L]][[1L]], as.name("function"))
        kind[i] <- if (is_fun) "function" else "assignment"
        name[i] <- paste(deparse(e[[2L]]), collapse = "")
      },
      library = ,
      require = ,
      requireNamespace = {
        kind[i] <- "attach"
        name[i] <- as.character(e[[2L]])
      },
      "if" = ,
      "for" = ,
      "while" = ,
      "repeat" = {
        kind[i] <- "control"
        name[i] <- head
      },
      "{" = {
        kind[i] <- "block"
        name[i] <- as.character(length(e) - 1L)
      },
      {
        kind[i] <- "call"
        name[i] <- head
      }
    )
  }

  data.frame(line = line, kind = kind, name = name, stringsAsFactors = FALSE)
}
