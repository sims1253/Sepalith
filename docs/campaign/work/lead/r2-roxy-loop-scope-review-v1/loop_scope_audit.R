#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: loop_scope_audit.R INPUT.json OUTPUT.jsonl")
input <- jsonlite::fromJSON(args[[1L]], simplifyVector = FALSE)

call_head <- function(node) {
  if (!is.call(node) || !is.symbol(node[[1L]])) return(NA_character_)
  as.character(node[[1L]])
}

find_targets <- function(expressions, target_name) {
  result <- list()
  for (node in as.list(expressions)) {
    if (!is.call(node) || !(call_head(node) %in% c("<-", "=", "<<-")) || length(node) != 3L) next
    if (is.symbol(node[[2L]]) && identical(as.character(node[[2L]]), target_name) &&
        is.call(node[[3L]]) && identical(call_head(node[[3L]]), "function")) result[[length(result) + 1L]] <- node[[3L]]
  }
  result
}

binding_hash <- function(node) digest::digest(serialize(node, NULL, ascii = TRUE), algo = "sha256", serialize = FALSE)

audit_target <- function(fun, residual_names) {
  occurrences <- setNames(lapply(residual_names, function(x) list()), residual_names)
  record <- function(name, classification, binding_type = NA_character_, binding_sha256 = NA_character_) {
    occurrences[[name]][[length(occurrences[[name]]) + 1L]] <<- list(
      classification = classification, binding_type = binding_type, binding_sha256 = binding_sha256)
  }
  visit <- function(node, bindings = list(), parent_head = NA_character_, parent_index = NA_integer_) {
    if (is.symbol(node)) {
      name <- as.character(node)
      if (!(name %in% residual_names)) return(invisible(NULL))
      if (identical(parent_head, "$") && identical(parent_index, 3L)) {
        record(name, "member_literal_not_lexical")
      } else if (!is.null(bindings[[name]])) {
        record(name, "lexically_bound_occurrence", bindings[[name]]$type, bindings[[name]]$sha256)
      } else {
        record(name, "unbound_or_unsupported_occurrence")
      }
      return(invisible(NULL))
    }
    if (!is.call(node)) return(invisible(NULL))
    head <- call_head(node)
    if (identical(head, "function") && length(node) >= 3L) {
      local <- bindings
      formal_names <- names(node[[2L]])
      for (name in formal_names) if (nzchar(name)) local[[name]] <- list(type = "function_formal", sha256 = binding_hash(node[[2L]]))
      visit(node[[3L]], local, "function", 3L)
      return(invisible(NULL))
    }
    if (identical(head, "for") && length(node) == 4L && is.symbol(node[[2L]])) {
      binder <- as.character(node[[2L]])
      if (binder %in% residual_names) record(binder, "binding_definition", "for_loop_variable", binding_hash(node))
      visit(node[[3L]], bindings, "for", 3L)
      local <- bindings
      local[[binder]] <- list(type = "for_loop_variable", sha256 = binding_hash(node))
      visit(node[[4L]], local, "for", 4L)
      return(invisible(NULL))
    }
    if (length(node) >= 2L) for (index in 2:length(node)) {
      if (!identical(node[[index]], quote(expr = ))) visit(node[[index]], bindings, head, as.integer(index))
    }
  }
  # Root formals are bindings too, although prior residual construction should
  # already have removed them. This makes the lexical rule explicit.
  root_bindings <- list()
  for (name in names(fun[[2L]])) if (nzchar(name)) root_bindings[[name]] <- list(type = "root_function_formal", sha256 = binding_hash(fun[[2L]]))
  visit(fun[[3L]], root_bindings, "function", 3L)
  globals <- tryCatch(codetools::findGlobals(eval(fun, envir = baseenv()), merge = FALSE)$variables, error = function(e) character())
  supported <- c("binding_definition", "lexically_bound_occurrence")
  by_name <- list()
  for (name in residual_names) {
    classes <- vapply(occurrences[[name]], function(x) x$classification, character(1))
    all_structurally_bound <- length(classes) > 0L && all(classes %in% supported) &&
      any(classes == "lexically_bound_occurrence") && !(name %in% globals)
    by_name[[name]] <- list(
      occurrences = length(classes), class_counts = as.list(table(classes)),
      absent_from_codetools_globals = !(name %in% globals),
      all_occurrences_structurally_bound = all_structurally_bound,
      occurrence_evidence = occurrences[[name]])
  }
  list(
    codetools_global_variables = sort(intersect(globals, residual_names)), residuals = by_name,
    all_residuals_supported = length(by_name) > 0L && all(vapply(by_name, function(x) isTRUE(x$all_occurrences_structurally_bound), logical(1))))
}

output <- file(args[[2L]], open = "wt")
on.exit(close(output), add = TRUE)
for (group in input$source_groups) {
  expressions <- tryCatch(parse(file = group$source_path, keep.source = TRUE, encoding = "UTF-8"), error = identity)
  for (row in group$rows) {
    if (inherits(expressions, "error")) result <- list(row_id = row$row_id, status = "hold_source_parse_error", error = conditionMessage(expressions))
    else {
      targets <- find_targets(expressions, row$target_definition_name)
      if (length(targets) != 1L) result <- list(row_id = row$row_id, status = "hold_target_definition_not_unique", target_matches = length(targets))
      else {
        evidence <- audit_target(targets[[1L]], unlist(row$residual_noncall_names, use.names = FALSE))
        result <- list(
          row_id = row$row_id,
          status = if (isTRUE(evidence$all_residuals_supported)) "recoverable_lexically_bound_residuals" else "hold_unbound_mixed_or_outside_scope",
          target_definition_name = row$target_definition_name, target_definition_span = row$target_definition_span,
          source_path = group$source_path, source_sha256 = group$source_sha256,
          selected_context_tokens = row$selected_context_tokens, target_body_tokens = row$target_body_tokens,
          evidence = evidence)
      }
    }
    writeLines(jsonlite::toJSON(result, auto_unbox = TRUE, null = "null", na = "null", digits = NA), output)
  }
}
