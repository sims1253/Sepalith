#!/usr/bin/env Rscript
# Corrected lexical-scope audit for the frozen 2,273 residual rows.
# This is a fresh packet. It never modifies the earlier loop-scope output.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: corrected_loop_scope_audit.R INPUT.json OUTPUT.jsonl")
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
        is.call(node[[3L]]) && identical(call_head(node[[3L]]), "function")) {
      result[[length(result) + 1L]] <- node[[3L]]
    }
  }
  result
}

binding_hash <- function(node) digest::digest(serialize(node, NULL, ascii = TRUE), algo = "sha256", serialize = FALSE)

missing_formal <- function(value) {
  inherits(value, "sepalith_missing_formal")
}

safe_list_element <- function(values, index) {
  rendered <- tryCatch(paste(deparse(values[[index]]), collapse = ""), error = function(error) "")
  if (identical(rendered, "")) return(structure(list(), class = "sepalith_missing_formal"))
  values[[index]]
}

safe_find_globals <- function(fun) {
  tryCatch({
    found <- codetools::findGlobals(eval(fun, envir = baseenv()), merge = FALSE)
    list(ok = TRUE, variables = as.character(found$variables), error = NULL)
  }, error = function(error) {
    # A globals-analysis error is evidence failure. It must never become an
    # empty global set, because that would turn an unbound name into a recovery.
    list(ok = FALSE, variables = character(), error = conditionMessage(error))
  })
}

audit_target <- function(fun, residual_names) {
  occurrences <- setNames(lapply(residual_names, function(x) list()), residual_names)
  record <- function(name, classification, binding_type = NA_character_, binding_sha256 = NA_character_, location = NA_character_) {
    occurrences[[name]][[length(occurrences[[name]]) + 1L]] <<- list(
      classification = classification, binding_type = binding_type,
      binding_sha256 = binding_sha256, location = location)
  }
  visit <- function(node, bindings = list(), parent_head = NA_character_, parent_index = NA_integer_, location = "body") {
    if (is.symbol(node)) {
      name <- as.character(node)
      if (!(name %in% residual_names)) return(invisible(NULL))
      if (identical(parent_head, "$") || identical(parent_head, "@") ||
          identical(parent_head, "::") || identical(parent_head, ":::")) {
        record(name, "member_literal_not_lexical", location = location)
      } else if (!is.null(bindings[[name]])) {
        record(name, "lexically_bound_occurrence", bindings[[name]]$type,
               bindings[[name]]$sha256, location)
      } else {
        record(name, "unbound_or_unsupported_occurrence", location = location)
      }
      return(invisible(NULL))
    }
    if (!is.call(node)) return(invisible(NULL))
    head <- call_head(node)
    # Call heads are executable symbol uses. The old visitor skipped node[[1]],
    # which could hide an unbound callable or a bound function-valued variable.
    visit(node[[1L]], bindings, head, 1L, "call_head")
    if (identical(head, "function") && length(node) >= 3L) {
      local <- bindings
      formal_names <- names(node[[2L]])
      formal_values <- as.list(node[[2L]])
      for (index in seq_along(formal_names)) {
        name <- formal_names[[index]]
        if (nzchar(name) && !identical(name, "...")) {
          local[[name]] <- list(type = "function_formal", sha256 = binding_hash(node[[2L]]))
        }
      }
      # Formal defaults execute in the function call environment. Visit them
      # with all formal bindings and enclosing lexical bindings instead of
      # dropping them as the old visitor did.
      for (index in seq_along(formal_values)) {
        value <- safe_list_element(formal_values, index)
        if (!missing_formal(value)) {
          visit(value, local, "formal_default", as.integer(index), "formal_default")
        }
      }
      visit(node[[3L]], local, "function", 3L, "function_body")
      return(invisible(NULL))
    }
    if (identical(head, "for") && length(node) == 4L && is.symbol(node[[2L]])) {
      binder <- as.character(node[[2L]])
      if (binder %in% residual_names) record(binder, "binding_definition", "for_loop_variable", binding_hash(node), "for_binder")
      # The sequence is evaluated before the binder is established.
      visit(node[[3L]], bindings, "for", 3L, "for_sequence")
      local <- bindings
      local[[binder]] <- list(type = "for_loop_variable", sha256 = binding_hash(node))
      visit(node[[4L]], local, "for", 4L, "for_body")
      return(invisible(NULL))
    }
    if (length(node) >= 2L) {
      # Convert the call to a list first. Direct node[[index]] access throws
      # for an intentionally empty call argument; empty arguments carry no
      # symbol occurrence and must be skipped safely.
      values <- as.list(node)
      for (index in 2:length(values)) {
        value <- safe_list_element(values, index)
        if (!missing_formal(value)) visit(value, bindings, head, as.integer(index), location)
      }
    }
    invisible(NULL)
  }
  # Root formals and their defaults are bindings in the function call frame.
  root_bindings <- list()
  root_formal_names <- names(fun[[2L]])
  root_formal_values <- as.list(fun[[2L]])
  for (index in seq_along(root_formal_names)) {
    name <- root_formal_names[[index]]
    if (nzchar(name) && !identical(name, "...")) {
      root_bindings[[name]] <- list(type = "root_function_formal", sha256 = binding_hash(fun[[2L]]))
    }
  }
  for (index in seq_along(root_formal_values)) {
    value <- safe_list_element(root_formal_values, index)
    if (!missing_formal(value)) visit(value, root_bindings, "formal_default", as.integer(index), "root_formal_default")
  }
  visit(fun[[3L]], root_bindings, "function", 3L, "root_function_body")
  globals <- safe_find_globals(fun)
  by_name <- list()
  supported <- c("binding_definition", "lexically_bound_occurrence")
  for (name in residual_names) {
    evidence <- occurrences[[name]]
    classes <- if (length(evidence)) vapply(evidence, function(x) x$classification, character(1)) else character()
    all_structurally_bound <- length(classes) > 0L && all(classes %in% supported) &&
      any(classes == "lexically_bound_occurrence") && globals$ok && !(name %in% globals$variables)
    by_name[[name]] <- list(
      occurrences = length(classes), class_counts = as.list(table(classes)),
      absent_from_codetools_globals = globals$ok && !(name %in% globals$variables),
      all_occurrences_structurally_bound = all_structurally_bound,
      occurrence_evidence = evidence)
  }
  list(
    codetools_ok = globals$ok, codetools_global_variables = sort(intersect(globals$variables, residual_names)),
    codetools_error = globals$error, residuals = by_name,
    all_residuals_supported = length(by_name) > 0L && globals$ok &&
      all(vapply(by_name, function(x) isTRUE(x$all_occurrences_structurally_bound), logical(1)))
  )
}

output <- file(args[[2L]], open = "wt")
on.exit(close(output), add = TRUE)
for (group in input$source_groups) {
  expressions <- tryCatch(parse(file = group$source_path, keep.source = TRUE, encoding = "UTF-8"), error = identity)
  for (row in group$rows) {
    if (inherits(expressions, "error")) {
      result <- list(row_id = row$row_id, status = "hold_source_parse_error", error = conditionMessage(expressions))
    } else {
      targets <- find_targets(expressions, row$target_definition_name)
      if (length(targets) != 1L) {
        result <- list(row_id = row$row_id, status = "hold_target_definition_not_unique", target_matches = length(targets))
      } else {
        evidence <- audit_target(targets[[1L]], unlist(row$residual_noncall_names, use.names = FALSE))
        result <- list(
          row_id = row$row_id,
          status = if (isTRUE(evidence$all_residuals_supported)) "recoverable_lexically_bound_residuals" else "hold_unbound_mixed_or_outside_scope",
          target_definition_name = row$target_definition_name, target_definition_span = row$target_definition_span,
          source_path = group$source_path, source_sha256 = group$source_sha256,
          selected_context_tokens = row$selected_context_tokens, target_body_tokens = row$target_body_tokens,
          audit_version = "corrected_formal_defaults_call_heads_fail_closed_globals_v1",
          evidence = evidence)
      }
    }
    writeLines(jsonlite::toJSON(result, auto_unbox = TRUE, null = "null", na = "null", digits = NA), output)
  }
}
