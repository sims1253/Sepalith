#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: nse_occurrence_audit.R INPUT.json OUTPUT.jsonl")
input <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)

call_head <- function(node) {
  if (!is.call(node)) return(NA_character_)
  head <- node[[1L]]
  if (is.symbol(head)) return(as.character(head))
  if (is.call(head) && length(head) == 3L && is.symbol(head[[1L]]) &&
      is.symbol(head[[2L]]) && is.symbol(head[[3L]]) &&
      as.character(head[[1L]]) %in% c("::", ":::")) {
    return(paste0(as.character(head[[2L]]), as.character(head[[1L]]), as.character(head[[3L]])))
  }
  NA_character_
}

symbols_in <- function(node) {
  if (is.null(node)) return(character())
  all.names(node, functions = TRUE, unique = TRUE)
}

literal_strings <- function(node) {
  result <- character()
  visit <- function(value) {
    if (is.character(value)) result <<- c(result, value)
    else if (is.call(value) || is.expression(value) || is.pairlist(value)) {
      for (index in seq_along(value)) {
        if (!identical(value[[index]], quote(expr = ))) visit(value[[index]])
      }
    }
  }
  visit(node)
  unique(result)
}

colnames_receiver <- function(node) {
  if (!is.call(node) || !identical(call_head(node), "colnames") || length(node) != 2L || !is.symbol(node[[2L]])) return(NULL)
  as.character(node[[2L]])
}

schema_evidence <- function(fun) {
  result <- list()
  visit <- function(node) {
    if (!is.call(node)) return(invisible(NULL))
    if (identical(call_head(node), "%in%") && length(node) == 3L) {
      left <- colnames_receiver(node[[2L]])
      right <- colnames_receiver(node[[3L]])
      if (!is.null(left)) {
        for (name in literal_strings(node[[3L]])) result[[paste(left, name, sep = "\t")]] <<- TRUE
      }
      if (!is.null(right)) {
        for (name in literal_strings(node[[2L]])) result[[paste(right, name, sep = "\t")]] <<- TRUE
      }
    }
    if (length(node) >= 2L) for (index in 2:length(node)) {
      if (!identical(node[[index]], quote(expr = ))) visit(node[[index]])
    }
  }
  visit(fun)
  names(result)
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

audit_target <- function(fun, residual_names) {
  schema <- schema_evidence(fun)
  occurrences <- setNames(lapply(residual_names, function(x) list()), residual_names)
  visit <- function(node, frames = list()) {
    if (is.symbol(node)) {
      name <- as.character(node)
      if (!(name %in% residual_names)) return(invisible(NULL))
      classification <- "ordinary_expression_or_unresolved_global"
      construct <- NA_character_
      receiver <- NA_character_
      argument <- NA_character_
      support_node <- node
      for (frame in rev(frames)) {
        if (identical(frame$head, "[") && frame$index >= 3L && is.symbol(frame$call[[2L]])) {
          candidate_receiver <- as.character(frame$call[[2L]])
          marker <- intersect(symbols_in(frame$call), c(".N", ".I", ".SD", ".BY", ".GRP", ".EACHI", ".NGRP", ":="))
          if (length(marker) && paste(candidate_receiver, name, sep = "\t") %in% schema) {
            classification <- "data_table_nse_with_local_schema_evidence"
            construct <- paste(marker, collapse = ",")
            receiver <- candidate_receiver
          } else {
            classification <- "data_table_like_subset_without_sufficient_column_evidence"
            construct <- paste(marker, collapse = ",")
            receiver <- candidate_receiver
          }
          support_node <- frame$call
          break
        }
        if (identical(frame$head, "ggforce::facet_zoom") && frame$argument_name %in% c("x", "y")) {
          classification <- "facet_zoom_nse_pending_same_name_column_proof"
          construct <- frame$head
          argument <- frame$argument_name
          support_node <- frame$call
          break
        }
      }
      occurrence <- list(
        classification = classification,
        construct = construct,
        receiver = receiver,
        argument = argument,
        expression_sha256 = digest::digest(serialize(support_node, NULL, ascii = TRUE), algo = "sha256", serialize = FALSE)
      )
      occurrences[[name]][[length(occurrences[[name]]) + 1L]] <<- occurrence
      return(invisible(NULL))
    }
    if (!is.call(node)) return(invisible(NULL))
    head <- call_head(node)
    arg_names <- names(node)
    if (is.null(arg_names)) arg_names <- rep("", length(node))
    if (identical(head, "function")) {
      # Formal names and defaults are outside this audit's applied-body evidence.
      # Missing defaults are special R objects and must not be forced while walking.
      visit(node[[3L]], frames)
      return(invisible(NULL))
    }
    if (length(node) >= 2L) {
      for (index in 2:length(node)) {
        if (identical(node[[index]], quote(expr = ))) next
        frame <- list(head = head, index = index, argument_name = arg_names[[index]], call = node)
        visit(node[[index]], c(frames, list(frame)))
      }
    }
  }
  visit(fun)
  # A facet_zoom mapping is supported only when another occurrence of the
  # same name proves the data.table column using local schema evidence.
  for (name in residual_names) {
    direct <- any(vapply(occurrences[[name]], function(x) identical(x$classification, "data_table_nse_with_local_schema_evidence"), logical(1)))
    if (direct) for (index in seq_along(occurrences[[name]])) {
      if (identical(occurrences[[name]][[index]]$classification, "facet_zoom_nse_pending_same_name_column_proof")) {
        occurrences[[name]][[index]]$classification <- "facet_zoom_nse_with_same_name_column_proof"
      }
    }
  }
  supported_classes <- c("data_table_nse_with_local_schema_evidence", "facet_zoom_nse_with_same_name_column_proof")
  by_name <- list()
  for (name in residual_names) {
    classes <- vapply(occurrences[[name]], function(x) x$classification, character(1))
    by_name[[name]] <- list(
      occurrences = length(classes),
      class_counts = as.list(table(classes)),
      all_occurrences_supported = length(classes) > 0L && all(classes %in% supported_classes),
      occurrence_evidence = occurrences[[name]]
    )
  }
  list(
    schema_evidence = lapply(schema, function(value) {
      parts <- strsplit(value, "\t", fixed = TRUE)[[1L]]
      list(receiver = parts[[1L]], column = parts[[2L]])
    }),
    residuals = by_name,
    all_residuals_supported = length(by_name) > 0L && all(vapply(by_name, function(x) isTRUE(x$all_occurrences_supported), logical(1)))
  )
}

output <- file(args[[2]], open = "wt")
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
          status = if (isTRUE(evidence$all_residuals_supported)) "recoverable_occurrence_proven_nse" else "hold_unresolved_or_insufficient_nse_evidence",
          target_definition_name = row$target_definition_name,
          target_definition_span_prior_before_state = row$target_definition_span,
          source_path = group$source_path,
          source_sha256 = group$source_sha256,
          selected_context_tokens = row$selected_context_tokens,
          target_body_tokens = row$target_body_tokens,
          evidence = evidence
        )
      }
    }
    writeLines(jsonlite::toJSON(result, auto_unbox = TRUE, null = "null", na = "null", digits = NA), output)
  }
}
