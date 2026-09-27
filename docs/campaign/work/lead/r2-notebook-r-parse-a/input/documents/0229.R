resolve_engine_args_mmap <- function(
  engine,
  engine_spec,
  engine_args,
  legacy_named_args,
  dot_engine_args
) {
  allowed <- engine_spec$engine_keys
  allow_extra_args <- isTRUE(engine_spec$allow_extra_args)

  unknown_engine_args <- setdiff(names(engine_args), allowed)
  if (length(unknown_engine_args) && !allow_extra_args) {
    warning(
      paste0(
        "Ignoring unknown `engine.args` key(s) for engine ",
        engine,
        ": ",
        paste(unknown_engine_args, collapse = ", "),
        ". Allowed keys are: ",
        if (length(allowed)) paste(allowed, collapse = ", ") else "<none>",
        "."
      ),
      call. = FALSE
    )
    keep_idx <- names(engine_args) %in% allowed
    engine_args <- engine_args[keep_idx]
  }
  if (allow_extra_args) {
    allowed <- unique(c(
      allowed,
      names(engine_args),
      names(dot_engine_args),
      names(legacy_named_args)
    ))
  }

  resolved <- engine_spec$defaults
  source_map <- if (length(resolved)) {
    stats::setNames(rep("default", length(resolved)), names(resolved))
  } else {
    stats::setNames(character(0), character(0))
  }

  apply_source <- function(src_list, source_label) {
    if (!length(src_list)) {
      return(invisible(NULL))
    }
    for (ii in seq_along(src_list)) {
      nm <- names(src_list)[ii]
      if (!nzchar(nm) || !(nm %in% allowed)) {
        next
      }

      if (
        nm %in%
          names(source_map) &&
          source_map[[nm]] != "default" &&
          source_map[[nm]] != source_label
      ) {
        warning(
          paste0(
            "Argument conflict for `",
            nm,
            "`: using value from ",
            source_label,
            " and overriding value from ",
            source_map[[nm]],
            "."
          ),
          call. = FALSE
        )
      }
      resolved[[nm]] <<- src_list[[ii]]
      source_map[[nm]] <<- source_label
    }
  }

  # Lowest to highest precedence.
  apply_source(dot_engine_args, "`...`")
  apply_source(legacy_named_args, "deprecated top-level arguments")
  apply_source(engine_args, "`engine.args`")

  resolved
}