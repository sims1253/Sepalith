.registry <- new.env(parent = emptyenv())

#' Path of the on-disk model registry
#'
#' Reads the registry directory from the `modelcrate.registry_dir` option,
#' falling back to a user cache directory.
#'
#' @return A single directory path.
#' @keywords internal
registry_dir <- function() {
  dir <- getOption("modelcrate.registry_dir")
  if (is.null(dir)) {
    dir <- tools::R_user_dir("modelcrate", which = "cache")
  }
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  }
  dir
}

#' Register a fitted model
#'
#' Stores a model under a name and version, both in memory and as an RDS file,
#' together with metadata such as metrics and the training timestamp.
#'
#' @param model A fitted object inheriting from `mc_model`.
#' @param name Registry name of the model.
#' @param version Optional version string; defaults to the next integer.
#' @param metrics Optional named numeric vector of validation metrics.
#' @return The version string, invisibly.
#' @family registry
#' @export
register_model <- function(model, name, version = NULL, metrics = NULL) {
  stopifnot(inherits(model, "mc_model"), is.character(name), length(name) == 1L)
  existing <- list_versions(name)
  if (is.null(version)) {
    version <- as.character(length(existing) + 1L)
  }
  if (version %in% existing) {
    stop(sprintf("Model '%s' already has version %s.", name, version), call. = FALSE)
  }
  entry <- list(
    model = model,
    name = name,
    version = version,
    class = class(model)[1],
    metrics = metrics,
    created = Sys.time()
  )
  key <- paste(name, version, sep = "@")
  assign(key, entry, envir = .registry)
  saveRDS(entry, file.path(registry_dir(), paste0(key, ".rds")))
  invisible(version)
}

#' List the versions of a registered model
#'
#' Combines versions held in memory with those saved on disk.
#'
#' @param name Registry name of the model.
#' @return A character vector of versions, sorted numerically where possible.
#' @family registry
#' @export
list_versions <- function(name) {
  prefix <- paste0(name, "@")
  in_memory <- ls(.registry, pattern = paste0("^", prefix))
  on_disk <- sub("\\.rds$", "", list.files(registry_dir(), pattern = paste0("^", prefix)))
  versions <- unique(sub(prefix, "", c(in_memory, on_disk), fixed = TRUE))
  as_num <- suppressWarnings(as.numeric(versions))
  versions[order(as_num, versions)]
}

#' Load a registered model
#'
#' Returns the requested version, or the latest one, reading it from disk if
#' it is not already in memory.
#'
#' @param name Registry name of the model.
#' @param version Optional version string; defaults to the latest version.
#' @return A registry entry: a list with `model` and its metadata.
#' @family registry
#' @export
load_model <- function(name, version = NULL) {
  versions <- list_versions(name)
  if (length(versions) == 0L) {
    stop(sprintf("No model registered under '%s'.", name), call. = FALSE)
  }
  if (is.null(version)) {
    version <- versions[length(versions)]
  }
  key <- paste(name, version, sep = "@")
  if (exists(key, envir = .registry, inherits = FALSE)) {
    return(get(key, envir = .registry))
  }
  path <- file.path(registry_dir(), paste0(key, ".rds"))
  if (!file.exists(path)) {
    stop(sprintf("Version %s of '%s' not found.", version, name), call. = FALSE)
  }
  entry <- readRDS(path)
  assign(key, entry, envir = .registry)
  entry
}

#' Remove a registered model version
#'
#' Deletes a version from memory and disk.
#'
#' @param name Registry name of the model.
#' @param version Version string to remove.
#' @return `TRUE` if anything was removed, invisibly.
#' @family registry
#' @export
remove_model <- function(name, version) {
  key <- paste(name, version, sep = "@")
  path <- file.path(registry_dir(), paste0(key, ".rds"))
  found <- FALSE
  if (exists(key, envir = .registry, inherits = FALSE)) {
    rm(list = key, envir = .registry)
    found <- TRUE
  }
  if (file.exists(path)) {
    found <- file.remove(path) || found
  }
  invisible(found)
}

#' Tabulate every registered model
#'
#' Builds a data frame of all registered models and versions with their
#' class, creation time and a chosen metric.
#'
#' @param metric Optional metric name to include as a column.
#' @return A data frame with one row per model version.
#' @family registry
#' @export
registry_table <- function(metric = NULL) {
  files <- list.files(registry_dir(), pattern = "@.*\\.rds$")
  keys <- sub("\\.rds$", "", files)
  rows <- lapply(keys, function(key) {
    parts <- strsplit(key, "@", fixed = TRUE)[[1]]
    entry <- load_model(parts[1], parts[2])
    value <- if (!is.null(metric) && metric %in% names(entry$metrics)) {
      entry$metrics[[metric]]
    } else {
      NA_real_
    }
    data.frame(name = entry$name, version = entry$version, class = entry$class,
      created = entry$created, metric = value, stringsAsFactors = FALSE)
  })
  if (length(rows) == 0L) {
    return(data.frame(name = character(), version = character(), class = character(),
      created = as.POSIXct(character()), metric = numeric()))
  }
  do.call(rbind, rows)
}

#' Predict with a registered model
#'
#' Loads a model from the registry and returns predictions in a tidy data
#' frame keyed by row, along with the model name and version used.
#'
#' @param name Registry name of the model.
#' @param newdata A data frame of predictors.
#' @param version Optional version string; defaults to the latest version.
#' @param ... Passed to the model's `predict()` method.
#' @return A data frame with columns `.row`, `.pred`, `model` and `version`.
#' @export
predict_registered <- function(name, newdata, version = NULL, ...) {
  entry <- load_model(name, version)
  pred <- stats::predict(entry$model, newdata, ...)
  if (is.matrix(pred)) {
    # keep the most likely class alongside the full probability matrix
    out <- data.frame(.row = seq_len(nrow(newdata)),
      .pred = colnames(pred)[max.col(pred, ties.method = "first")])
    out <- cbind(out, as.data.frame(pred, optional = TRUE))
  } else {
    out <- data.frame(.row = seq_len(nrow(newdata)), .pred = pred)
  }
  out$model <- entry$name
  out$version <- entry$version
  out
}
