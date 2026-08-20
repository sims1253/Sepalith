#!/usr/bin/env Rscript
# rw_harness.R — execution harness for the rewrite-verification track.
#
# Runs INSIDE the sandboxed Rscript process launched by
# rewrite_verify_proto.py (unshare -rn + rlimits + temp HOME). Three modes:
#
#   probe   source files, locate `target`, generate inputs (tier 0 default
#           call / tier 1 name-heuristic / tier 2 type-ladder search), run
#           at several sizes, canonicalize + digest the outputs.
#   compare same as probe (single variant). The DRIVER runs one process per
#           variant (orig / rewritten) with identical payloads and compares
#           digests across processes — full process isolation.
#   bench   source orig files, snapshot fn, source new files, interleave
#           timed batches of both variants at each size (microbenchmark-lite
#           on base system.time: adaptive batch so one batch >= 50 ms,
#           ABAB interleaving to decorrelate machine drift), plus gc-peak and
#           Rprofmem allocation totals per variant.
#
# Output: one JSON object written to argv[2]. Zero non-base dependencies
# except digest + jsonlite (both present in the system library, which
# --vanilla still sees).
#
# Determinism contract: every do.call is preceded by set.seed(SEED_BASE +
# size) and inputs are REGENERATED from that stream, so the orig and the
# rewritten variant see byte-identical arguments in their own processes.

args_all <- commandArgs(TRUE)
payload_file <- args_all[1]
out_path <- args_all[2]

# real (read) library paths travel as argv[3] (colon-joined): --vanilla
# ignores the R_LIBS* env vars, and the sandboxed HOME no longer owns the
# user library where jsonlite/digest live. jsonlite needs them BEFORE the
# payload can be parsed.
if (length(args_all) >= 3 && nzchar(args_all[3])) {
  libs <- strsplit(args_all[3], ":")[[1]]
  suppressWarnings(.libPaths(unique(c(libs, .libPaths()))))
}
payload <- jsonlite::fromJSON(payload_file, simplifyVector = FALSE)

`%||%` <- function(a, b) if (is.null(a)) b else a

SEED_BASE <- as.integer(payload$seed %||% 1234L)
sizes <- as.integer(unlist(payload$sizes %||% list(10L, 100L, 1000L)))
max_attempts <- as.integer(payload$max_attempts %||% 8L)
fixed_args <- payload$fixed_args            # named R exprs, SZ = current size
attach_pkgs <- unlist(payload$attach %||% list())
for (p in attach_pkgs) suppressMessages(try(library(p, character.only = TRUE),
                                           silent = TRUE))

result <- list(status = "init", mode = payload$mode)

# ---------------------------------------------------------------------------
# sourcing (best-effort, per top-level expression, errors recorded not fatal)
# ---------------------------------------------------------------------------

source_best_effort <- function(files) {
  errs <- list()
  for (f in unlist(files)) {
    p <- tryCatch(parse(f), error = function(e) e)
    if (inherits(p, "try-error") || inherits(p, "error")) {
      errs[[length(errs) + 1]] <- list(file = basename(f),
                                       message = conditionMessage(p))
      next
    }
    for (i in seq_along(p)) {
      invisible(tryCatch(eval(p[i], envir = globalenv()),
                         error = function(e) NULL))
    }
  }
  errs
}

result$source_errors <- source_best_effort(payload$files)

target <- payload$target
fn <- if (exists(target, envir = globalenv(), mode = "function"))
  get(target, envir = globalenv(), mode = "function") else NULL
if (is.null(fn)) {
  result$status <- "target_not_found"
  jsonlite::write_json(result, out_path, auto_unbox = TRUE, null = "null")
  quit(save = "no", status = 0)
}

# ---------------------------------------------------------------------------
# canonicalization + digest
# ---------------------------------------------------------------------------
# env_count lives in globalenv: environments/functions/pointers are replaced
# by a type tag (they do not serialize deterministically), but they are
# COUNTED so a digest over env-bearing output can be flagged as weak.

canon_ <- function(x, digits) {
  if (is.null(x)) return(list(t = "null"))
  ty <- typeof(x)
  if (is.environment(x) || is.function(x) || ty %in%
      c("externalptr", "symbol", "expression", "raw", "bytecode", "weakref",
        "S4")) {
    .GlobalEnv$rw_env_count <- .GlobalEnv$rw_env_count + 1L
    return(list(t = typeof(x)))
  }
  if (is.numeric(x)) {
    y <- signif(x, digits)
    y[y == 0] <- 0                       # normalize -0
    at <- attributes(x)
    keep <- at[intersect(names(at), c("names", "dim", "dimnames"))]
    return(list(t = "num", v = y, a = if (length(keep)) keep else NULL))
  }
  if (is.factor(x)) return(list(t = "factor", v = as.character(x),
                                l = levels(x)))
  if (is.list(x)) {
    at <- attributes(x)
    keep <- at[intersect(names(at), c("names"))]
    cls <- class(x)[1]
    return(list(t = if (!identical(cls, "list")) paste0("list:", cls) else "list",
                v = lapply(x, canon_, digits),
                rn = if (identical(cls, "data.frame"))
                  rownames(x) else NULL))
  }
  if (is.complex(x)) return(list(t = "cplx", v = signif(x, digits)))
  list(t = typeof(x), v = x)             # logical / character
}

canon_and_digest <- function(x) {
  .GlobalEnv$rw_env_count <- 0L
  c <- canon_(x, 8)
  n_env <- .GlobalEnv$rw_env_count
  d <- tryCatch(digest::digest(c, algo = "xxhash64"),
                error = function(e) NA_character_)
  list(digest = d, env_count = n_env)
}

shape_of <- function(x) {
  list(class = paste(class(x), collapse = "/"), typeof = typeof(x),
       length = length(x))
}

# ---------------------------------------------------------------------------
# input generation: tier 0 default call / tier 1 name heuristics /
# tier 2 type-ladder search (coordinate descent, seeded)
# ---------------------------------------------------------------------------

NAME_RULES <- list(
  list(rx = "^(n|size|count|num\\w*|times|k|len|nrow|ncol|niter|nboot|reps|nsim|n_sim|nmax|n_start|n_end|steps|iters|m$)",
       gen = function(sz) as.integer(sz)),
  list(rx = "^(i|j|idx|index|which|ids?)$", gen = function(sz) seq_len(sz)),
  list(rx = "(df|dat|data|frame|tbl|table|sample_?\\w*s?|obs|records?)$|^\\.",
       gen = function(sz) data.frame(a = stats::rnorm(sz),
                                     b = sample(c("x", "y", "z"), sz, TRUE),
                                     c = stats::runif(sz),
                                     stringsAsFactors = FALSE)),
  list(rx = "^(m|mat|matrix|a|b|x|y|z|v|w|vec|values?|nums?|numbers|scores|xs|ys|coef|coefs|beta|theta|mu|sigma|eps|e|u|lambda_?\\w*)$",
       gen = function(sz) stats::rnorm(sz)),
  list(rx = "(str?|string|name|prefix|suffix|pattern|sep|path|file|label|key|group|gender|sex|treat\\w*|type|level|colou?rs?|labels?|title|id)s?$",
       gen = function(sz) paste(sample(letters, 6), collapse = "")),
  list(rx = "^(flag|verbose|quiet|keep|drop|warn|debug|is_|use_|na_?rm|rm|drop_na|all|force|recursive|simplify|infer|progress)",
       gen = function(sz) TRUE),
  list(rx = "^(p|q|prob|alpha|beta|lambda|mu|sigma|sd|mean|tol|eps|delta|rate|scale|shape|lo|hi|min|max|const|c\\d|threshold|conf|level)$",
       gen = function(sz) stats::runif(1, 0.1, 0.9)),
  list(rx = "^(f|fn|fun|\\.f|func|agg|statistic|stat_fn|simplify|transformer)",
       gen = function(sz) function(x) x),
  list(rx = "(formula|fml|spec|model)$", gen = function(sz) NULL)  # ladder
)

LADDER <- list(
  function(sz) 1L,
  function(sz) stats::rnorm(sz),
  function(sz) paste(sample(letters, 6), collapse = ""),
  function(sz) TRUE,
  function(sz) seq_len(sz),
  function(sz) stats::rexp(sz),
  function(sz) as.integer(seq_len(sz)),
  function(sz) data.frame(a = stats::rnorm(sz),
                          b = sample(c("x", "y"), sz, TRUE),
                          stringsAsFactors = FALSE),
  function(sz) matrix(stats::rnorm(max(sz, 2L) * 3L), ncol = 3),
  function(sz) sample(c(TRUE, FALSE), sz, TRUE),
  function(sz) NULL,
  function(sz) list_of(sz)
)

list_of <- function(sz) lapply(seq_len(sz), function(i) list(a = i, b = i^2))

heuristic_for <- function(nm) {
  low <- tolower(nm)
  for (r in NAME_RULES) if (grepl(r$rx, low)) return(r$gen)
  NULL
}

build_attempts <- function(f) {
  fm <- formals(f)
  nms <- names(fm)
  # an argument is "missing" (no default) iff deparse of its default is "";
  # "..." is never filled by the harness
  missing_nms <- nms[vapply(fm, function(x) identical(deparse(x)[1], ""),
                            logical(1))]
  missing_nms <- setdiff(missing_nms, "...")
  attempts <- list()
  if (length(missing_nms) == 0L) attempts[[length(attempts) + 1]] <- list()
  if (length(missing_nms) > 0L) {
    a1 <- list()
    for (nm in missing_nms) {
      g <- heuristic_for(nm)
      if (is.null(g)) g <- LADDER[[2]]           # numeric vector default guess
      a1[[nm]] <- g
    }
    attempts[[length(attempts) + 1]] <- a1
    # tier 2: coordinate descent over the ladder (deterministic)
    pos <- rep(1L, length(missing_nms))
    base <- attempts[[length(attempts)]]
    for (it in seq_len(max_attempts)) {
      vary <- (it - 1L) %% length(missing_nms) + 1L
      pos[vary] <- pos[vary] %% length(LADDER) + 1L
      a <- base
      for (k in seq_along(missing_nms))
        a[[missing_nms[k]]] <- LADDER[[pos[k]]]
      attempts[[length(attempts) + 1]] <- a
    }
  }
  attempts
}

eval_args <- function(spec, sz) {
  set.seed(SEED_BASE + sz)                       # identical stream per size
  vals <- lapply(spec, function(g) g(sz))
  # keep dots out
  vals[vapply(vals, is.null, logical(1))] <- NULL
  vals
}

# curated-spec inputs: named R expressions in which SZ is the current size;
# regenerate deterministically per call (same contract as eval_args)
fixed_spec <- function(fa) {
  lapply(fa, function(e) {
    function(sz) eval(parse(text = gsub("\\bSZ\\b", as.character(sz), e)))
  })
}

run_call <- function(f, spec, sz) {
  set.seed(SEED_BASE + sz)
  args <- eval_args(spec, sz)
  warns <- character(0)
  out <- withCallingHandlers(
    do.call(f, args),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  list(out = out, args = args, warns = warns)
}

# ---------------------------------------------------------------------------
# probe / compare
# ---------------------------------------------------------------------------

if (payload$mode %in% c("probe", "compare")) {
  attempts <- if (is.null(fixed_args)) build_attempts(fn) else
    list(fixed_spec(fixed_args))
  chosen <- NULL
  errors_seen <- list()
  for (ai in seq_along(attempts)) {
    ok <- FALSE
    tryCatch({
      r <- run_call(fn, attempts[[ai]], sizes[1])
      ok <- TRUE
      chosen <- attempts[[ai]]
    }, error = function(e) {
      errors_seen[[length(errors_seen) + 1]] <<-
        list(stage = paste0("attempt_", ai),
             class = class(e)[1], message = conditionMessage(e))
    })
    if (ok) break
  }
  if (is.null(chosen)) {
    last <- if (length(errors_seen)) errors_seen[[length(errors_seen)]]$message else ""
    result$status <- if (grepl("could not find function|object .* not found", last))
      "deps_missing" else "runtime_error"
    result$errors_seen <- errors_seen
    jsonlite::write_json(result, out_path, auto_unbox = TRUE, null = "null")
    quit(save = "no", status = 0)
  }
  result$tier <- if (length(chosen) == 0L) 0L else 1L   # refined by driver
  result$arg_names <- names(chosen)
  size_res <- list()
  any_ok <- FALSE
  for (sz in sizes) {
    tryCatch({
      r <- run_call(fn, chosen, sz)
      cd <- canon_and_digest(r$out)
      size_res[[length(size_res) + 1]] <-
        list(size = sz, ok = TRUE, digest = cd$digest,
             env_count = cd$env_count, shape = shape_of(r$out),
             n_warnings = length(r$warns))
      any_ok <- TRUE
    }, error = function(e) {
      size_res[[length(size_res) + 1]] <<-
        list(size = sz, ok = FALSE, error = conditionMessage(e))
    })
  }
  result$sizes <- size_res
  result$status <- if (any_ok) "ok" else "runtime_error"
  result$errors_seen <- errors_seen
  jsonlite::write_json(result, out_path, auto_unbox = TRUE, null = "null")
  quit(save = "no", status = 0)
}

# ---------------------------------------------------------------------------
# bench (both variants in ONE process, interleaved ABAB timing)
# ---------------------------------------------------------------------------

if (payload$mode == "bench") {
  min_total <- as.numeric(payload$min_total %||% 0.25)
  max_reps <- as.integer(payload$max_reps %||% 40L)
  fn_orig <- fn
  err_new <- source_best_effort(payload$files_new)
  fn_new <- if (exists(target, envir = globalenv(), mode = "function"))
    get(target, envir = globalenv(), mode = "function") else NULL
  if (is.null(fn_new)) {
    result$status <- "new_not_found"
    result$source_errors_new <- err_new
    jsonlite::write_json(result, out_path, auto_unbox = TRUE, null = "null")
    quit(save = "no", status = 0)
  }
  attempts <- if (is.null(fixed_args)) build_attempts(fn_orig) else
    list(fixed_spec(fixed_args))
  chosen <- NULL
  for (a in attempts) {
    ok <- tryCatch({
      run_call(fn_orig, a, sizes[1]); TRUE
    }, error = function(e) FALSE)
    if (ok) {
      chosen <- a
      break
    }
  }
  if (is.null(chosen)) {
    result$status <- "no_working_inputs"
    jsonlite::write_json(result, out_path, auto_unbox = TRUE, null = "null")
    quit(save = "no", status = 0)
  }
  per_size <- list()
  for (sz in sizes) {
    one <- function(f) {
      set.seed(SEED_BASE + sz)
      args <- eval_args(chosen, sz)
      do.call(f, args)
    }
    t_probe <- system.time(one(fn_orig))[["elapsed"]]
    if (!is.finite(t_probe)) next
    per_call <- max(t_probe, 2e-4)
    batch <- min(2000L, max(1L, as.integer(0.05 / per_call)))
    reps <- max(3L, min(max_reps, as.integer(min_total / (per_call * batch))))
    t_orig <- numeric(reps)
    t_new <- numeric(reps)
    # warmup
    invisible(tryCatch(one(fn_orig), error = function(e) NULL))
    invisible(tryCatch(one(fn_new), error = function(e) NULL))
    for (r in seq_len(reps)) {
      t_orig[r] <- system.time(
        for (b in seq_len(batch)) invisible(tryCatch(one(fn_orig),
                                                    error = function(e) NULL)))[["elapsed"]]
      t_new[r] <- system.time(
        for (b in seq_len(batch)) invisible(tryCatch(one(fn_new),
                                                    error = function(e) NULL)))[["elapsed"]]
    }
    # memory: gc peak delta + Rprofmem allocation total (one call each)
    mem_of <- function(f) {
      gc(reset = TRUE)
      invisible(tryCatch(one(f), error = function(e) NULL))
      g <- gc()                                  # cols: 3 = Ncells max Mb,
      peak_mb <- round(g[1, 3] + g[2, 6], 1)     # 6 = Vcells max Mb
      mf <- file.path(tempdir(), "rwpm.log")
      unlink(mf)
      ok <- tryCatch({
        utils::Rprofmem(mf); invisible(tryCatch(one(f), error = function(e) NULL)); TRUE
      }, error = function(e) FALSE)
      if (ok) try(utils::Rprofmem(NULL), silent = TRUE)
      # NOTE: this R build undercounts small interpreted-loop allocations in
      # Rprofmem — treated as an opportunistic signal, never a gate
      alloc_kb <- 0
      if (ok && file.exists(mf)) {
        lines <- readLines(mf, warn = FALSE)
        nums <- as.numeric(sub("^([0-9]+) :.*$", "\\1",
                               lines[grepl("^[0-9]+ :", lines)]))
        alloc_kb <- round(sum(nums[is.finite(nums)]) / 1024, 1)
      }
      unlink(mf)
      list(peak_mb = peak_mb, alloc_kb = alloc_kb)
    }
    mo <- tryCatch(mem_of(fn_orig), error = function(e) list(peak_mb = NA, alloc_kb = NA))
    mn <- tryCatch(mem_of(fn_new), error = function(e) list(peak_mb = NA, alloc_kb = NA))
    med_o <- stats::median(t_orig) / batch
    med_n <- stats::median(t_new) / batch
    per_size[[length(per_size) + 1]] <- list(
      size = sz, batch = batch, reps = reps,
      orig_ms = round(med_o * 1000, 4), new_ms = round(med_n * 1000, 4),
      speedup = round(med_o / max(med_n, 1e-9), 3),
      orig_iqr_ms = round(stats::IQR(t_orig) / batch * 1000, 4),
      new_iqr_ms = round(stats::IQR(t_new) / batch * 1000, 4),
      mem_orig = mo, mem_new = mn)
  }
  result$status <- "ok"
  result$per_size <- per_size
  jsonlite::write_json(result, out_path, auto_unbox = TRUE, null = "null",
                       digits = NA)
  quit(save = "no", status = 0)
}

result$status <- "unknown_mode"
jsonlite::write_json(result, out_path, auto_unbox = TRUE, null = "null")
