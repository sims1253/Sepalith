# Differential checks for the exec-env gate (v3). Runs inside sandbox.sh.
# Usage: Rscript --vanilla r_diff.R UNIT.R SUITE.R TESTLIB.R FUNCTION MUTANT_DIR MAX_INPUTS
# MUTANT_DIR holds <id>.R (full mutated unit files) and conditions.tsv (<id>\t<R condition or empty>).
#
# 1. Records every call the suite makes to FUNCTION (argument values, matched to the formals).
# 2. Derives perturbed inputs from those calls: numeric x2 / negated / shifted / reversed / one
#    element / empty / a tie / an NA; character reversed / one element / empty / NA / blank;
#    logical flags flipped; data frame rows reversed / first row; match.arg alternatives; swaps
#    of equal-typed arguments.
# 3. For every declared-equivalent mutant: compares original and mutant on the recorded calls
#    and on every perturbed input the original accepts (inputs on which the original errors are
#    out of contract), on return value (all.equal plus class, attributes and NA/NaN pattern),
#    error versus value, and whether a warning fires. Inputs matching the mutant's stated
#    condition (SCOPED/BUGFIX[...]) are excluded.
# 4. Reports, per formal argument the function body reads, how many distinct values it takes
#    across the recorded calls (distinct supplied values; whether the default is used), argument pairs that
#    are always equal, and matrix-like arguments that never have more than one nonzero cell
#    per row and column.
# Output (stdout, tab-separated): CALLS n | INPUTS n | ARG name distinct_supplied default_used logical sparse |
#   EQUAL a b |
#   SAME id n | DIFF id input orig mut | FAIL message
args <- commandArgs(trailingOnly = TRUE)
unit <- args[[1]]; suite <- args[[2]]; testlib <- args[[3]]; fn <- args[[4]]
mdir <- args[[5]]; max_inputs <- as.integer(args[[6]])
lines <- character()
emit <- function(...) lines[length(lines) + 1L] <<- paste(..., sep = "\t")
clean <- function(x) substr(gsub("[\t\r\n]+", " ", paste(x, collapse = " ")), 1, 300)
show <- function(x) clean(tryCatch(deparse(x, nlines = 3, width.cutoff = 100), error = function(e) "<undeparsable>"))
sink(nullfile())
sink(file(nullfile(), open = "w"), type = "message")

result <- tryCatch({
  e <- new.env(parent = globalenv())
  sys.source(unit, envir = e)
  orig <- get(fn, envir = e)
  lib <- new.env(parent = globalenv())
  sys.source(testlib, envir = lib)
  t <- new.env(parent = lib)
  for (n in ls(e, all.names = TRUE)) assign(n, get(n, envir = e), envir = t)
  rec <- list()
  wrapper <- function(...) {
    call <- tryCatch(match.call(orig, sys.call()), error = function(err) NULL)
    if (!is.null(call) && length(rec) < 200L) {
      vals <- tryCatch(lapply(as.list(call)[-1], eval, envir = parent.frame()), error = function(err) NULL)
      if (!is.null(vals)) rec[[length(rec) + 1L]] <<- vals
    }
    orig(...)
  }
  assign(fn, wrapper, envir = t)
  set.seed(1L)
  texprs <- parse(suite, keep.source = FALSE)
  for (ex in texprs) try(eval(ex, t), silent = TRUE)
  rec <- unique(rec)
  emit("CALLS", length(rec))

  fmls <- formals(orig)
  read <- intersect(names(fmls), all.names(body(orig)))
  tryCatch({
  # argument diversity across the recorded calls
  for (a in setdiff(read, "...")) {
    vals <- vapply(rec, function(r) if (a %in% names(r)) show(r[[a]]) else "<default>", "")
    sparse <- NA
    mats <- Filter(function(v) (is.matrix(v) || is.data.frame(v) || is.table(v)) &&
                     is.numeric(as.matrix(v)) && length(dim(v)) == 2L,
                   lapply(rec, function(r) r[[a]]))
    if (length(mats)) {
      sparse <- all(vapply(mats, function(m) {
        nz <- as.matrix(m) != 0 & !is.na(as.matrix(m))
        all(rowSums(nz) <= 1) && all(colSums(nz) <= 1)
      }, TRUE))
    }
    given <- unique(vals[vals != "<default>"])
    flag <- all(vapply(rec, function(r) !(a %in% names(r)) || (is.logical(r[[a]]) && length(r[[a]]) == 1), TRUE))
    emit("ARG", a, length(given), any(vals == "<default>"), flag, sparse)
  }
  supplied <- unique(unlist(lapply(rec, names)))
  both_read <- intersect(supplied, read)
  pairs <- if (length(both_read) >= 2) utils::combn(both_read, 2, simplify = FALSE) else list()
  for (p in pairs) {
    both <- Filter(function(r) all(p %in% names(r)), rec)
    if (length(both) && all(vapply(both, function(r) identical(r[[p[1]]], r[[p[2]]]), TRUE)) &&
        !all(vapply(both, function(r) is.null(r[[p[1]]]) || is.logical(r[[p[1]]]), TRUE))) {
      emit("EQUAL", p[1], p[2])
    }
  }

  }, error = function(err) emit("WARN", paste("argument checks:", clean(conditionMessage(err)))))

  # perturbed inputs
  inputs <- lapply(rec, function(r) list(args = r, tag = "test call", recorded = TRUE))
  add <- function(base, k, v, tag) {
    b <- base
    b[k] <- list(v)
    inputs[[length(inputs) + 1L]] <<- list(args = b, tag = paste(k, tag), recorded = FALSE)
  }
  choices <- lapply(fmls, function(d) if (is.call(d) && identical(d[[1]], as.name("c"))) {
    v <- tryCatch(eval(d), error = function(err) NULL)
    if (is.character(v) && length(v) > 1) v else NULL
  } else NULL)
  for (r in rec) {
    for (k in names(r)) {
      v <- r[[k]]
      if (is.numeric(v) && is.null(dim(v)) && length(v) >= 1 && is.null(attr(v, "class"))) {
        add(r, k, v * 2, "x2"); add(r, k, -v, "negated"); add(r, k, v + 0.5, "+0.5")
        add(r, k, v[1], "first element"); add(r, k, v[0], "empty")
        if (length(v) >= 2) { w <- v; w[2] <- w[1]; add(r, k, w, "tie"); add(r, k, rev(v), "reversed") }
        w <- v; w[1] <- NA; add(r, k, w, "NA")
      } else if (is.matrix(v) && is.numeric(v)) {
        add(r, k, v * 2, "x2"); add(r, k, -v, "negated")
        w <- v; w[1] <- NA; add(r, k, w, "NA")
        if (nrow(v) > 1) add(r, k, v[rev(seq_len(nrow(v))), , drop = FALSE], "rows reversed")
      } else if (is.character(v) && is.null(attr(v, "class")) && length(v) >= 1) {
        if (length(v) >= 2) add(r, k, rev(v), "reversed")
        add(r, k, v[1], "first element"); add(r, k, character(), "empty")
        w <- v; w[1] <- NA; add(r, k, w, "NA"); w <- v; w[1] <- ""; add(r, k, w, "blank")
      } else if (is.logical(v) && length(v) == 1 && !is.na(v)) {
        add(r, k, !v, "flipped")
      } else if (is.data.frame(v) && nrow(v) >= 2) {
        add(r, k, v[rev(seq_len(nrow(v))), , drop = FALSE], "rows reversed")
        add(r, k, v[1, , drop = FALSE], "first row")
      }
    }
    for (k in names(choices)) {
      if (is.null(choices[[k]])) next
      for (ch in choices[[k]]) if (!identical(r[[k]], ch)) add(r, k, ch, paste0("= '", ch, "'"))
    }
    ks <- names(r)
    if (length(ks) >= 2) for (p in utils::combn(ks, 2, simplify = FALSE)) {
      a <- r[[p[1]]]; b <- r[[p[2]]]
      if (identical(class(a), class(b)) && length(a) == length(b) && !identical(a, b)) {
        s <- r; s[p[1]] <- list(b); s[p[2]] <- list(a)
        inputs[[length(inputs) + 1L]] <- list(args = s, tag = paste("swap", p[1], p[2]), recorded = FALSE)
      }
    }
  }
  if (length(inputs) > max_inputs) {
    keep <- c(which(vapply(inputs, `[[`, TRUE, "recorded")), which(!vapply(inputs, `[[`, TRUE, "recorded")))
    inputs <- inputs[keep[seq_len(max_inputs)]]
  }
  emit("INPUTS", length(inputs))

  outcome <- function(f, a) {
    warned <- FALSE
    set.seed(1L)
    r <- tryCatch(withCallingHandlers({
      setTimeLimit(elapsed = 2, transient = TRUE)
      v <- do.call(f, a, quote = TRUE)
      setTimeLimit(elapsed = Inf)
      list(ok = TRUE, value = v)
    }, warning = function(w) { warned <<- TRUE; invokeRestart("muffleWarning") },
    message = function(m) invokeRestart("muffleMessage")),
    error = function(err) { setTimeLimit(elapsed = Inf); list(ok = FALSE, value = conditionMessage(err)) })
    r$warn <- warned
    r
  }
  na_pattern <- function(x) {
    u <- tryCatch(unlist(x, use.names = FALSE), error = function(err) NULL)
    if (is.numeric(u) || is.complex(u)) list(is.na(u), is.nan(u)) else NULL
  }
  same_value <- function(a, b) {
    if (identical(a, b)) return(TRUE)
    if (is.function(a) && is.function(b)) return(identical(deparse(a), deparse(b)))
    if (is.environment(a) && is.environment(b)) return(isTRUE(all.equal(as.list(a), as.list(b))))
    identical(class(a), class(b)) && identical(names(attributes(a)), names(attributes(b))) &&
      isTRUE(all.equal(a, b, tolerance = 1e-8)) && identical(na_pattern(a), na_pattern(b))
  }
  summary_of <- function(o) {
    paste0(if (o$ok) show(o$value) else paste("error:", clean(o$value)), if (o$warn) " [warning]" else "")
  }
  baseline <- lapply(inputs, function(i) outcome(orig, i$args))
  conds <- if (file.exists(file.path(mdir, "conditions.tsv"))) {
    x <- readLines(file.path(mdir, "conditions.tsv"), warn = FALSE)
    stats::setNames(sub("^[^\t]*\t", "", x), sub("\t.*$", "", x))
  } else character()
  for (mf in sort(list.files(mdir, pattern = "\\.R$", full.names = TRUE))) {
    id <- sub("\\.R$", "", basename(mf))
    m <- new.env(parent = globalenv())
    ok <- tryCatch({ sys.source(mf, envir = m); TRUE }, error = function(err) FALSE)
    if (!ok) { emit("DIFF", id, "<sourcing the mutant>", "sources", "error"); next }
    mutant <- get(fn, envir = m)
    cond <- conds[id]
    compared <- 0L
    found <- FALSE
    for (j in seq_along(inputs)) {
      inp <- inputs[[j]]
      base <- baseline[[j]]
      if (!inp$recorded && !base$ok) next  # out of contract: the original rejects this input
      if (!is.na(cond) && nzchar(cond)) {
        excluded <- tryCatch(isTRUE(eval(parse(text = cond)[[1]], envir = list2env(inp$args, parent = e))),
                             error = function(err) FALSE)
        if (excluded) next
      }
      mo <- outcome(mutant, inp$args)
      compared <- compared + 1L
      differs <- base$ok != mo$ok || base$warn != mo$warn || (base$ok && !same_value(base$value, mo$value))
      if (differs) {
        emit("DIFF", id, paste0(inp$tag, ": ", show(inp$args)), summary_of(base), summary_of(mo))
        found <- TRUE
        break
      }
    }
    if (!found) emit("SAME", id, compared)
  }
  TRUE
}, error = function(err) { emit("FAIL", clean(conditionMessage(err))); FALSE })
sink(type = "message")
sink()
writeLines(lines)
