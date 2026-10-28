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
#    condition (SCOPED/BUGFIX/QUIRK[...]) are excluded.
# 4. Reports, per formal argument the function body reads, how many distinct values it takes
#    across the recorded calls (distinct supplied values; whether the default is used), argument pairs that
#    are always equal, and matrix-like arguments that never have more than one nonzero cell
#    per row and column.
# Output (stdout, tab-separated): CALLS n | INPUTS n | ARG name distinct_supplied default_used logical sparse |
#   EQUAL a b |
#   SAME id n | DIFF id input orig mut | FAIL message
local({
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
    used_default <- any(vapply(rec, function(r) !(a %in% names(r)), TRUE))
    given_vals <- lapply(Filter(function(r) a %in% names(r), rec), function(r) r[[a]])
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
    flag <- all(vapply(rec, function(r) !(a %in% names(r)) || (is.logical(r[[a]]) && length(r[[a]]) == 1), TRUE))
    emit("ARG", a, length(unique(given_vals)), used_default, flag, sparse)
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
      ev <- new.env(parent = globalenv())
      assign(fn, f, envir = ev)
      quoted <- lapply(a, function(x) if (is.language(x)) call("quote", x) else x)
      v <- eval(as.call(c(list(as.name(fn)), quoted)), ev)
      setTimeLimit(elapsed = Inf)
      list(ok = TRUE, value = v)
    }, warning = function(w) { warned <<- TRUE; invokeRestart("muffleWarning") },
    message = function(m) invokeRestart("muffleMessage")),
    error = function(err) { setTimeLimit(elapsed = Inf); list(ok = FALSE, value = conditionMessage(err)) })
    r$warn <- warned
    r
  }
  outcome1 <- function(f, a) {
    warned <- FALSE
    r <- tryCatch(withCallingHandlers({
      setTimeLimit(elapsed = 1, transient = TRUE)
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
  suite_text <- paste(readLines(suite, warn = FALSE), collapse = "\n")
  # stored calls, terms and headings are presentation; they count only if the suite reads them ($call, "call")
  ignorable <- Filter(function(n) !grepl(paste0("\\$\\s*", n, "\\b|[\"']", n, "[\"']"), suite_text),
                      c("call", "terms", "heading"))
  ignore_attrs <- character()
  strip <- function(x) {
    if (length(ignore_attrs) && !isS4(x) && !is.environment(x) && !is.function(x) && !is.language(x) && !is.null(x)) {
      # ATTR[...] / NAMES: the listed attributes are incidental; values, order, class and other attributes count
      for (n in ignore_attrs) {
        if (n == "row.names" && is.data.frame(x)) attr(x, "row.names") <- .set_row_names(nrow(x))
        else if (n == "names" && is.data.frame(x)) names(x) <- NULL
        else if (n == "dimnames" && !is.null(dim(x)) && !is.data.frame(x)) dimnames(x) <- NULL
        else if (n == "names" && !is.null(names(x))) names(x) <- NULL
        else if (!(n %in% c("names", "dimnames", "row.names", "class", "dim")) &&
                 !is.null(attr(x, n, exact = TRUE))) attr(x, n) <- NULL
      }
    }
    for (n in ignorable) if (!is.null(attr(x, n, exact = TRUE))) attr(x, n) <- NULL
    if (is.list(x) && !is.data.frame(x) && !is.null(names(x))) {
      drop <- names(x) %in% ignorable & vapply(x, function(v) is.language(v) || is.character(v), TRUE)
      if (any(drop)) {
        keep_attr <- attributes(x)[setdiff(names(attributes(x)), "names")]
        x <- x[!drop]
        attributes(x) <- c(attributes(x), keep_attr)
      }
    }
    x
  }
  probes <- list(c(-2, -0.5, 0, 0.3, 1, 2.5), 0.5, c(0, 1), NA_real_, 1:3)
  same_closure <- function(a, b, depth) {
    if (!identical(names(formals(a)), names(formals(b)))) return(FALSE)
    for (x in probes) {
      oa <- outcome1(a, list(x)); ob <- outcome1(b, list(x))
      if (oa$ok != ob$ok || oa$warn != ob$warn || (oa$ok && !same_value(oa$value, ob$value, depth + 1L))) return(FALSE)
    }
    TRUE
  }
  same_value <- function(a, b, depth = 0L) {
    a <- strip(a); b <- strip(b)
    if (identical(a, b)) return(TRUE)
    if (depth > 6L) return(FALSE)
    if (is.function(a) && is.function(b)) return(same_closure(a, b, depth))
    if (is.environment(a) && is.environment(b)) {
      la <- as.list(a, all.names = TRUE); lb <- as.list(b, all.names = TRUE)
      if (!identical(sort(names(la)), sort(names(lb)))) return(FALSE)
      for (n in names(la)) if (!same_value(la[[n]], lb[[n]], depth + 1L)) return(FALSE)
      return(TRUE)
    }
    if (isS4(a) || isS4(b)) {
      if (!isS4(a) || !isS4(b) || !identical(class(a), class(b))) return(FALSE)
      for (n in methods::slotNames(a)) {
        if (!same_value(methods::slot(a, n), methods::slot(b, n), depth + 1L)) return(FALSE)
      }
      return(TRUE)
    }
    if (is.language(a) && is.language(b)) return(identical(deparse(a), deparse(b)))
    if (!identical(class(a), class(b)) || !identical(sort(names(attributes(a))), sort(names(attributes(b))))) return(FALSE)
    if (is.list(a) && is.list(b) && !is.data.frame(a)) {
      if (length(a) != length(b) || !identical(names(a), names(b))) return(FALSE)
      for (k in seq_along(a)) if (!same_value(a[[k]], b[[k]], depth + 1L)) return(FALSE)
      other <- setdiff(names(attributes(a)), c("names", "class"))
      for (n in other) if (!same_value(attr(a, n, exact = TRUE), attr(b, n, exact = TRUE), depth + 1L)) return(FALSE)
      return(TRUE)
    }
    if (!identical(na_pattern(a), na_pattern(b))) return(FALSE)
    if (num_tol > 0 && (is.numeric(a) || is.complex(a)) && (is.numeric(b) || is.complex(b))) {
      # NUMTOL[tol]: absolute or relative tolerance per element; shape and attributes still exact
      if (!identical(dim(a), dim(b)) || length(a) != length(b) || !identical(names(a), names(b))) return(FALSE)
      ok <- is.na(a) | abs(a - b) <= num_tol | abs(a - b) <= num_tol * abs(a)
      return(all(ok | (is.na(a) & is.na(b))))
    }
    isTRUE(all.equal(a, b, tolerance = 1e-8))
  }
  num_tol <- 0
  summary_of <- function(o) {
    paste0(if (o$ok) show(o$value) else paste("error:", clean(o$value)), if (o$warn) " [warning]" else "")
  }
  cond_env <- function(a) {
    env <- new.env(parent = e)
    nm <- names(a)
    if (is.null(nm)) nm <- rep("", length(a))
    for (k in seq_along(a)) if (nzchar(nm[k])) assign(nm[k], a[[k]], envir = env)
    assign("dots", unname(a[!nzchar(nm)]), envir = env)
    # formals not supplied: their defaults, evaluated lazily in this environment (as R would)
    for (f in setdiff(names(fmls), c(nm, "..."))) {
      d <- fmls[[f]]
      if (!(is.name(d) && !nzchar(as.character(d)))) {
        eval(call("delayedAssign", f, d, env, env))
      }
    }
    env
  }
  baseline <- lapply(inputs, function(i) outcome(orig, i$args))
  # conditions.tsv: <id>\t<condition>\t<tolerance>\t<attr,attr> (any field may be empty)
  spec <- if (file.exists(file.path(mdir, "conditions.tsv"))) {
    x <- strsplit(readLines(file.path(mdir, "conditions.tsv"), warn = FALSE), "\t", fixed = TRUE)
    stats::setNames(lapply(x, function(v) trimws(c(v, "", "", "")[2:4])), vapply(x, `[`, "", 1))
  } else list()
  for (mf in sort(list.files(mdir, pattern = "\\.R$", full.names = TRUE))) {
    id <- sub("\\.R$", "", basename(mf))
    m <- new.env(parent = globalenv())
    ok <- tryCatch({ sys.source(mf, envir = m); TRUE }, error = function(err) FALSE)
    if (!ok) { emit("DIFF", id, "<sourcing the mutant>", "sources", "error"); next }
    mutant <- get(fn, envir = m)
    sp <- if (is.null(spec[[id]])) c("", "", "") else spec[[id]]
    cond <- sp[1]
    num_tol <- if (nzchar(sp[2])) as.numeric(sp[2]) else 0
    ignore_attrs <- if (nzchar(sp[3])) strsplit(sp[3], ",", fixed = TRUE)[[1]] else character()
    # which eligible inputs the condition excludes (reported; a condition excluding all is a blanket)
    eligible <- which(vapply(seq_along(inputs), function(j) inputs[[j]]$recorded || baseline[[j]]$ok, TRUE))
    excluded_set <- integer()
    if (nzchar(cond)) {
      expr <- tryCatch({ px <- parse(text = cond, keep.source = FALSE); if (length(px) != 1L) stop("one expression"); px[[1]] },
                       error = function(err) NULL)
      if (is.null(expr)) { emit("BADCOND", id, "does not parse as one R expression"); next }
      bad <- NULL
      for (j in eligible) {
        v <- tryCatch(eval(expr, envir = cond_env(inputs[[j]]$args)), error = function(err) err)
        if (inherits(v, "error") || !is.logical(v) || length(v) != 1L || is.na(v)) {
          bad <- paste0(inputs[[j]]$tag, ": ", if (inherits(v, "error")) clean(conditionMessage(v)) else
            paste("not TRUE/FALSE:", show(v)))
          break
        }
        if (v) excluded_set <- c(excluded_set, j)
      }
      if (!is.null(bad)) { emit("BADCOND", id, bad); next }
    }
    emit("EXCL", id, length(excluded_set), length(eligible))
    compared <- 0L
    found <- FALSE
    for (j in eligible) {
      inp <- inputs[[j]]
      base <- baseline[[j]]
      if (j %in% excluded_set) next
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
})
