`Simplify.+` <- function(expr, add = TRUE, scache = NULL) {
  if (length(expr) == 2) {
    if (add) {
      return(expr[[2]])
    } else if (is.uminus(expr[[2]])) {
      return(expr[[2]][[2]])
    } else if (is.uplus(expr[[2]])) {
      return(call("-", expr[[2]][[2]]))
    } else {
      return(expr)
    }
  }
  a <- expr[[2]]
  b <- expr[[3]]

  if (
    is.numconst(a, 0) ||
      (is.call(a) &&
        format1(a[[1]]) %in% c("rep", "rep.int", "rep_len") &&
        is.numconst(a[[2]], 0))
  ) {
    #browser()
    return(if (add) b else call("-", b))
  } else if (
    is.numconst(b, 0) ||
      (is.call(b) &&
        format1(b[[1]]) %in% c("rep", "rep.int", "rep_len") &&
        is.numconst(b[[2]], 0))
  ) {
    #browser()
    return(a)
  } else if (add && is.uminus(a) && !is.uminus(b)) {
    a <- b
    b <- expr[[2]][[2]]
    add <- FALSE
    expr <- call("-", a, b)
  } else if (identical(a, b)) {
    return(if (add) Simplify_(call("*", 2, a), scache) else 0)
  } else if (!is.call(a) && !is.call(b)) {
    if (add) {
      # just reorder
      expr[-1] <- expr[1 + order(sapply(expr[-1], format1))]
    }
    return(expr)
  }
  # factorise most repeated terms
  alc <- Lincomb(a)
  blc <- Lincomb(b)
  if (add) {
    lc <- c(alc, blc)
  } else {
    # inverse sminus in b
    blc <- lapply(blc, function(it) {
      it$sminus <- !it$sminus
      it
    })
    lc <- c(alc, blc)
  }
  #browser()
  # sum purely numeric terms
  inum <- which(sapply(lc, function(it) {
    length(it$num) == 0 && length(it$den) == 0
  }))
  if (length(inum) > 1) {
    term <- sum(sapply(lc[inum], function(it) {
      (if (it$sminus) -1 else 1) * it$fa$num / it$fa$den
    }))
    lc[[inum[1]]] <- list(
      fa = list(num = abs(term), den = 1),
      sminus = term < 0
    )
    lc <- lc[-inum[-1]]
  }
  bch <- ta <- tsim <- po <- ilc <- ind <- list()
  for (cnd in c("num", "den")) {
    # character bases in num/den
    bch[[cnd]] <- unlist(lapply(lc, function(it) {
      lapply(it[[cnd]]$b, format1)
    }))
    # powers
    po[[cnd]] <- do.call(c, lapply(lc, function(it) it[[cnd]]$p), quote = TRUE)
    # index of the lc term for each bnch
    ta[[cnd]] <- table(bch[[cnd]])
    ta[[cnd]] <- ta[[cnd]][ta[[cnd]] > 1] # keep only repeated bases
    tsim[[cnd]] <- outer(bch[[cnd]], names(ta[[cnd]]), `==`)
    ilc[[cnd]] <- unlist(lapply(seq_along(lc), function(i) {
      rep(i, length(lc[[i]][[cnd]]$b))
    }))
    # index of the base in a given term (nd) for each bnch
    ind[[cnd]] <- unlist(lapply(seq_along(lc), function(i) {
      seq_along(lc[[i]][[cnd]]$b)
    }))
  }
  #browser()
  # fnd will be the name "num" or "den" where the first factor
  # will be taken. ond is the "other" name (if fnd=="num", then ond == "den")
  # we select the candidate which is most repeated provided that it
  # has at least one numeric power occurance.
  taa <- unlist(ta)
  ota <- order(taa, decreasing = TRUE)
  ntan <- length(ta$num)
  fnd <- NA
  #browser()
  for (i in ota) {
    cnd <- if (i > ntan) "den" else "num"
    ita <- i - if (i > ntan) ntan else 0
    ib <- bch[[cnd]] == names(ta[[cnd]])[ita]
    pisnum <- any(sapply(po[[cnd]][ib], is.numeric))
    if (pisnum || (idup <- anyDuplicated(sapply(po[[cnd]][ib], format1)))) {
      fnd <- cnd
      iit <- which(ib) # the bases equal to factor
      if (pisnum) {
        p_fa <- min(
          sapply(po[[cnd]][ib], function(p) if (is.numeric(p)) p else NA),
          na.rm = TRUE
        )
      } else {
        p_fa <- po[[cnd]][ib][[idup]]
      }
      i_lc <- ilc[[cnd]][iit]
      i_nd <- ind[[cnd]][iit]
      break
    }
  }
  if (is.na(fnd)) {
    return(lc2expr(lc, scache))
  } # nothing to factorize, just order terms
  ond <- if (fnd == "num") "den" else "num"
  # create nd with the first factor
  fa_nd <- list(
    num = list(b = list(), p = list()),
    den = list(b = list(), p = list()),
    sminus = FALSE,
    fa = list(num = 1, den = 1)
  )
  fa_nd[[fnd]]$b <- lc[[i_lc[1]]][[fnd]]$b[i_nd[1]]
  fa_nd[[fnd]]$p <- list(p_fa)
  # decrease p in the lc terms
  for (i in seq_along(i_lc)) {
    lc[[i_lc[i]]][[fnd]]$p[[i_nd[i]]] <- Simplify_(
      call("-", lc[[i_lc[i]]][[fnd]]$p[[i_nd[i]]], p_fa),
      scache
    )
  }

  for (cnd in c(fnd, ond)) {
    # see if other side can provide factors
    for (i in seq_along(ta[[cnd]])) {
      if (
        (cnd == fnd && i == ita) ||
          ta[[fnd]][ita] != ta[[cnd]][i] ||
          any(ilc[[cnd]][tsim[[cnd]][, i]] != i_lc)
      ) {
        next # no common layout with the factor
      }
      ib <- bch[[cnd]] == names(ta[[cnd]])[i]
      # see if it has numeric power
      if (!any(sapply(po[[cnd]][ib], is.numeric))) {
        next
      }
      iit <- which(ib) # the bases equal to factor
      p_fa <- min(
        sapply(po[[cnd]][ib], function(p) if (is.numeric(p)) p else NA),
        na.rm = TRUE
      )
      i_lc <- ilc[[cnd]][iit]
      i_nd <- ind[[cnd]][iit]
      fa_nd[[cnd]]$b <- append(fa_nd[[cnd]]$b, lc[[i_lc[1]]][[cnd]]$b[i_nd[1]])
      fa_nd[[cnd]]$p <- append(fa_nd[[cnd]]$p, p_fa)
      # decrease p in the lc terms
      for (i in seq_along(i_lc)) {
        lc[[i_lc[i]]][[cnd]]$p[[i_nd[i]]] <- Simplify_(
          call("-", lc[[i_lc[i]]][[cnd]]$p[[i_nd[i]]], p_fa),
          scache
        )
      }
    }
  }
  #browser()
  # form final symbolic expression
  # replace all i_lc by one product of fa_nd and lincomb of the reduced nds
  rest <- Simplify_(lc2expr(lc[i_lc], scache), scache)
  if (is.neg.expr(rest)) {
    rest <- negate.expr(rest)
    fa_nd$sminus <- !fa_nd$sminus
  }
  fa_nd$num$b <- append(fa_nd$num$b, rest)
  fa_nd$num$p <- append(fa_nd$num$p, 1)
  lc <- c(list(fa_nd), lc[-i_lc])
  return(lc2expr(lc, scache))
}