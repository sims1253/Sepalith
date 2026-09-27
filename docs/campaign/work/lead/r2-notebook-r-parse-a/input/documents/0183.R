.sumtab_by <- function(data, data.name, vlst, group, robust = TRUE, compare = TRUE, pairwise = 3L, ...) { # SMD = FALSE, 
  
  if (!is.character(group) || length(group) != 1L || anyNA(group) || !nzchar(group)) stop('`group` must be len-1 character')
  
  fgrp <- factor(data[[group]])
  gidx <- split.default(seq_along(fgrp), f = fgrp) # missingness in `fgrp` dropped
  gN <- lengths(gidx, use.names = FALSE)
  
  ret <- do.call(cbind, args = lapply(gidx, FUN = \(id) { # (id = gidx[[1L]])
    .sumtab(data[id, , drop = FALSE], data.name = '', vlst = vlst, ...)
  }))
  colnames(ret) <- sprintf(fmt = '%s\nn=%d (%.1f%%)', names(gidx), gN, 1e2*gN/length(fgrp)) # before removing NA!!!
  
  # removing single 'group' for p-values
  txt_g1 <- if (any(g1 <- (gN == 1L))) {
    gidx <- gidx[!g1]
    #paste0('(', sum(g1), ' ', sQuote(group), ' level(s) of\n single obs omitted)')
    paste0('(', sum(g1), ' level(s) of\n single obs omitted)')
  } # else NULL
  
  ng <- length(gidx)
  if (ng < 2L) return(ret)
  
  if (compare) {
    p_double <- c(vlst$integer, vlst$numeric, vlst$difftime) |> vapply(FUN = \(i) compare_double(demo_get(x = data[[i]], gidx = gidx), robust = robust, pairwise = pairwise, ...), FUN.VALUE = '')
    p_bool <- vlst$logical |> vapply(FUN = \(i) compare_bool(demo_get(x = data[[i]], gidx = gidx), pairwise = pairwise, ...), FUN.VALUE = '')
    p_factor <- c(vlst$character, vlst$factor, vlst$ordered) |> vapply(FUN = \(i) compare_factor(x = data[[i]], g = fgrp, ...), FUN.VALUE = '')
    pval <- c(p_double, p_bool, p_factor)
    if (dim(ret)[1L] != length(pval)) stop('demographic table contruction wrong: pval do not match summary stats')
    ret_compare <- as.matrix(pval)
    colnames(ret_compare) <- paste(c('Signif', txt_g1), collapse = '\n')
  } else ret_compare <- NULL
  
  ret <- cbind(ret, ret_compare)
  attr(ret, which = 'group') <- group
  attr(ret, which = 'data.name') <- data.name
  attr(ret, which = 'compare') <- compare
  class(ret) <- 'sumtab'
  return(ret)

}