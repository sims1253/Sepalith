.sumtab <- function(data, data.name, vlst, fmt = '%.1f', ...) {# useNA = c('no', 'always'), 
  
  out_num <- if (length(.num <- c(vlst$integer, vlst$numeric))) {
    names(.num) <- .num
    data[.num] |>
      lapply(FUN = .sumstat.default, fmt = fmt, ...) |>
      unlist(use.names = TRUE)
  } #else NULL
  
  out_difft <- if (length(.difft <- vlst$difftime)) {
    d_difft <- data[.difft]
    names(d_difft) <- paste0(.difft, ' (', vapply(data[.difft], FUN = attr, which = 'units', exact = TRUE, FUN.VALUE = ''), ')') # ?base::units.difftime
    d_difft |>
      lapply(FUN = .sumstat.default, fmt = fmt, ...) |>
      unlist(use.names = TRUE)
  } #else NULL
  
  out_bool <- if (length(.bool <- vlst$logical)) {
    d_bool <- data[.bool]
    names(d_bool) <- paste0(.bool, ': n (%)')
    d_bool |>
      vapply(FUN = .sumstat.logical, fmt = fmt, ..., FUN.VALUE = '')
  } #else NULL
  
  out_factor <- if (length(.fact <- c(vlst$character, vlst$factor, vlst$ordered))) {
    d_fact <- data[.fact]
    names(d_fact) <- paste0(.fact, ': n (%)')
    d_fact |>
      lapply(FUN = .sumstat, fmt = fmt, ...) |>
      unlist(use.names = TRUE) 
  } #else NULL

  ret0 <- c(out_num, out_difft, out_bool, out_factor)
  ret <- array(ret0, dim = c(length(ret0), 1L), dimnames = list(
    names(ret0), 
    paste0('n=', .row_names_info(data, type = 2L))
  ))
  attr(ret, which = 'data.name') <- data.name
  attr(ret, which = 'group') <- '' # important
  attr(ret, which = 'compare') <- FALSE
  #class(ret) <- c('sumtab', class(ret))
  class(ret) <- 'sumtab'
  return(ret)
}