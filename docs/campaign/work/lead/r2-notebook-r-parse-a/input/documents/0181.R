DemographicTable <- function(
    data, data.name = substitute(data), 
    groups = NULL,
    exclude = NULL, exclude_rx, 
    include, include_rx, 
    robust = TRUE,
    overall = TRUE, 
    compare = TRUE,
    pairwise = 3L,
    ...
) {
  
  force(data.name)
  if (is.character(data.name)) {
    if (length(data.name) != 1L || anyNA(data.name) || !nzchar(data.name)) stop('illegal data.name')
  } else data.name <- deparse1(data.name)
  
  if (!is.data.frame(data)) stop('input must be data.frame')
  data <- as.data.frame(data) # use S3
  if (anyDuplicated.default(names(data))) stop('Duplicated column names in raw data?')
  data <- data[!vapply(data, FUN = \(i) all(is.na(i)), FUN.VALUE = NA)] # remove all-missing columns 
  
  if (length(groups)) {
    if (!is.character(groups) || anyNA(groups) || !all(nzchar(groups))) stop('groups must be character without NA or zchar')
    groups <- unique.default(groups)
    if (any(id <- is.na(match(groups, table = names(data), nomatch = NA_integer_)))) stop(sQuote(groups[id]), ' not in names of data. Removed accidentally?')
    if (any(id <- vapply(data[groups], FUN = is.matrix, FUN.VALUE = NA, USE.NAMES = FALSE))) stop(sQuote(groups[id]), ' is/are matrix column(s).')
    if ((data[groups]) |> 
        vapply(FUN = is.logical, FUN.VALUE = NA) |>
        any()) warning(msg_logical())
  }
  
  if (!missing(exclude_rx)) {
    exclude <- unique.default(c(exclude, grepv(pattern = exclude_rx, x = names(data))))
  }
  
  include <- if (missing(include_rx)) {
    if (missing(include)) names(data) else include
  } else {
    ptn_include <- grepv(pattern = include_rx, x = names(data))
    if (missing(include)) ptn_include else unique.default(c(include, ptn_include))
  }
  
  include <- setdiff(x = include, y = exclude)
  if (!length(include)) stop('length-0 `include`: Nothing on the rows of demographic table')
  
  rm(exclude)
  
  if (any(id <- is.na(match(include, table = names(data), nomatch = NA_integer_)))) {
    message('Unknown variable(s): ', sQuote(include[id]), ' removed.')
    include <- include[!id]
  }
  
  include <- sort.default(include)
  data <- data[c(include, groups)]
  
  for (i in include) {
    if (is.character(data[[i]])) data[[i]] <- factor(data[[i]]) 
    # MUST!! otherwise missing groups in subset-data will not print zero-count
  }
  
  ##################################################################
  ## Inspect `groups` in detail (removing rows if needed)
  ##################################################################
  
  if (length(groups)) {
    
    names(groups) <- groups
    
    for (i in seq_along(groups)) {
      
      grp <- groups[i]
      grpv <- data[[grp]]
      
      if (any(id <- is.na(grpv))) {
        names(groups)[i] <- sprintf(fmt = '%s\nn=%d (%.1f%%) missing', names(groups)[i], sum(id), 1e2*mean.default(id))
      } # else do nothing
      
      if (length(unique(grpv[!id])) == 1L) {
        message('Column ', sQuote(grp), ' has single value, thus removed from `groups`.')
        groups <- setdiff(groups, grp)
      }
      
    } # remove any group with all-same entries
    
  }
  
  ############################################
  ## Inspect `include` in detail
  ############################################
  
  # without `groups`
  # .. copy tzh::class1List
  cl1 <- vapply(data[include], FUN = \(x) class(x)[1L], FUN.VALUE = '', USE.NAMES = TRUE)
  vlst <- split.default(names(cl1), f = factor(cl1))
  
  if (length(vlst$matrix)) {
    mtype <- vapply(data[vlst$matrix], FUN = typeof, FUN.VALUE = '')
    if (any(id_double <- (mtype == 'double'))) {
      vlst$difftime <- c(vlst$difftime, vlst$matrix[id_double][id_difftime <- vapply(data[vlst$matrix[id_double]], FUN = inherits, what = 'difftime', FUN.VALUE = NA)])
      vlst$numeric <- c(vlst$numeric, vlst$matrix[id_double][!id_difftime])
    }
    # c('factor', 'matrix') is in `vlst$factor` already...
    if (any(id_bool <- (mtype == 'logical'))) {
      vlst$logical <- c(vlst$logical, vlst$matrix[id_bool])
    }
    #if (any(!id_bool & !id_double)) stop('uncovered matrix column') # allow, just ignore these cases
  }
  
  ######################
  # Done! use `data`, `vlst` and `groups` below
  ######################
  
  ret0 <- if (overall) list(.sumtab(data, data.name = data.name, vlst = vlst, ...)) # else NULL      
  ret1 <- if (length(groups)) {
    # in this way, `names(groups)` won't be passed into [.sumtab_by]
    groups |>
      seq_along() |>
      lapply(FUN = \(i) {
        .sumtab_by(data = data, data.name = data.name, group = groups[i], vlst = vlst, compare = compare, robust = robust, pairwise = pairwise, ...)
      })
  }
  ret <- c(ret0, ret1)
  if (!length(ret)) stop('wont happen')
  
  class(ret) <- c('DemographicTable', 'listof')
  return(ret)
  
}