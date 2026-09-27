generate.pvalues <- function(
  dat,
  test.fun,
  test.args = NULL,
  preprocess.fun = NULL,
  preprocess.args = NULL
) {
  #  make sure test function originates from package 'DiscreteTests'
  ## make sure it is a function or a string
  assert(
    check_function(test.fun),
    check_string(test.fun)
  )
  ## get all functions from package 'DiscreteTests'
  funs <- ls(asNamespace("DiscreteTests"))
  ## extract available test functions
  funs <- funs[endsWith(funs, "_test_pv")]
  ## make sure input 'test.fun' matches an available test function
  if(is.character(test.fun)) {
    ### match input 'test.fun' string to available test functions
    test.fun <- match.arg(tolower(test.fun), funs)
    ### convert string to actual function
    test.fun <- eval(parse(text = paste0("DiscreteTests::", test.fun)))
  } else {
    ### make sure input 'test.fun' matches an available test function
    OK <- FALSE
    for(fun in funs) {
      pkg_fun <- eval(parse(text = paste0("DiscreteTests::", fun)))
      if(all(all.equal(test.fun, pkg_fun) == TRUE)) {
        OK <- TRUE
        break;
      }
    }
    if(!OK) stop(paste("'test.fun' must be one of the '*_test_pv' functions of",
                       "package 'DiscreteTests'."))
  }
  
  # make sure parameters for 'test.fun' are in a named list or NULL
  assert_list(test.args, names = "unique", null.ok = TRUE)
  
  # make sure date preprocessing function is a function, string or NULL
  assert(
    check_function(preprocess.fun, null.ok = TRUE),
    check_string(preprocess.fun)
  )
  ## convert string to actual function (if necessary)
  if(is.character(preprocess.fun))
    preprocess.fun <- eval(parse(text = preprocess.fun))
  
  # make sure parameters for 'preprocess.fun' are in a named list or NULL
  assert_list(preprocess.args, names = "unique", null.ok = TRUE)
  
  if(!is.null(preprocess.fun)){
    # prepend data to preprocessing function's arguments list
    preprocess.args <- c(list(dat), preprocess.args)
    # set data parameter name according to first parameter of 'preprocess.fun'
    names(preprocess.args)[1] <- names(as.list(args(preprocess.fun)))[1]
    # perform preprocessing
    dat <- do.call(preprocess.fun, preprocess.args)
  }
  
  # get original data name
  data.name <- sapply(match.call(), deparse1)["dat"]
  # assign data to new variable with original name
  assign(data.name, dat)
  
  # prepend preprocessed data to test function's arguments list
  test.args <- c(list(as.name(data.name)), test.args)
  # set data parameter name according to first parameter of 'test.fun'
  names(test.args)[1] <- names(as.list(args(test.fun)))[1]
  # perform test(s)
  res <- do.call(test.fun, test.args)
  
  # output results
  return(res)
}