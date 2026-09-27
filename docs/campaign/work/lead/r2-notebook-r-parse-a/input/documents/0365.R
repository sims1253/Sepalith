mFLSSSparVariableTree <- function(maxCore = 7L, len, mV, mTarget, mME, viaConjugate = NULL, solutionNeed = 1L, tlimit = 60, dl = ncol(mV), du = ncol(mV), randomizeTargetOrder = FALSE, useBiSrchInFB = FALSE, useFloat = FALSE)
{
  premineRst = premine(len, mV, mTarget, mME)
  if(!is.null(premineRst)) return(premineRst)


  # source("../src/legacy/rfuns.r")
  if(is.matrix(mV)) mV = as.data.frame(mV)


  dl = dl + 0L # materialize the values
  du = du + 0L


  fixedSize = T
  if(len == 0)
  {
    fixedSize = F
    len = nrow(mV)
    mV = as.data.frame(lapply(mV, function(x)c(rep(0, len), x)))
    randomizeTargetOrder = T
  }
  backout = function(mV, len){unique(lapply(mV, function(x) sort(x[x > len] - len)))}


  shouldConjugate = F
  if(is.null(viaConjugate) & 2L * len > nrow(mV)) shouldConjugate = T
  else if(!is.null(viaConjugate))
  {
    if(viaConjugate) shouldConjugate = T
  }


  if(shouldConjugate)
  {
    len = nrow(mV) - len
    mTarget = colSums(mV) - mTarget
  }


  if(ncol(mV) == 1L)
  {
    cat("Please call FLSSS for single dimensional set.\n")
    return(list())
  }


  LB = 1L : len
  UB = (nrow(mV) - len + 1L) : nrow(mV)


  # _______________________________________________________________________________________________
  # If mV are comonotonic
  # if(min(unlist(lapply(mV, function(x) min(diff(x))))) >= 0)
  # {
  #   d = ncol(mV)
  #   return(z_mFLSSScomo(len, mV, d, 0L, dl, d - du, du, mTarget, mME, LB, UB, solutionNeed, tlimit, useFloat, useBiSrchInFB))
  # }
  # _______________________________________________________________________________________________




  # _______________________________________________________________________________________________
  # z_viaLeadingCol <- function(len, mV, ME, target, randomOrderKeyTarget)
  info = z_viaLeadingCol(len, mV, mME, mTarget, randomizeTargetOrder, dl, du, F)
  if(is.character(info)) return(info)


  # _______________________________________________________________________________________________
  # Find the leading column that most correlates the rest columns, order other columns by the leading column. In the recent research, this substantially accelerates the speed.
  leadingCol = which.max(colSums(cor(mV, method = "spearman")))
  leadingColOrder = order(mV[[leadingCol]])
  mV = mV[leadingColOrder, ]
  info2 = z_adhereIndexCol(len, mV, mME, mTarget, randomizeTargetOrder, dl, du, F)
  if(!is.list(info2)) return(info2)


  # info = info2
  # It is mythical for now that, even length(info$keyTarget) > length(info2$keyTarget), adding index as the key column still makes things faster... Interesting, so I decided to do "/2", to encourage adding the index column.
  if(is.null(info) | length(info$keyTarget) > length(info2$keyTarget) / 2) info = info2
  # if(verbose) cat("Final atom tasks = ", length(info$keyTarget), "\n")


  # tlimit = tlimit * maxCore
  d = ncol(info$v)
  dl = info$dldu[1]
  du = info$dldu[2]
  zeroBasedKeyInd = info$zeroBasedKeyInd
  rst = z_mFLSSSvariableTree(maxCore, len, info$v, d, 0, dl, d - du, du, zeroBasedKeyInd, info$originalTarget, info$keyTarget, info$scaleFactor, info$ME, LB, UB, solutionNeed, tlimit, useFloat, useBiSrchInFB)
  rst = lapply(rst, function(x) leadingColOrder[x])
  if(shouldConjugate)
  {
    tmp = 1L : nrow(mV)
    rst = lapply(rst, function(x) tmp[-x])
  }
  if(!fixedSize) rst = backout(rst, len)
  rst
  # list(solution = rst, memoryImage = NA)
}