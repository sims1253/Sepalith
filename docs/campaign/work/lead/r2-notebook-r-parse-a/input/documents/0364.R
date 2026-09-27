FLSSSvariableTree <- function(len, v, target, ME, solutionNeed = 1L, LB = 1L : len, UB = (length(v) - len + 1L) : length(v), viaConjugate = FALSE, tlimit = 60, useBiSrchInFB = FALSE, useFloat = FALSE)
{
  premineRst = premine(len, v, target, ME)
  if(!is.null(premineRst)) return(premineRst)


  if(len == 0)
  {
    len = length(v)
    v = c(rep(0, len), v)
    vindex = c(rep(0L, len), 1L : len)
    sortOrder = order(v)
    v = v[sortOrder]
    vindex = vindex[sortOrder]
    rst = z_FLSSSvariableTree(len, v, target, ME, LB = 1L : len, UB = (length(v) - len + 1L) : length(v), solutionNeed, tlimit, useBiSrchInFB, useFloat)
    rst = unique(lapply(rst, function(x) sort(vindex[x][vindex[x] > 0L])))
    return(rst)
  }


  if(is.null(viaConjugate))
  {
    if(2L * len < length(v)) viaConjugate = T
  }


  if(viaConjugate)
  {
    target = sum(v) - target
    LBresv = LB
    LB = (1L : length(v))[-UB]
    UB = (1L : length(v))[-LBresv]
    len = length(v) - len
  }


  rst = z_FLSSSvariableTree(len, v, target, ME, LB, UB, solutionNeed, tlimit, useBiSrchInFB, useFloat)


  if(viaConjugate)
  {
    tmp = 1L : length(v)
    rst = lapply(rst, function(x) tmp[-x])
  }
  rst
}