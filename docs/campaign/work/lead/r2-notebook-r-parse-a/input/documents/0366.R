FLSSS <- function(len, v, target, ME, solutionNeed = 1L, LB = 1:len, UB = (length(v) - len + 1L):length(v), viaConjugate = FALSE, tlimit = 60, useBiSrchInFB = FALSE, NfractionDigits = Inf)
{
  valtype = "int"
  if(is.infinite(NfractionDigits)) valtype = "double"
  else
  {
    scaler = 10 ^ NfractionDigits
    v = as.integer(round(v * scaler))
    target = as.integer(round(target * scaler))
    ME = as.integer(round(ME * scaler))
  }


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
    if(valtype == "double")
    {
      target = target / ME
      v = v / ME
      ME = 1
    }
    rst = z_FLSSS(len, v, target, ME, LB = 1:len, UB = (length(v) - len + 1L):length(v), solutionNeed, tlimit, useBiSrchInFB, valtype)
    rst = unique(lapply(rst, function(x) sort(vindex[x][vindex[x] > 0L])))
    return(rst)
  }


  if(valtype == "double")
  {
    target = target / ME
    v = v / ME
    ME = 1
  }


  if(is.null(viaConjugate))
  {
    if(2L * len < length(v)) viaConjugate = T
  }


  if(viaConjugate)
  {
    target = sum(v) - target
    LBresv = LB
    LB = (1:length(v))[-UB]
    UB = (1:length(v))[-LBresv]
    len = length(v) - len
  }


  rst = z_FLSSS(len, v, target, ME, LB, UB, solutionNeed, tlimit, useBiSrchInFB, valtype)


  if(viaConjugate)
  {
    tmp = 1:length(v)
    rst = lapply(rst, function(x) tmp[-x])
  }
  rst
}