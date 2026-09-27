`*.Kernel` <- function(k1, k2) {
  composed_kernel <- KernelMulComposed$new(
    k1,
    k2,
    paste0(k1$name, ' * ', k2$name)
  )
  return(composed_kernel)
}