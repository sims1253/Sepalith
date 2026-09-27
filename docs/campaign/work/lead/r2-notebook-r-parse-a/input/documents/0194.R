CommunityColor <- function (ecotonefinder, method = c("vegclust", "cmeans"),
                            pal = c("diverge_hcl", "terrain_hcl", "sequential_hcl", "rainbow_hcl"))
{
  # Warnings:
  if (method == "vegclust") {
    if (is.null(ecotonefinder$vegclust)) {
      stop("Community color require vegclust method in EcotoneFinder")
    }
    Cluster <- rep(NA, length(ecotonefinder$vegclust$mobileCenters))
    for (i in 1:length(ecotonefinder$vegclust$mobileCenters)) {
      Cluster[i] <- which.max(ecotonefinder$vegclust$mobileCenters[,i])
    }
  }
  if (method == "cmeans") {
    if (is.null(ecotonefinder$cmeans)) {
      stop("Community color require cmeans method in EcotoneFinder")
    }
    Cluster <- rep(NA, length(ecotonefinder$cmeans$centers))
    for (i in 1:length(ecotonefinder$cmeans$centers)) {
      Cluster[i] <- which.max(ecotonefinder$cmeans$centers[,i])
    }
  }

  # Color patterns:
  if (pal == "diverge_hcl") {
    colvec <- colorspace::diverge_hcl(max(Cluster))
  }
  if (pal == "terrain_hcl") {
    colvec <- colorspace::terrain_hcl(max(Cluster))
  }
  if (pal == "sequential_hcl") {
    colvec <- colorspace::sequential_hcl(max(Cluster))
  }
  if (pal == "rainbow_hcl") {
    colvec <- colorspace::rainbow_hcl(max(Cluster))
  }
  CommunityColor <- rep(NA, length(Cluster))
  for (i in 1:length(Cluster)) {
    CommunityColor[i] <- colvec[Cluster[i]]
  }
  return(CommunityColor)
}