#' Contour3d
#'
#' @param griddata A \code{sp} \code{gridded} object
#' @param level The contour level
#' @param x,y,z The coordinates of the grid
#'
#' @return A list of triangles, vertices, and normals
#'
#' @export
contour3d <- function(
  griddata,
  level,
  x,
  y,
  z
) {
  # Run the marching cubes algorithm
  result <- marching_cubes(
    data = griddata,
    x = x,
    y = y,
    z = z,
    iso = level
  )

  # Set the vectors and coordinates
  list(
    triangles = result$triangles + 1,
    vertices = result$vertices,
    normals = result$normals
  )
}