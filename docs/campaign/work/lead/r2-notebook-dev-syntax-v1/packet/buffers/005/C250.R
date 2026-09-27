#' Contour3d
#'
#' @param griddata A 3D array of data
#' @param level The level of the contour
#' @param x The x coordinates
#' @param y The y coordinates
#' @param z The z coordinates
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