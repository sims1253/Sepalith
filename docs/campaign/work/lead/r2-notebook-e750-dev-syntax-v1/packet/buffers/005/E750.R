#' @title Contour3d
#'
#' @description
#' Draws a contour on a 3D mesh.
#'
#' @param griddata A \code{gridded} object.
#' @param level The level of the contour.
#' @param x,y,z The coordinates of the grid.
#'
#' @return A list with the following elements:
#' \itemize{
#'   \item \code{triangles} The indices of the triangles in the mesh.
#'   \item \code{vertices} The coordinates of the vertices.
#'   \item \code{normals} The normals of the triangles.
#' }
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