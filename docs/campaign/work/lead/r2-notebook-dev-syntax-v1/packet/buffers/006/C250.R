#' Create a class for a given status and decay
#'
#' @param status_vec A numeric vector of status values.
#' @param decay_vec A numeric vector of decay values.
#' @return A numeric vector of class values.
#' @keywords internal
.create_conclass <- function(status_vec, decay_vec) {
  ifelse(
    status_vec == 2 & decay_vec %in% 1:2,
    3,
    ifelse(
      status_vec == 2 & decay_vec %in% 3:4,
      4,
      ifelse(
        status_vec == 2 & decay_vec %in% 5:6,
        5,
        ifelse(
          status_vec == 3,
          6,
          ifelse(
            status_vec == 4 & decay_vec %in% 1:2,
            3,
            ifelse(
              status_vec == 4 & decay_vec %in% 3:4,
              4,
              ifelse(
                status_vec == 4 & decay_vec %in% 5:6,
                5,
                ifelse(
                  status_vec == 4 & decay_vec == 7,
                  8,
                  ifelse(status_vec == 6, 7, ifelse(status_vec == 5, 7, NA))
                )
              )
            )
          )
        )
      )
    )
  )
}