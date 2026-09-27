#' @title Create a class for a status-decay pair
#' @description
#' Given a status and decay vector, return a class for that pair.
#' @param status_vec A vector of length 2 with the status values.
#' @param decay_vec A vector of length 2 with the decay values.
#' @return A class for the status-decay pair.
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