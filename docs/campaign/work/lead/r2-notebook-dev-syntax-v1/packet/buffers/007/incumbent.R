#' Treatment Comparison Protocol for One-Sided Tests
#'
#' Generates a randomized layout for the Treatment Comparison Protocol for one-sided tests.
#'
#' @param s Number of test treatments (must be at least 3).
#' @param control Number of control replicates (non-negative integer).
#'
#' @return A list containing:
#'   \itemize{
#'     \item Parameters: A data frame with protocol details.
#'     \item NonRandomized_Layout: A list of data frames showing the non-randomized layout.
#'     \item Randomized_Layout: A list of data frames showing the randomized layout.
#'   }
#'
#' @keywords internal
#' @noRd
TCpRep1 <- function(s, control) {
  # Input Validation
  if (s < 3) {
    stop("s must be at least 3")
  }
  if (control < 0 || control != as.integer(control)) {
    stop("'control' must be a non-negative integer")
  }

  # Construct MOLS
  mols <- vector("list", s - 1)

  for (m in 1:(s - 1)) {
    L <- matrix(0, s, s)

    for (i in 0:(s - 1)) {
      for (j in 0:(s - 1)) {
        L[i + 1, j + 1] <- (i + m * j) %% s
      }
    }

    mols[[m]] <- L
  }

  # Treatment Array
  Tmat <- matrix(1:(s^2), nrow = s, byrow = TRUE)

  # Parallel Classes
  P <- list()

  P[[1]] <- split(
    as.vector(t(Tmat)),
    rep(1:s, each = s)
  )

  P[[2]] <- lapply(
    1:s,
    function(j) Tmat[, j]
  )

  pc <- 3

  for (m in 1:(s - 1)) {
    blocks <- vector("list", s)

    for (a in 0:(s - 1)) {
      pos <- which(
        mols[[m]] == a,
        arr.ind = TRUE
      )

      blocks[[a + 1]] <- Tmat[pos]
    }

    P[[pc]] <- blocks
    pc <- pc + 1
  }

  # Environment Formation
  environments <- list()

  for (i in 1:s) {
    env <- P[[i]]

    env[[s + 1]] <- P[[s + 1]][[i]]

    env.mat <- do.call(
      rbind,
      lapply(env, as.vector)
    )

    rownames(env.mat) <- paste0("B", 1:(s + 1))

    environments[[i]] <- env.mat
  }

  names(environments) <- paste0(
    "Environment_",
    1:s
  )

  # Add Controls
  Design_TC <- lapply(
    environments,
    function(env) {
      env <- as.matrix(env)

      if (control > 0) {
        control_mat <- matrix(
          rep(
            paste0("C", 1:control),
            each = nrow(env)
          ),
          nrow = nrow(env),
          ncol = control,
          byrow = FALSE
        )

        colnames(control_mat) <- paste0(
          "C",
          1:control
        )

        env <- cbind(
          env,
          control_mat
        )
      }

      env
    }
  )

  # Randomized Layout
  Randomized_TC <- lapply(
    Design_TC,
    function(env) {
      rnd <- t(apply(env, 1, sample))
      rownames(rnd) <- rownames(env)
      rnd
    }
  )

  # Parameters
  Parameters <- data.frame(
    Parameter = c(
      "Number of Test Treatments (vt)",
      "Number of Controls (vc)",
      "Number of Environments (e)",
      "Number of Blocks (b)",
      "Test Replication (rt)",
      "Control Replication (rc)",
      "Block Size (k)"
    ),
    Value = c(
      s^2,
      control,
      s,
      s * (s + 1),
      s + 1,
      if (control > 0) s * (s + 1) else 0,
      s + control
    )
  )

  # Non-Randomized Layout
  NonRandomized_Layout <- lapply(
    Design_TC,
    function(env) {
      data.frame(
        Block = rownames(env),
        Lines = apply(
          env,
          1,
          function(x) paste(x, collapse = " ")
        ),
        row.names = NULL,
        stringsAsFactors = FALSE
      )
    }
  )

  # Randomized Layout
  Randomized_Layout <- lapply(
    Randomized_TC,
    function(env) {
      data.frame(
        Block = rownames(env),
        Lines = apply(
          env,
          1,
          function(x) paste(x, collapse = " ")
        ),
        row.names = NULL,
        stringsAsFactors = FALSE
      )
    }
  )

  # Output
  return(
    structure(
      list(
        Parameters = Parameters,
        NonRandomized_Layout = NonRandomized_Layout,
        Randomized_Layout = Randomized_Layout
      ),
      class = "TCpRep"
    )
  )
}