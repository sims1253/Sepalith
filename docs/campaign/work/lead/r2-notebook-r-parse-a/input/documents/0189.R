loading2weights <- function(
  data,
  ds_list = UKB_349_disease,
  topics = UKB_HES_10topics
) {
  profile_size <- dim(topics)
  message(
    paste0(
      "The age profiles contain ",
      profile_size[3],
      " disease topics of ",
      profile_size[2],
      " diseases up to age ",
      profile_size[1]
    ),
    "."
  )

  data_not_in_disease_list <- data %>%
    filter(!(diag_icd10 %in% ds_list$diag_icd10)) %>%
    dim()
  message(paste0(
    data_not_in_disease_list[1],
    " records are not in the ds_list, if this number is high it means
                 many of the records is not covered by the comorbidity profiles."
  ))

  # if age is not in the range
  size_outrange <- data %>%
    filter((age_diag <= 0) | (age_diag > profile_size[1])) %>%
    dim
  message(paste0(
    size_outrange[1],
    " records have age outside the disease range, they will be thrown away,
                 considering using the age_imputation function."
  ))
  # data filtering
  data <- data %>%
    filter(diag_icd10 %in% ds_list$diag_icd10) %>%
    filter(age_diag > 0, age_diag <= profile_size[1])
  if (dim(data)[1] == 0) {
    stop(
      "No records that are covered by the profile
         It might be the age information is missing: either use age_imputation function or
         just assign the age_diag to a fixed number such as 50."
    )
  }

  para <- topic_init_age(
    data,
    ds_list,
    dim(topics)[length(dim(topics))],
    degree_free_num = 5
  ) # for internal note: degree_free_num doesn't really matter in this case
  # update beta_w: list of Ns-by-K
  para$beta_w_full <- apply(topics, 3, function(x) {
    x[as.matrix(select(para$unlist_Ds_id, age_diag, Ds_id))]
  })
  para$beta_w <- lapply(para$patient_lst, function(x) {
    para$beta_w_full[x, , drop = F]
  })

  # update z_n until convergence
  para$max_itr <- 50
  para$lb <- data.frame("Iteration" = 0, "Lower_bound" = CVB_lb(para))
  para$tol <- 10^(-7)
  for (itr in 1:para$max_itr) {
    message(paste0("Interation: ", itr))
    para <- CVB0_E_zn(para) # we choose CVB0 as papers shown it could converge quicker
    para$lb[nrow(para$lb) + 1, ] <- c(itr, CVB_lb(para))
    curr_lb <- pull(filter(para$lb, Iteration == itr), Lower_bound)
    prev_lb <- pull(filter(para$lb, Iteration == (itr - 1)), Lower_bound)
    message(paste0("Current Lower bound ", curr_lb, " at iteration: ", itr))
    try({
      if (
        is.finite((curr_lb - prev_lb)) &
          abs(curr_lb - prev_lb) / abs(prev_lb) < para$tol
      ) {
        message(paste0("Optimization converged at step ", itr))
        break
      }
    })
  }
  new_weights <- sweep(
    (para$alpha_z - 1),
    1,
    rowSums(para$alpha_z - 1),
    FUN = "/"
  )
  weights_results <- list()
  weights_results$topic_weights <- data.frame(
    eid = para$eid,
    topic_weights = new_weights
  )
  weights_results$incidence_weight_sum <- data.frame(
    eid = para$eid,
    incidence_weight_sum = (para$alpha_z - 1)
  )
  return(weights_results)
}