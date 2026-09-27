topic_init_age <- function(rec_data, ds_list, topic_num, degree_free_num) {
  # arrange the data stack by individual is very important as we need to make sure the matrix could be rejoined into a single matrix
  # first_incidence_age <- rec_data %>%
  #   mutate(diag_icd10 = as.character(diag_icd10)) %>% # seem to fix the bug
  #   arrange(eid) %>%
  #   group_by(eid, diag_icd10) %>%
  #   filter(n() == 1 | age_diag == min(age_diag) ) %>% # this row is highly optimized, a lot faster the slice_min ### don't change
  #   dplyr::slice(1) %>%
  #   # arrange(age_diag, .by_group = T) %>% # keep only the first records of repeated diagnosis
  #   dplyr::ungroup()

  # use data.table to make filtering faster
  message(
    "remove duplicated diagnosis; keep the earliest age-at-diagnosis when mulitple diagnosis are presented."
  )
  data.table::setDT(rec_data)
  # Fast + simple: sort once, keep first row per group
  data.table::setorder(rec_data, eid, diag_icd10, age_diag)
  first_incidence_age <- unique(rec_data, by = c("eid", "diag_icd10"))

  # plot the number distribution of indiviudal diseases
  df_number_records <- first_incidence_age %>%
    group_by(eid) %>%
    summarise(n())

  para <- list()

  para$eid <- df_number_records$eid
  # add death to it
  para$list_above500occu <- ds_list
  para$D <- dim(para$list_above500occu)[1] # disease number
  para$M <- length(para$eid) # subject number
  para$K <- topic_num # start with 10 component
  para$P <- degree_free_num # degrees of freedom
  # also need to compute record number per individual for computing lower bound of cvb
  para$Ns <- df_number_records$`n()`

  code2id <- function(x) {
    return(match(x, para$list_above500occu$diag_icd10))
  }

  # here I am rounding the disease time to year for computation efficiency
  para$unlist_Ds_id <- first_incidence_age %>%
    mutate(Ds_id = code2id(diag_icd10)) %>%
    select(-diag_icd10) %>%
    mutate(age_diag = ceiling(age_diag))
  # the patient_list provide the column index for efficiently breaking down matrix into list of matrices
  # para$patient_lst <- para$unlist_Ds_id %>%
  #   mutate(id = dplyr::row_number()) %>%
  #   select(eid, id) %>%
  #   group_by(eid) %>%
  #   dplyr::group_split(.keep = F) %>%
  #   lapply(pull)

  # make it slightly faster
  data.table::setDT(para$unlist_Ds_id)
  data.table::set(
    para$unlist_Ds_id,
    j = "row_id",
    value = seq_len(nrow(para$unlist_Ds_id))
  )
  para$patient_lst <- split(para$unlist_Ds_id$row_id, para$unlist_Ds_id$eid)

  para$w <- para$unlist_Ds_id %>%
    group_by(eid) %>%
    dplyr::group_split(.keep = F)

  # this list is column ID for each disease
  # para$disease_id_list <- para$unlist_Ds_id %>%
  #   select(-eid) %>%
  #   mutate(id = dplyr::row_number()) %>%
  #   group_by(Ds_id) %>%
  #   dplyr::group_split()
  para$disease_id_list <- split(
    para$unlist_Ds_id$row_id,
    para$unlist_Ds_id$Ds_id
  )

  # create an age matrix for each disease, it is a list same length as para$w, each of Ns-by-F matrix
  para$age_max <- first_incidence_age %>%
    summarise(max(age_diag)) %>%
    pull
  para$age_min <- first_incidence_age %>%
    summarise(min(age_diag)) %>%
    pull

  # I scale the magnitude of age to 1 (divided by para$age_max) to avoid numeric infinity in exponentials
  # basis is for a age grid
  para$age_basis <- age_basis_spline(
    para$P,
    (1:ceiling(para$age_max)) / ceiling(para$age_max),
    para$age_min / para$age_max,
    para$age_max / para$age_max
  )
  # below is the set of age basis that are used for discrete computation
  para$age_basis_discrte <- para$age_basis[
    min(para$unlist_Ds_id$age_diag):max(para$unlist_Ds_id$age_diag),
  ]
  para$basis_phi <- para$age_basis[para$unlist_Ds_id$age_diag, , drop = F]

  # each column is a topic; D*K matrix
  para$beta <- array(
    rnorm(para$P * para$D * para$K, sd = 0.1),
    dim = c(para$P, para$D, para$K)
  )

  # initiate alpha, later we will set it to 1: here it is for random initialization of E_lntheta
  para$alpha <- rgamma(para$K, shape = 50, rate = 10)

  # first compute the whole age_beta_basis, for each topic it is T-by-D
  para$exp_age_beta_basis <- array(
    sapply(1:para$K, function(i) exp(para$age_basis %*% para$beta[,, i])),
    dim = c(dim(para$age_basis)[1], para$D, para$K)
  )
  para$sum_exp_age_beta_basis <- apply(para$exp_age_beta_basis, c(1, 3), sum)
  # this is the basis of softmax function
  para$pi_beta_basis <- apply(para$exp_age_beta_basis, c(1, 3), function(x) {
    x / sum(x)
  }) %>%
    aperm(perm = c(2, 1, 3))
  # this beta_w parameter save the beta for each word: it is a list of M elements and each contain a K*Ns matrix
  para$beta_w_full <- apply(para$pi_beta_basis, 3, function(x) {
    x[as.matrix(select(para$unlist_Ds_id, age_diag, Ds_id))]
  })
  para$beta_w <- lapply(para$patient_lst, function(x) {
    para$beta_w_full[x, , drop = F]
  })

  # Matrix of M*K
  para$E_lntheta <- t(sapply(
    rgamma(para$M, shape = 100, rate = 100),
    function(x) x * (digamma(para$alpha) - digamma(sum(para$alpha)))
  ))

  # update E_zn: list of M; each element is matrix of Ns*K
  para <- comp_E_zn(para)

  # eventually reset alpha to be non-informative
  para$alpha <- rep(1, para$K)

  message(
    "Rough RAM needed: ",
    format(4 * utils::object.size(para), units = "GB", standard = "SI")
  )

  return(para)
}