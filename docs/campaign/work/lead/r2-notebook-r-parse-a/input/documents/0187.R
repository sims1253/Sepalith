topic_init_baseline <- function(rec_data, ds_list, topic_num) {
  # arrange the data stack by individual is very important as we need to make sure the matrix could be rejoined into a single matrix
  # first_incidence_age <- rec_data %>%
  #   arrange(eid) %>%
  #   group_by(eid, diag_icd10) %>%
  #   arrange(age_diag, .by_group = T) %>% # keep only the first records of repeated diagnosis
  #   dplyr::slice(1) %>%
  #   dplyr::ungroup()

  message(
    "remove duplicated diagnoses; keep the earliest age-at-diagnosis when mulitple diagnoses are presented."
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
  para$list_above500occu <- ds_list
  para$D <- dim(para$list_above500occu)[1] # disease number
  para$M <- length(para$eid) # subject number
  para$K <- topic_num # start with 10 component
  para$Ns <- df_number_records$`n()`

  code2id <- function(x) {
    return(match(x, para$list_above500occu$diag_icd10))
  }

  # here I am rounding the disease time to year for computation efficiency
  para$unlist_Ds_id <- first_incidence_age %>%
    mutate(Ds_id = code2id(diag_icd10)) %>%
    select(-diag_icd10) %>%
    mutate(age_diag = round(age_diag))

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

  # this list is splitted by disease
  # para$disease_id_list <- para$unlist_Ds_id %>%
  #   select(-eid) %>%
  #   mutate(id = dplyr::row_number()) %>%
  #   group_by(Ds_id) %>%
  #   dplyr::group_split()
  # making this slightly faster
  para$disease_id_list <- split(
    para$unlist_Ds_id$row_id,
    para$unlist_Ds_id$Ds_id
  )

  # initiate beta
  para$eta <- rgamma(para$D, shape = 100, rate = 100)
  # each column is a topic; D*K matrix
  para$beta <- t(gtools::rdirichlet(para$K, para$eta))

  # initiate alpha
  para$alpha <- rgamma(para$K, shape = 50, rate = 10)

  # this beta_w parameter save the beta for each word: it is a list of M elements and each contain a K*Ns matrix
  para$beta_w_full <- para$beta[para$unlist_Ds_id$Ds_id, , drop = FALSE]
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