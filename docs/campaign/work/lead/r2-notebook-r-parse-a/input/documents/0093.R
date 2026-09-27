CorShift <- function (samplesA = samplesGroupA, samplesB = samplesGroupB, shift_threshold = 0.6, writeTable = FALSE, min_cor_in_samples = 0.6, cor_signif = 0.01, quant_data_all_local = quant_data_all, int_pairs_local = int_pairs) {

#globalVariables(c("quant_data_all", "samplesGroupA", "samplesGroupB", "int_pairs"))
quant_samplesA = quant_data_all_local[, samplesA]
quant_samplesB = quant_data_all_local[, samplesB]

measured_proteins = row.names (quant_data_all_local)
protA_ok <- int_pairs_local$ProtA %in% measured_proteins
protB_ok <- int_pairs_local$ProtB %in% measured_proteins
ok_elements = protA_ok & protB_ok
int_pairs_ok <- int_pairs_local[ok_elements,c(1,2)]

cor_table = as.data.frame(matrix(nrow = 0, ncol = 7))

n_pairs_ok = nrow (int_pairs_ok)
#print ("Comparing individual correlations...")
row_nu = 1
for (r in 1:n_pairs_ok)
{
    prot1 = int_pairs_ok [r,1]
    prot2 = int_pairs_ok [r,2]
    pair_name = paste (prot1, prot2, sep = "-")

    nas_protA_samplesA = is.na (as.numeric(quant_samplesA[prot1, ]))
    nas_protB_samplesA = is.na (as.numeric(quant_samplesA[prot2, ]))
    nas_protA_samplesB = is.na (as.numeric(quant_samplesB[prot1, ]))
    nas_protB_samplesB = is.na (as.numeric(quant_samplesB[prot2, ]))

    notNA_samplesA = !(nas_protA_samplesA | nas_protB_samplesA)
    notNA_samplesB = !(nas_protA_samplesB | nas_protB_samplesB)
    measured_in_samplesA = length (notNA_samplesA [as.numeric(notNA_samplesA) > 0])
    measured_in_samplesB = length (notNA_samplesB [as.numeric(notNA_samplesB) > 0])

    if ((measured_in_samplesA > 3) & (measured_in_samplesB > 3)) {
        cor_A = cor (as.numeric(quant_samplesA[prot1, ]), as.numeric(quant_samplesA[prot2, ]), use = "pairwise.complete.obs")
        cor_B = cor (as.numeric(quant_samplesB[prot1, ]), as.numeric(quant_samplesB[prot2, ]), use = "pairwise.complete.obs")
        cor_A_p = cor.test (as.numeric(quant_samplesA[prot1, ]), as.numeric(quant_samplesA[prot2, ]))
        cor_B_p = cor.test (as.numeric(quant_samplesB[prot1, ]), as.numeric(quant_samplesB[prot2, ]))

        if (((cor_A > min_cor_in_samples) && (cor_A_p$p.value < cor_signif)) || ((cor_B > min_cor_in_samples) && (cor_B_p$p.value < cor_signif))) {
            #jao = paste (pair_name, cor_A, cor_B, cor_A_p$p.value, cor_B_p$p.value, sep = " ")
            #print (jao)
            cor_table[row_nu, 1] = round (cor_A, 2)
            cor_table[row_nu, 2] = round (cor_A_p$p.value, 4)
            cor_table[row_nu, 3] = round (cor_B, 2)
            cor_table[row_nu, 4] = round (cor_B_p$p.value, 4)
            cor_table[row_nu, 5] = measured_in_samplesA
            cor_table[row_nu, 6] = measured_in_samplesB
            row.names (cor_table)[row_nu] = pair_name
            row_nu = row_nu + 1
        }
    }
}

diff_pears = cor_table[,3] - cor_table[,1]
cor_table[,7] = round (diff_pears, 2)

cor_table = cor_table[abs(cor_table[,7]) > shift_threshold,]
cor_table = cor_table [order (cor_table[,3], decreasing = T), ]
colnames(cor_table) <- c('Pearson_cor_samplesA', 'cor_p_value', 'Pearson_cor_samplesB', 'cor_p_value', 'NumberOfSamplesA', 'NumberOfSamplesB', 'Correlation_shift')
if (writeTable) {
    write.table (cor_table, file = "protCorrelationShift.txt", col.names = TRUE, row.names = TRUE, sep = "\t", quote = FALSE)
}
return (cor_table)
}