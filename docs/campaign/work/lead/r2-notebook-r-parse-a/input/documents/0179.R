sig_correlation2<-function(input_edgeD,padj=TRUE,method="BH", verb){
  corr<-rcorr(as.matrix(input_edgeD),type='p')
  corr_table<-flattenCorrMatrix(corr$r,corr$P)
  corr_table$adj.p<-p.adjust(corr_table$p,method = method)

  if(padj){
    corr_tableSig <- corr_table %>% filter(.data$adj.p<0.05)
    if(nrow(corr_tableSig) == 0){
      print('No features significantly correlate after padjustment for vertexD')
      print('Using non adjusted pvalues')
      corr_tableSig<-corr_table %>% filter(.data$p<0.05)}
  }else{
    print("Ajustment for multiple testing was set to FALSE for correlations in vertex Data")
    corr_tableSig<-corr_table %>% filter(.data$p<0.05)
  }
  if(verb){print(paste('Significant correlations',nrow(corr_tableSig),sep=" "))}
  corr_tableSig
}