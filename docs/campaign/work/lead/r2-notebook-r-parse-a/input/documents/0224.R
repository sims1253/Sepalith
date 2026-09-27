resamplingSBSB<- function(modelslBF, positions){
	#Given the matrix created within BvsF containing a binary expression of models and the log(BF) in the last columnbtained 
	#with GibbsBvs(...prior.models = "SBSB"...)
	#(meaning that prior over the model space is obtained with the hierarchical
	#Pr(Mg)=Pr(Mg|those factors and x's)*Pr(those factors and x's)
	#and Pr(Mg|those factors and x's)\proto 1/(number of models of that dimension with those factors and x's)
	#and Pr(those factors and x's)\propto 1/(number of models of that dimension))
	#the function makes a resampling with the new priolevelsfullr being
	#Pr(Mg|those factors and x's)\propto 1/(number of different models of that rank with those factors and x's)
	#returning a matrix of the same size with the resampled models
	levelsfull<- rowSums(positions)
	levelsf<- levelsfull[levelsfull>1]

	kplusp<- dim(positions)[1]
	#A matrix containing, for each sampled models, the number of active "levels" for each regressor:
	if (kplusp==1)
		m.actlevels<- matrix(apply(modelslBF[,-dim(modelslBF)[2]], MARGIN=1, FUN=function(x){positions%*%x}), ncol=1)
	else
		m.actlevels<- t(apply(modelslBF[,-dim(modelslBF)[2]], MARGIN=1, FUN=function(x){positions%*%x}))
	#for each sampled model obtain the SBSB2 prior prob:
	allmodelsprior2<- t(apply(m.actlevels, MARGIN=1, FUN=priorSBSB2, levelsfull=levelsfull, levelsf=levelsf, kplusp=kplusp))
	#for each sampled model obtain the SBSB1 prior prob:
	allmodelsprior1<- t(apply(m.actlevels, MARGIN=1, FUN=priorSBSB1, levelsfull=levelsfull, kplusp=kplusp))
	#now the resampling:
	resamp<- sample(x=1:dim(modelslBF)[1], size=dim(modelslBF)[1], replace=T, prob=exp(allmodelsprior2-allmodelsprior1))
	return(modelslBF[resamp,])	
}