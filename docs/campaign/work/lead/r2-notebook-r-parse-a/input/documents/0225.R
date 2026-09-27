resamplingConstConst<- function(modelslBF, positions){
	#Given the matrix created within BvsF containing a binary expression of models and the log(BF) in the last column obtained 
	#with GibbsBvs(...prior.models = "ConstConst"...)
	#(meaning that prior over the model space is obtained with the hierarchical
	#Pr(Mg)=Pr(Mg|those factors and x's)*Pr(those factors and x's)
	#and Pr(Mg|those factors and x's)\proto 1/(number of models with those factors and x's)
	#and Pr(those factors and x's)\propto 1/(number of models))
	#the function makes a resampling with the new prior being the same but only
	#keeping unique models (in a same class the full is kept and the others are not)
	#returning a matrix of the same size with the resampled models
	levelsfull<- rowSums(positions)
	levelsf<- levelsfull[levelsfull>1]
	kplusp<- dim(positions)[1]
	#A matrix containing, for each sampled models, the number of active "levels" for each regressor:
	if (kplusp==1)
		m.actlevels<- matrix(apply(modelslBF[,-dim(modelslBF)[2]], MARGIN=1, FUN=function(x){positions%*%x}), ncol=1)
	else
		m.actlevels<- t(apply(modelslBF[,-dim(modelslBF)[2]], MARGIN=1, FUN=function(x){positions%*%x}))
	#for each sampled model obtain the ConstConst2 prior prob:
	allmodelsprior2<- t(apply(m.actlevels, MARGIN=1, FUN=priorConstConst2, levelsfull=levelsfull, levelsf=levelsf, kplusp=kplusp))
	#for each sampled model obtain the SBSB1 prior prob:
	allmodelsprior1<- t(apply(m.actlevels, MARGIN=1, FUN=priorConstConst1, kplusp=kplusp, levelsfull=levelsfull))
	#now the resampling:
	resamp<- sample(x=1:dim(modelslBF)[1], size=dim(modelslBF)[1], replace=T, prob=exp(allmodelsprior2-allmodelsprior1))
	return(modelslBF[resamp,])	
}