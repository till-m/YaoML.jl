module YaoML

include("QSVM.jl")
export qsvmtrain, qsvmpredict
include("VQC.jl")
export VQC, vqctrain!, vqcpredict, vqcpredict_proba
include("datasets.jl")
export ad_hoc_data, partition
include("FeatureMaps.jl")
export FeatureMaps

end # module
