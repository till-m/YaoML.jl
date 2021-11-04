module YaoML

include("QSVM.jl")
export qsvmtrain, qsvmpredict
include("datasets.jl")
export ad_hoc_data

end # module
