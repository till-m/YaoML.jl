using Test

import Yao
using YaoML

# load data
X, y = ad_hoc_data(40, 0.5, seed=42)
train, test = partition(eachindex(y), 0.75, seed=42+42, shuffle=true)

# QSVM
feature_map = FeatureMaps.zz_feature_map
model = qsvmtrain(feature_map, X[train, :], y[train])

ŷ, decision_values = qsvmpredict(model, X[test,:])

@test ŷ == y[test]

# VQC
variational_map = FeatureMaps.two_local_demo
vqc = VQC((x, theta) -> Yao.chain(feature_map(x), variational_map(theta)), [0 for i=1:16], 0.01)

vqctrain!(vqc, X[train, :], y[train])

ŷ = vqcpredict(vqc, X[test,:])

@test ŷ == y[test]