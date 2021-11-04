using Test

using YaoML

X, y = ad_hoc_data(40, 0.5, seed=42)
train, test = partition(eachindex(y), 0.75, seed=42+42, shuffle=true)

feature_map = FeatureMaps.zz_feature_map
model = qsvmtrain(feature_map, X[train, :], y[train])

ŷ, decision_values = qsvmpredict(model, X[test,:])

@test ŷ == y[test]
