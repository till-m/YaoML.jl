# YaoML
Quantum machine learning in Julia using Yao.jl

This package aims to implement popular QML algorithms such as QSVMs ("Quantum Support Vector Machines") and VQCs ("Variational Quantum Circuit", also known as "Quantum Neural Network").

## Usage

### QSVM
The QSVM uses a circuit architecture first described in [[1]](#1).
![alt text](/resources/quantum_kernel.png)

```julia
using YaoML

# Load the built-in example dataset
X, y = ad_hoc_data(40, 0.5, seed=42)

# Partition into train/test sets 
train, test = partition(eachindex(y), 0.75, seed=42+42, shuffle=true)

# Select the feature map U(x)
feature_map = FeatureMaps.zz_feature_map

# Train the model
model = qsvmtrain(feature_map, X[train, :], y[train])

# Use the trained model to predict on new values
ŷ, decision_values = qsvmpredict(model, X[test,:])
```

### VQC
```julia
import Yao
using YaoML

# Generate the example dataset
X, y = ad_hoc_data(40, 0.5, seed=42)

# Partition into train/test sets 
train, test = partition(eachindex(y), 0.75, seed=42+42, shuffle=true)

# Build the circuit
feature_map = FeatureMaps.zz_feature_map
variational_map = FeatureMaps.two_local_demo

# First argument of `circuit` is data, second argument will be optimized over
circuit = (x, theta) -> Yao.chain(feature_map(x), variational_map(theta))

# Initialize the VQC
vqc = VQC(circuit, [0 for i=1:16], 0.01)

# train
vqctrain!(vqc, X[train, :], y[train])

# predict
ŷ = vqcpredict(vqc, X[test,:])
```


## References
<a id="1">[1]</a> 
Havlíček, V., Córcoles, A.D., Temme, K. et al.;
*Supervised learning with quantum-enhanced feature spaces*;
Nature 567, 209–212 (2019). https://doi.org/10.1038/s41586-019-0980-2
