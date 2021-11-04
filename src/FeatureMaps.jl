module FeatureMaps

import Yao

""" zz_feature_map(x)
Implements a Pauli-Z evolution circuit on two qubits with two repetitions.

Analogous to qiskit.circuit.library.ZZFeatureMap(2, reps=2)
"""
function zz_feature_map(x)
    hadamards = Yao.kron(Yao.H, Yao.H)
    enc = Yao.chain(hadamards, Yao.kron(Yao.shift(2*x[1]), Yao.shift(2*x[2])))
    ent = Yao.chain(Yao.cnot(1,2),
        Yao.put(2 => Yao.shift(2* (π-x[1])*(π-x[2]))),
        Yao.cnot(1,2))
    qc = Yao.chain(enc, ent, enc, ent)
    return qc
end

end # module