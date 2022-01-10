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

function two_local_demo(theta)
    # mirrors https://qiskit.org/documentation/stable/0.24/tutorials/machine_learning/03_vqc.html
    qc = nothing
    for i=1:4
        ry = Yao.kron(Yao.Ry(theta[4*i-3]), Yao.Ry(theta[4*i-2]))
        rz = Yao.kron(Yao.Rz(theta[4*i-1]), Yao.Ry(theta[4*i]))
        
        if i==1
            qc = Yao.chain(ry, rz, Yao.cnot(1,2))
        elseif i ==4
            qc = Yao.chain(qc, ry, rz)
        else
            qc = Yao.chain(qc, ry, rz, Yao.cnot(1,2))
        end
    end
    return qc
end

end # module