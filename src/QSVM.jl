import Yao
import LIBSVM

function make_kernel(feature_map, nshots::Int=1024)
    function kernel(x1, x2)
        qc = Yao.chain(feature_map(x1), Yao.adjoint(feature_map(x2)))
        n_qubits = Yao.nqubits(qc)
        r = Yao.zero_state(n_qubits)
        result =  r |> qc |> r-> Yao.measure(r, nshots=nshots)
        return sum(result.==0)/length(result)   
    end
    return kernel
end


function make_gram(kernel::Function, X::Matrix)
    # X should be a matrix containing datapoints in the rows
    n1 = axes(X)[1]
    res = Array{Float64}(undef, n1.stop, n1.stop)
    for i in n1
        for j in 1:i
            if i==j
                res[i,i] = 1 # <phi_i|phi_i> = 1 due to normalization
            else
                interm = kernel(X[i,:], X[j,:])
                res[i,j] = interm
                res[j,i] = interm
            end
        end
    end
    return res
end


function make_gram(kernel::Function, T::Matrix, X::Matrix)
    # X, T should be a matrix containing datapoints in the rows
    # i.e. (n,d) --> n datapoints of dim d each.
    n1 = axes(X)[1]
    n2 = axes(T)[1]
    res = Array{Float64}(undef, n1.stop, n2.stop)
    for i in n1
        for j in n2
            res[i,j] = kernel(T[j,:], X[i,:])
        end
    end
    return res
end


function make_gram(kernel::Function, T::Matrix, X::Matrix, SVs::Vector{Int})
    # If the support vectors are supplied we can make sure to calculate only
    # the entries that are actually used.
    # X, T should be a matrix containing datapoints in the rows
    # i.e. (n,d) --> n datapoints of dim d each.
    n1 = axes(X)[1]
    n2 = axes(T)[1]
    res = Array{Float64}(undef, n1.stop, n2.stop)

    for i in SVs
        for j in n2
            res[i,j] = kernel(T[j,:], X[i,:])
        end
    end
    return res
end
