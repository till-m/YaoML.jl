import Yao
import LIBSVM

function make_kernel(feature_map::Function; nshots::Int=1024)
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


function make_sparse_gram(kernel::Function, T::Matrix, X::Matrix, ntrain::Int, sv_indices)
    # Calculates only the values of the gram matrix for support vectors
    # but makes sure to return a matrix of appropriate shape.
    gram_sparse = make_gram(kernel::Function, T::Matrix, X::Matrix)
    gram = similar(gram_sparse, ntrain, size(T, 1))
    gram[sv_indices, :] = gram_sparse
    return gram
end


struct QSVM{T}
    SVM::LIBSVM.SVM{T}
    feature_map::Function
    kernel_function::Function
    # needed since LIBSVM.jl only saves SVs indices for precomputed kernels:
    support_vectors::Matrix
    # needed until LIBSVM.jl fixes that says precomputed kernels have one feature only
    nfeatures::Int
end

function qsvmtrain(
    feature_map::Function, X::AbstractMatrix{U}, y::AbstractVector{T} = [];
    nshots::Int=1024,
    svmtype::Type = LIBSVM.SVC,
    #kernel::Kernel.KERNEL = Kernel.RadialBasis,
    #degree::Integer = 3,
    #gamma::Float64 = 1.0 / size(X, 1),
    #coef0::Float64 = 0.0,
    cost::Float64 = 1.0,
    nu::Float64 = 0.5,
    epsilon::Float64 = 0.1,
    tolerance::Float64 = 0.001,
    shrinking::Bool = true,
    probability::Bool = false,
    weights::Union{Dict{T,Float64},Cvoid} = nothing,
    cachesize::Float64 = 200.0,
    verbose::Bool = false,
    nt::Integer = 1) where {T,U<:Real}

    kernel_function = make_kernel(feature_map, nshots=nshots)
    K = make_gram(kernel_function, X)

    model = LIBSVM.svmtrain(K, y, kernel=LIBSVM.Kernel.Precomputed,
        svmtype=svmtype,
        cost=cost,
        nu=nu,
        epsilon=epsilon,
        tolerance=tolerance,
        shrinking=shrinking,
        probability=probability,
        weights=weights,
        cachesize=cachesize,
        verbose=verbose,
        nt=nt)

    return QSVM(model, feature_map, kernel_function, X[model.SVs.indices,:], size(X, 1))
end

function qsvmpredict(model::QSVM{T}, X::AbstractMatrix{U}; nt::Integer = 0) where {T,U<:Real}
    K = make_sparse_gram(model.kernel_function, X, model.support_vectors, model.nfeatures, model.SVM.SVs.indices)
    return LIBSVM.svmpredict(model.SVM, K, nt=nt)
end
