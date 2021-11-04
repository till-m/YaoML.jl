import Random
import LinearAlgebra

function partition(indices::Base.OneTo{Int64}, fraction::Float64;
    shuffle::Bool=false,
    seed::Union{Int, Nothing}=nothing)

    if shuffle && !isnothing(seed)
        Random.seed!(seed)
    end

    n_1 = Int(fraction * indices.stop)
    if shuffle
        shuffled = Random.shuffle(1:indices.stop)
        return indices[shuffled[1:n_1]], indices[shuffled[(n_1+1):end]]
    else
        return indices[1:n_1], indices[(n_1+1):end]
    end
end

""" 
    ad_hoc_data(
        n::Int, gap::Float;
        seed::Union{Int, Nothing}=nothing,
        shuffle::Bool=false
    )
Generates data similar to the `qiskit_machine_learning.datasets.ad_hoc_data` function. This is mostly a
(manual) transpilation of Qiskit's function with fewer options (e.g. only dim=2 is possible).
# Arguments
* `n::Int`: How many datapoints to generate for each class.
* `gap::Float`: The size of the gap.
* `seed::Union{Int, Nothing} = nothing`: The seed for the random generator.
* `shuffle::Bool = false`: Whether to shuffle the output. If false, the output will be ordered by class.
"""
function ad_hoc_data(n::Int, gap::Float64; seed::Union{Int, Nothing}=nothing, shuffle::Bool=false)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    dim = 2
    count = 100
    

    sample_a = zeros(Float64, n, dim)
    sample_b = zeros(Float64, n, dim)

    sample_total = zeros(Float64, count, count)

    steps = 2 * π / count

    z_m = [1 0; 0 -1] 
    
    j_m = [1 0; 0 1]
    h_m = [1 1; 1 -1] / sqrt(2)
    h_2 = kron(h_m, h_m)

    f_a = Array([i for i=0:(2^dim-1)])
    my_array = zeros(Int, 2^dim, dim)

    for idx=1:size(my_array,1)
        temp_f = string(f_a[idx], base=2)
        temp_f = lpad(temp_f, 2, '0')
        for findex=1:dim
            my_array[idx,findex] = parse(Int,temp_f[findex])
        end
    end

    my_array = transpose(my_array)

    parity = (-1).^(sum(my_array, dims=1)[1,:])
    d_m = LinearAlgebra.Diagonal(parity)
    
    basis = Random.rand(Float64, (4, 4)) + 1im * Random.rand(Float64, (4, 4))
    basis = basis' * basis

    s_a = LinearAlgebra.eigvals(basis)
    u_a = LinearAlgebra.eigvecs(basis)
    
    idx = reverse(sortperm(s_a))
    s_a = s_a[idx]
    u_a = u_a[idx, :]

    m_m = u_a' * d_m * u_a

    psi_plus = transpose(ones(2)) / sqrt(2)
    psi_0 = kron(psi_plus, psi_plus)

    
    for n_1=1:count
        for n_2=1:count
            x_1 = steps * n_1
            x_2 = steps * n_2
            phi = x_1 * kron(z_m, j_m) + x_2 * kron(j_m, z_m) + (π - x_1) * (π -x_2) * kron(z_m, z_m)
            u_u = exp(1im * phi)
            psi = u_u * h_2 * u_u * transpose(psi_0)
            
            temp = real(psi' * m_m * psi)
            if temp > gap
                sample_total[n_1, n_2] = +1
            elseif temp < -gap
                sample_total[n_1, n_2] = -1
            else
                sample_total[n_1, n_2] = 0
            end
        end
    end

    # Now sample randomly from sample_total
    t_r = 1
    t_max = 0
    while t_r <= n && t_max < 1000
        draw1 = Random.rand(1:count)
        draw2 = Random.rand(1:count)
        if sample_total[draw1, draw2] == 1
            sample_a[t_r, :] = [2 * π * draw1 / count, 2 * π * draw2 / count]
            t_r += 1
        else
            t_max +=1
        end
    end

    t_r = 1
    t_max = 0
    while t_r <= n && t_max < 1000
        draw1 = Random.rand(1:count)
        draw2 = Random.rand(1:count)
        if sample_total[draw1, draw2] == -1
            sample_b[t_r, :] = [2 * π * draw1 / count, 2 * π * draw2 / count]
            t_r += 1
        else
            t_max +=1
        end
    end

    y = vcat(zeros(n), ones(n))
    samples = vcat(sample_a, sample_b)
    if shuffle
        perm = Random.randperm(2*n)
        return samples[perm, :], y[perm]
    end
    return samples, y
end