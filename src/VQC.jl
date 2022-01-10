import Yao

mutable struct VQC
    circuit
    theta::Vector{Float64}
    step:: Float64
end


function BCE_loss(y_true, y_pred)
    eps = 0.01
    res = sum(.*(y_true, log.(y_pred .+ eps)) + .*(1 .- y_true, log.(1 .- y_pred .+ eps)))
    return (-1)*res/length(y_true)
end


function parity(x)
    return Int(sum(x) % 2)
end


function parity_proba(x)
    return sum(parity.(x))/length(x)
end


function vqcevaluate_(vqc::VQC, x::AbstractVector, nshots::Int, theta::Vector{Float64})
    if theta==nothing
        theta = vqc.theta
    end
    qc = vqc.circuit(x, theta) 
    n_qubits = Yao.nqubits(qc)
    r = Yao.zero_state(n_qubits)
    result =  r |> qc |> r-> Yao.measure(r, nshots=nshots)
    return parity_proba(result)
end


function vqcevaluate(vqc::VQC, X::AbstractMatrix; nshots=1024::Int, theta=vqc.theta::Vector{Float64})
    n = axes(X)[1].stop
    y = Vector{Float64}(undef, n)
    for (i, x) in enumerate(eachrow(X))
        y[i] = vqcevaluate_(vqc, x, nshots, theta)
    end
    return y
end


function vqctrain!(vqc::VQC, X::AbstractMatrix, y::AbstractVector; verbose=false, loss=BCE_loss)
    maxit = 100
    multiplier = 25
    loss_cur = loss(y, vqcevaluate(vqc, X))
    all_losses = [loss_cur]
    i=0
    print("Begin VQC training...\n")
    while i < maxit
        theta_pr = vqc.theta
        for i=1:length(theta_pr)
            theta_hat = vqc.theta
            
            # increase parameter
            theta_inc = theta_hat[i] + 2*π* vqc.step
            theta_hat[i] = theta_inc
            loss_inc = loss(y, vqcevaluate(vqc, X, theta=theta_hat, nshots=128))
            
            # decrease parameter
            theta_dec = theta_hat[i] - 2*π* vqc.step
            theta_hat[i] = theta_dec
            loss_dec = loss(y, vqcevaluate(vqc, X, theta=theta_hat, nshots=128))
            
            # update
            if (loss_inc < loss_cur) & (loss_inc < loss_dec)
                theta_pr[i] = theta_pr[i] + 2*π* vqc.step * (loss_cur - loss_inc) * multiplier
            elseif (loss_dec < loss_cur)
                theta_pr[i] = theta_pr[i] - 2*π* vqc.step * (loss_cur - loss_dec) * multiplier
            end
        end

        loss_pr = loss(y, vqcevaluate(vqc, X, theta=theta_pr))
        loss_diff = loss_cur - loss_pr
        
        #if loss_diff > 0.0
            vqc.theta = theta_pr
        #end
        
        if verbose
            print("Iteration $i \n")
            print("Current loss: $(loss_pr) \n")
        end
        loss_cur = loss_pr
        push!(all_losses, loss_cur)
        i+=1
    end
    print("Stopping training.\n")
    return vqc, all_losses
end

function vqcpredict_proba(vqc::VQC, X::AbstractMatrix)
    y_hat = vqcevaluate(vqc, X)
    return y_hat
end


function vqcpredict(vqc::VQC, X::AbstractMatrix)
    y_hat = vqcevaluate(vqc, X)
    y_hat[y_hat .>=0.5] .= 1
    y_hat[y_hat .<0.5] .= 0
    return y_hat
end