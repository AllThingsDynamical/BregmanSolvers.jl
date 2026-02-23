using LinearAlgebra
using Plots
using QuasiMonteCarlo
include("custom_plots.jl")
VIS = true


function rosen_brock(x::Vector)
    f1 = (x,y) -> 100*(x-y^2)^2 + (1-y)^2
    N = 4
    s = 0
    for i=1:N
        s += f1(x[i+1], x[i])
    end
    return s
end

function rosen_brock(X::Matrix)
    d, M = size(X)
    F = []
    for i=1:M
        push!(F, rosen_brock(X[:,i]))
    end
    reduce(hcat, F)
end

function sample_weights_and_biases(K::Int)
    W = randn(K, 5)
    b = rand(K)
    return W, b
end

function rfnn(N::Int; K = 2500)
    lb = -1*ones(5)
    ub = ones(5)
    sampler = QuasiMonteCarlo.HaltonSample()
    X = QuasiMonteCarlo.sample(N, lb, ub, sampler)
    F = rosen_brock(X)

    W, b = sample_weights_and_biases(K)
    ϕ = x-> tanh.(W*x .+ b)
    Φ = zeros(N, K)
    for i=1:N
        Φ[i,:] .= ϕ(X[:,i])
    end

    A = Φ'*Φ
    b = Φ'*vec(F)

    return A,b
end

if VIS
    A,b = rfnn(1000)
    figure1 = spy(A, colorbar=true, title="Feature matrix")
    savefig("problems/figures/rfnn.png")
end