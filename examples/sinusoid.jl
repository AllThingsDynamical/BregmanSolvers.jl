using QuasiMonteCarlo
using Plots
using LinearAlgebra
using IterativeSolvers
using Statistics

function eval_kernel_matrix(kernel::Function, X::Matrix, σ::Float64)
    d, M = size(X)
    K = zeros(M, M)
    for i=1:M
        for j=1:M
            X1 = X[:,i]
            X2 = X[:,j]
            K[i,j] = kernel(X1, X2, σ)
        end
    end
    return K
end

function Gaussian(x, y, σ)
    d = norm(x-y)
    exp(-1/2*d^2/σ^2)        
end

function setup_linear_system(σ::Float64)
    N = 1000
    x = LinRange(-π, π, N)
    y = LinRange(-π, π, N)
    ll = (-π, -π)
    ul = (float(π), float(π))
    f(x, y) = sin(x)*sin(y) + sin(4*x)*sin(4*y)

    N_input_points = 5_000
    input_points = QuasiMonteCarlo.sample(N_input_points, ll, ul, QuasiMonteCarlo.HaltonSample())
    output_points = zeros(1, N_input_points)
    for i=1:N_input_points
        xi, yi = input_points[:,i]
        fi = f(xi, yi)
        output_points[1,i] = fi
    end

    K = Gram_matrix = eval_kernel_matrix(Gaussian, input_points, σ)
    F = output_points[:]
    return K,F
end

TEST = false
if TEST
    K, F = setup_linear_system(0.1)
    x, history = gmres(K + 1e-8*I, F; reltol=1e-8, verbose=true, maxiter=1_000, log=true)
    figure1 = plot(history, yaxis=:log10)
    quantile(history.data[:resnorm], [0.25, 0.50,  0.75, 1.0])
end