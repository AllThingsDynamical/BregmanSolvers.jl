using IterativeSolvers
include("custom_plots.jl")
include("../problems/all.jl")

function cg_reference(A::Matrix, b::Vector)
    x, history = cg(A, b; maxiter=size(A,2), verbose=false, log=true, reltol=1e-8)
    res = history[:resnorm]
    flag = history.isconverged
    @show flag
    return x, res     
end

function visualize_cg_convergence(A::Matrix, b::Vector)
    x, res = cg_reference(A,b)
    figure = plot(1:length(res), res, yaxis=:log, label="CG", xlabel="# Iterations", ylabel="|Residual|", 
        color=:red, minorgrid=true,tickfontsize = 16, guidefontsize=16,
    titlefontsize = 18)
    display(figure)
    return figure
end


# Modified Helmholtz.
Ns = [900, 1600, 2500, 3600, 4900, 6400, 8100]
for N in Ns
    n = Int(sqrt(N))
    A, b = helmholtz(n)
    fig = visualize_cg_convergence(A, b)
    savefig("scripts/figures/cg/helmholtz/helmholtz_N_$(N).png")
end

Ns = [900, 1600, 2500, 3600, 4900, 6400, 8100]
for N in Ns
    n = Int(sqrt(N))
    A, b = kernel_ridge_regression(n)
    A = A + 5e-8*I
    fig = visualize_cg_convergence(A, b)
    savefig("scripts/figures/cg/krr/krr_N_$(N).png")
end

Ns = [900, 1600, 2500, 3600, 4900, 6400, 8100]
for N in Ns
    A, b = rfnn(N)
    A = A + 5e-1*I
    fig = visualize_cg_convergence(A, b)
    savefig("scripts/figures/cg/rfnn/rfnn_N_$(N).png")
end