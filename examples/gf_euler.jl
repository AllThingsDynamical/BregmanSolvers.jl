include("../src/flows.jl")
include("../src/exponential.jl")
include("sinusoid.jl")

σ = 1.0
K, F = setup_linear_system(σ)
λ = 1e-1
K = K + λ*I
P = float.(Array(I(length(F))))

begin
    gf = GradientFlow(K, F, P)
    dt = 1e-3
    scheme = EulerStep(dt)
    iterator = scheme.step_function
    X_new = randn(length(F))
    X_old = randn(length(F))
    res = Inf
    iter = 0
    M = length(F)
    vf = gf.vf
    residuals = Dict()
    for i=1:M
        iterator(X_new, X_old, vf)
        X_old = X_new
        
        iter += 1 
        res = norm(vf(X_old))
        @info iter, res
        residuals[iter] = res
    end
end 

using Plots
using Measures
theme(:wong)
default(
    framestyle = :box,
    legend = :topright,
    grid = true,
    minorgrid=true,
    linewidth = 2,
    markersize = 6,
    gridstyle = :solid,
    minorgridstyle = :dot,
    margin=5mm,
    guidefont = font(12, "Helvetica"),
    tickfont  = font(10, "Helvetica"),
    legendfont = font(10, "Helvetica"),
    dpi=300
)

residual_vec = zeros(length(F))
for i=1:length(F)
    residual_vec[i] = residuals[i]
end

figure1 = plot(1:length(F), residual_vec, yaxis=:log, label="Residual",
        xlabel="Iterations", ylabel=" |PB - PAX|", title="Gradient flow - No preconditioner")
savefig("examples/figures/gf_no_precond.png")