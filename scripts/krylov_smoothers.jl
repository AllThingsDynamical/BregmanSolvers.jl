include("custom_plots.jl")
include("../problems/all.jl")
using LinearAlgebra
using ExponentialUtilities
using IterativeSolvers

function lanczos_decomp(A, b, k; tol = 1e-14, opnorm = nothing)
    b = copy(b)
    β0 = norm(b)
    β0 == 0 && throw(ArgumentError("starting vector must be nonzero"))
    b ./= β0

    n = length(b)
    T = eltype(b)

    Ks = KrylovSubspace{T}(n, k)
    if opnorm === nothing
        lanczos!(Ks, A, b; m = k, tol = tol)
    else
        lanczos!(Ks, A, b; m = k, tol = tol, opnorm = opnorm)
    end

    V = Matrix(ExponentialUtilities.getV(Ks))          # size n × (m_used+1)
    H = Matrix(ExponentialUtilities.getH(Ks))          # size (m_used+1) × m_used
    m = Ks.m

    Vk = V[:, 1:m]
    Tk = H[1:m, 1:m]

    βkp1 = m < size(H, 1) ? H[m+1, m] : zero(eltype(H))

    return Vk, Tk, βkp1, Ks
end

_apply(M, x) = M * x
_apply(f::Function, x) = f(x)


function krylov_smoothing(A, b, P, k, x0, maxiter;
    phi,
    rtol = 1e-14,
    atol = 0.0,
    tol = 1e-14,
    opnorm = nothing,
    verbose = false,
)

    x = copy(x0)

    PA = P*A
    Pb = P*b

    r = Pb - PA*x
    r0 = norm(r)
    thresh = max(atol, rtol * r0)

    reshist = Float64[r0]

    verbose && println("iter = 0, ‖r‖ = $(r0)")

    it = 0
    while reshist[end] > thresh && it < maxiter
        it += 1

        c = r
        nc = norm(c)

        if nc == 0
            push!(reshist, 0.0)
            break
        end

        Vk, Tk, βkp1, Ks = lanczos_decomp(PA, c, k; tol=tol, opnorm=opnorm)

        vals, Uk = eigen(Symmetric(Tk))

        e1 = zeros(eltype(Tk), length(vals))
        e1[1] = one(eltype(Tk))

        Φe1 = phi.(vals) .* (Uk' * e1)
        δx = nc * Vk * (Uk * Φe1)

        x .= x .+ δx
        r = Pb - PA*x

        rn = norm(r)
        push!(reshist, rn)

        verbose && println("iter = $it, ‖r‖ = $(rn)")
    end

    converged = reshist[end] <= thresh
    return x, reshist, converged, it
end

function krylov_reference(A, b; phi, k, x0, maxiter, rtol)
    x, reshist, converged, iters = krylov_smoothing(
        A, b, Matrix{eltype(b)}(I(size(A,1))), k, x0, maxiter;
        phi = phi,
        rtol = rtol,
        verbose = false,
    )
    return x, reshist
end

function cg_reference(A, b; rtol)
    x, history = cg(A, b; maxiter=size(A,2), log=true, reltol=rtol)
    return x, history[:resnorm]
end

function visualize_comparison(A, b; phi, k, x0, maxiter=200, rtol=1e-8)
    _, res_krylov = krylov_reference(A, b;
        phi=phi, k=k, x0=x0, maxiter=maxiter, rtol=rtol)

    _, res_cg = cg_reference(A, b; rtol=rtol)

    fig = plot(1:length(res_krylov), res_krylov,
        yaxis=:log, label="Krylov smoother",
        xlabel="# Iterations", ylabel="|Residual|",
        linewidth=2, minorgrid=true)

    plot!(1:length(res_cg), res_cg,
        label="CG", linewidth=2, linestyle=:dash)

    return fig
end

Ns = [900, 1600, 2500, 3600, 4900, 6400]

for N in Ns
    n = Int(sqrt(N))
    A, b = helmholtz(n)

    phi = λ -> (1 - exp(-1e14*λ)) / λ
    x0 = zeros(N)

    fig = visualize_comparison(A, b;
        phi=phi, k=30, x0=x0, rtol=1e-8)

    savefig("BregmanSolvers.jl/scripts/figures/comparison/helmholtz/helmholtz_N_$(N).png")
end



for N in Ns
    n = Int(sqrt(N))
    A, b = poisson(n)

    phi = λ -> (1 - exp(-1e14*λ)) / λ
    x0 = zeros(N)

    fig = visualize_comparison(A, b;
        phi=phi, k=30, x0=x0, rtol=1e-8)

    savefig("BregmanSolvers.jl/scripts/figures/comparison/poisson/poisson_N_$(N).png")
end


for N in Ns
    n = Int(sqrt(N))
    A, b = kernel_ridge_regression(n)
    A = A + 1e-5*I

    phi = λ -> (1 - exp(-1e8*λ)) / λ
    x0 = zeros(N)

    fig = visualize_comparison(A, b;
        phi=phi, k=100, x0=x0, rtol=1e-8)

    savefig("BregmanSolvers.jl/scripts/figures/comparison/krr/krr_N_$(N).png")
end

for N in Ns
    n = Int(sqrt(N))
    A, b = rfnn(N, K=N)
    A = A + 1e-1*I

    phi = λ -> (1 - exp(-1e3*λ)) / λ
    x0 = zeros(N)

    fig = visualize_comparison(A, b;
        phi=phi, k=100, x0=x0, rtol=1e-8)

    savefig("BregmanSolvers.jl/scripts/figures/comparison/rfnn/rfnn_N_$(N).png")
end

PROFILE = false
if PROFILE
    using BenchmarkTools
    using IterativeSolvers

    function run_cg(A, b; rtol=1e-8)
        x, history = cg(A, b; maxiter=size(A,2), verbose=false, log=true, reltol=rtol)
        return x, history[:resnorm]
    end

    function run_krylov(A, b; phi, k, x0, maxiter=200, rtol=1e-8)
        x, reshist, converged, iters = krylov_smoothing(
            A, b, Matrix{eltype(b)}(I(size(A,1))), k, x0, maxiter;
            phi=phi,
            rtol=rtol,
            verbose=false,
        )
        return x, reshist
    end

    N = 50^2
    n = Int(sqrt(N))
    A, b = helmholtz(n)
    phi = λ -> (1 - exp(-1e14*λ)) / λ
    x0 = zeros(N)

    @btime run_cg($A, $b; rtol=1e-8);
    @btime run_krylov($A, $b; phi=$phi, k=30, x0=$x0, maxiter=200, rtol=1e-8);

    using Profile
    using StatProfilerHTML

    Profile.clear()
    @profile run_cg(A, b; rtol=1e-8)
    statprofilehtml()

    Profile.clear()
    @profile run_krylov(A, b; phi=phi, k=30, x0=x0, maxiter=200, rtol=1e-8)
    statprofilehtml()
end
