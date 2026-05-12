include("custom_plots.jl")
include("../problems/all.jl")
using LinearAlgebra
using IterativeSolvers
using ExponentialUtilities


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

        if isnan(sum(δx))
            throw("Krylov smoothing is diverging.")
        end

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


function cg_reference(A, b, maxiter; rtol)
    x, history = cg(A, b; maxiter=maxiter, log=true, reltol=rtol)
    return x, history[:resnorm]
end

function richardson_smoothing(
    A, b, P, dt, x0, maxiter;
    rtol = 1e-14,
    atol = 0.0,
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

        dx = dt * r
        x .= x .+ dx

        r = Pb - PA*x
        rn = norm(r)

        push!(reshist, rn)

        verbose && println("iter = $it, ‖r‖ = $(rn)")
    end

    converged = reshist[end] <= thresh
    return x, reshist, converged, it
end

function richardson_reference(A, b, dt;
    x0 = zeros(eltype(b), size(A, 2)),
    maxiter = size(A, 1),
    rtol = 1e-14,
    atol = 0.0,
)
    x, reshist, converged, iters = richardson_smoothing(
        A, b, Matrix{eltype(b)}(I(size(A, 1))), dt, x0, maxiter;
        rtol = rtol,
        atol = atol,
        verbose = false,
    )

    return x, reshist
end

function visualize_comparison_k(
    A, b;
    phi,
    ks = [1, 5, 20, 50],
    dt,
    x0,
    maxiter = 200,
    rtol = 1e-8,
    pos = :bottomright
)

    fig = plot(
        yaxis = :log,
        xlabel = "# Iterations",
        ylabel = "|Residual|",
        linewidth = 2,
        minorgrid = true,
        legend=pos,
        tickfontsize = 14,
        guidefontsize = 16,
        legendfontsize = 14,
        titlefontsize = 18,
    )

    # Krylov smoothers for different k
    for k in ks
        _, res_krylov = krylov_reference(
            A, b;
            phi = phi,
            k = k,
            x0 = x0,
            maxiter = maxiter,
            rtol = rtol
        )

        plot!(
            1:length(res_krylov),
            res_krylov,
            label = "KS (k = $k)"
        )
    end

    # CG reference
    _, res_cg = cg_reference(
        A, b, maxiter;
        rtol = rtol
    )

    plot!(
        1:length(res_cg),
        res_cg,
        label = "CG",
        linestyle = :dash
    )

    # Richardson reference
    _, res_richardson = richardson_reference(
        A, b, dt;
        x0 = x0,
        maxiter = maxiter,
        rtol = rtol
    )

    plot!(
        1:length(res_richardson),
        res_richardson,
        label = "Richardson",
        linestyle = :dot
    )

    return fig
end


function make_phi(t; atol=1e-14)
    return λ -> begin
        if abs(λ) < atol
            t
        else
            -expm1(-t * λ) / λ
        end
    end
end
