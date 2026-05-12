include("../core.jl")
using LinearAlgebra
using SparseArrays
using Random

function variable_diffusion_2d(N::Int; α=1e-2, contrast=10.0, seed=1)
    rng = Xoshiro(seed)

    h = 1.0 / (N + 1)
    n = N^2

    idx(i,j) = i + (j-1)*N

    # coefficient field on cell/edge locations
    a = [1.0 + contrast*(0.5 + 0.5*sin(6π*i*h)*sin(4π*j*h)) for i in 1:N, j in 1:N]

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for j in 1:N
        for i in 1:N
            row = idx(i,j)

            diagval = α

            # left
            if i > 1
                ae = 0.5*(a[i,j] + a[i-1,j])
                push!(rows,row); push!(cols,idx(i-1,j)); push!(vals,-ae/h^2)
                diagval += ae/h^2
            end

            # right
            if i < N
                ae = 0.5*(a[i,j] + a[i+1,j])
                push!(rows,row); push!(cols,idx(i+1,j)); push!(vals,-ae/h^2)
                diagval += ae/h^2
            end

            # down
            if j > 1
                ae = 0.5*(a[i,j] + a[i,j-1])
                push!(rows,row); push!(cols,idx(i,j-1)); push!(vals,-ae/h^2)
                diagval += ae/h^2
            end

            # up
            if j < N
                ae = 0.5*(a[i,j] + a[i,j+1])
                push!(rows,row); push!(cols,idx(i,j+1)); push!(vals,-ae/h^2)
                diagval += ae/h^2
            end

            push!(rows,row); push!(cols,row); push!(vals,diagval)
        end
    end

    A = sparse(rows, cols, vals, n, n)

    x_exact = randn(rng, n)
    b = A * x_exact

    return A, b
end


begin
    N = 50
    A, b = variable_diffusion_2d(N; α=1e-1, contrast=1.0)

    phi = make_phi(1e80)
    x0 = zeros(N^2)

    dt = 1e-4
    @show isposdef(A)
    P = Diagonal(1.0 ./ diag(A))

    B = P*A*P
    c = P*b

    visualize_comparison_k(
        B,
        c;
        ks=[2, 5, 20],
        phi=phi,
        dt=dt,
        x0=x0,
        maxiter=800,
        rtol=1e-12,
        pos=:topright
    )
end

rtol = 1e-12
maxiter = 1000
k =  10
 _, res_krylov = krylov_reference(
            B, c;
            phi = phi,
            k = k,
            x0 = x0,
            maxiter = maxiter,
            rtol = rtol
        )