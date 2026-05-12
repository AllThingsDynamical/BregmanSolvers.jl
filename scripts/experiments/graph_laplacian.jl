using LinearAlgebra
using SparseArrays
using Random
include("../core.jl")


function erdos_renyi_laplacian(n::Integer, p::Real; rng=Random.default_rng())
    @assert n > 0 "n must be positive"
    @assert 0 ≤ p ≤ 1 "p must be in [0,1]"

    rows = Int[]
    cols = Int[]

    for i in 1:n-1
        for j in i+1:n
            if rand(rng) < p
                push!(rows, i); push!(cols, j)
                push!(rows, j); push!(cols, i)
            end
        end
    end

    vals = ones(Float64, length(rows))
    A = sparse(rows, cols, vals, n, n)

    d = vec(sum(A, dims=2))
    L = Diagonal(d) - A

    return Matrix(L)
end

function graph_laplacian_system(n::Int)
    L = erdos_renyi_laplacian(n, 0.05; rng=Xoshiro(1))
    b = randn(n)
    return L, b
end

begin
    N = 4_000
    A, b = graph_laplacian_system(N)
    phi = make_phi(1e50)
    x0 = zeros(N)
    dt = 1e-8
    isposdef(A)
    P = A'
    visualize_comparison_k(P*A, P*b; ks=[1,2,3,4,5], phi, dt, x0,maxiter=100, rtol=1e-10)
end


function run_graph_laplacian_experiment(;
    Ns = 9000:1000:14000,
    p = 0.05,
    ks = [1,2,3,4,5],
    t = 1e50,
    dt = 1e-8,
    maxiter = 100,
    rtol = 1e-10,
    outdir = "scripts/figures/comparison/graph_laplacian/"
)
    mkpath(outdir)

    for N in Ns
        @info "Running graph Laplacian experiment" N

        A, b = graph_laplacian_system(N)

        phi = make_phi(t)
        x0 = zeros(N)

        P = A'

        fig = visualize_comparison_k(P * A, P * b;
            ks = ks,
            phi = phi,
            dt = dt,
            x0 = x0,
            maxiter = maxiter,
            rtol = rtol
        )

        plot!(fig;
            title = "Erdős--Rényi graph Laplacian, N = $N",
            tickfontsize = 12,
            guidefontsize = 14,
            legendfontsize = 11,
            titlefontsize = 14,
            linewidth = 3,
            thickness_scaling = 1.3
        )

        filename = joinpath(outdir, "ER_laplacian_N_$(N).png")
        savefig(fig, filename)

        @info "Saved figure" filename
    end

    return nothing
end

run_graph_laplacian_experiment()