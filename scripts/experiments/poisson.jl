include("../core.jl")

function make_laplacian(n::Int)
    A, b = poisson(n)
    return A, b
end

begin
    n = 4900
    N = Int(sqrt(n))
    A, b = make_laplacian(N)
    phi = make_phi(1e50)
    x0 = randn(n)
    dt = 1e-1
    isposdef(A)
    P = inv(diagm(diag(A)))
    visualize_comparison_k(P*A, P*b; ks=[1,5,10,15], phi, dt, x0,maxiter=100, rtol=1e-10, pos=:topright)
end


function run_poisson_experiment(;
    grid_sizes = [70, 80, 90, 100, 110, 120],
    ks = [1,5,10,15],
    t = 1e50,
    dt = 1e-1,
    maxiter = 100,
    rtol = 1e-10,
    outdir = "scripts/figures/comparison/pdes/"
)

    mkpath(outdir)

    for N in grid_sizes
        n = N^2

        @info "Running Poisson experiment" N n

        A, b = make_laplacian(N)

        phi = make_phi(t)
        x0 = randn(n)

        P = Diagonal(1.0 ./ diag(A))

        fig = visualize_comparison_k(
            P * A,
            P * b;
            ks = ks,
            phi = phi,
            dt = dt,
            x0 = x0,
            maxiter = maxiter,
            rtol = rtol,
            pos = :topright
        )

        plot!(fig;
            title = "2D Poisson problem, n = $n",
            tickfontsize = 12,
            guidefontsize = 14,
            legendfontsize = 11,
            titlefontsize = 14,
            linewidth = 3,
            thickness_scaling = 1.3
        )

        filename = joinpath(outdir, "Poisson_n_$(n).png")

        savefig(fig, filename)

        @info "Saved figure" filename
    end

    return nothing
end

run_poisson_experiment()