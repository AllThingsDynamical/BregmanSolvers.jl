include("../core.jl")

function make_Q_b(n::Integer)
    A = fill(-1.0, n, n)
    @inbounds for i in 1:n
        A[i, i] = n
    end
    b = 4*randn(n)
    return A, b
end

begin
    N = 5_000
    A, b = make_Q_b(N)
    phi = make_phi(1e50)
    x0 = zeros(N)
    dt = 1.0
    isposdef(A)
    P = inv(diagm(diag(A)))
    visualize_comparison_k(P*A, P*b; ks=[1,2,3,4,5], phi, dt, x0,maxiter=100, rtol=1e-10)
end


function run_size_experiment(; Ns=1_000:1000:6_000,
    ks=[1,2,3,4,5],
    t=1e50,
    dt=1.0,
    maxiter=100,
    rtol=1e-10,
    outdir="scripts/figures/comparison/sdd/")

    mkpath(outdir)

    for N in Ns
        @info "Running experiment" N

        A, b = make_Q_b(N)
        phi = make_phi(t)
        x0 = zeros(N)

        P = Diagonal(1.0 ./ diag(A))

        fig = visualize_comparison_k(P * A, P * b;
            ks=ks,
            phi=phi,
            dt=dt,
            x0=x0,
            maxiter=maxiter,
            rtol=rtol
        )
        plot!(fig; title="Strictly diagonally dominant matrix N = $N",
            tickfontsize = 12,
            guidefontsize = 14,
            legendfontsize = 11,
            titlefontsize = 14,
            linewidth = 3,
            thickness_scaling = 1.3
        )


        filename = joinpath(outdir, "Q_experiment_N_$(N).png")
        savefig(fig, filename)

        @info "Saved figure" filename
    end

    return nothing
end

run_size_experiment()