include("custom_plots.jl")
using LinearAlgebra


"""
    poisson_problem_2d() -> (A, b)

Construct the linear system for a 2D Poisson problem on a uniform Cartesian grid
using a standard 5-point finite-difference Laplacian (Dirichlet-style interior
stencil, no explicit boundary handling).

The PDE being discretized is

    -Δu(x, y) = f(x, y),    (x, y) ∈ [-π, π] × [-π, π],

with a manufactured solution

    u(x, y) = sin(x)sin(y) + sin(4x)sin(4y),

and corresponding right-hand side

    f(x, y) = 2( sin(x)sin(y) + 16 sin(4x)sin(4y) ).

A uniform grid of `N = 100` points is used in each coordinate direction, and the
2D operator is assembled via Kronecker sums:

    L₂ = kron(L₁, I) + kron(I, L₁),

where `L₁` is the 1D second-difference matrix scaled by `1/dx^2`. The returned
system matrix is

    A = -L₂,

so that solving `A * u_vec = b` approximates the Poisson equation.

# Returns
- `A::Matrix{Float64}`: Dense system matrix of size `(N^2, N^2)` representing `-Δ`
  under the 5-point stencil on the tensor-product grid.
- `b::Vector{Float64}`: Right-hand side vector of length `N^2`, formed by sampling
  `f` on the grid and vectorizing in Julia's column-major order (`vec(F)`).
"""
function poisson(N::Int)
    # Parameters
    ndims = 2
    xmin = -π
    xmax = π
    ymin = -π
    ymax = π
    f = (x,y) -> 2*(sin(x)*sin(y) + 16*sin(4x)*sin(4y))
    u_func = (x,y) -> sin(x)*sin(y) + sin(4x)*sin(4y) 

    x = LinRange(xmin, xmax, N)
    y = LinRange(ymin, ymax, N)
    F = zeros(N, N)
    for (i,xi) in enumerate(x)
        for (j,yi) in enumerate(y)
            F[i,j] = f(xi, yi)
        end
    end

    b = vec(F)

    dx = x[2]-x[1]
    L1 = (1/dx^2)*diagm(0=> -2*ones(N), 1=> ones(N-1), -1=>ones(N-1))
    L2 = kron(L1, I(N)) + kron(I(N), L1)

    A = -L2
    return A,b
end

VIS = false
if VIS
    A, b = poisson(50)
    figure3 = spy(A, colorbar=false, title="Laplace")
    display(figure3)
    savefig("problems/figures/poisson.png")
end