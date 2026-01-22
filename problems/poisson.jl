using Plots
using LaTeXStrings
using LinearAlgebra

begin
# Global publication theme
default(
    fontfamily = "Computer Modern",
    linewidth = 2.2,
    markersize = 2,
    legendfontsize = 11,
    guidefontsize = 13,
    tickfontsize = 11,
    titlefontsize = 14,
    framestyle = :box,
    grid = false,
    minorgrid = false,
    tickdirection = :out,
    foreground_color_border = :black,
    foreground_color_axis = :black,
    foreground_color_text = :black,
    background_color = :white,
    size = (720, 480),
    dpi = 300
)

# Consistent color cycle (colorblind-safe, print-friendly)
const PUB_COLORS = [
    RGB(0.0, 0.2, 0.6),   # deep blue
    RGB(0.8, 0.2, 0.2),   # red
    RGB(0.2, 0.6, 0.2),   # green
    RGB(0.6, 0.4, 0.0),   # ochre
    RGB(0.4, 0.2, 0.6)    # purple
]
palette(PUB_COLORS)

# Convenience wrapper for axis labels (LaTeX by default)
xlabel!(s) = xlabel!(L"$s$")
ylabel!(s) = ylabel!(L"$s$")
end

TEST = false
VIS = false

if TEST # Parameters
    ndims = 2
    xmin = -π
    xmax = π
    ymin = -π
    ymax = π
    f = (x,y) -> 2*(sin(x)*sin(y) + 16*sin(4x)*sin(4y))
    u_func = (x,y) -> sin(x)*sin(y) + sin(4x)*sin(4y) 
    N = 100
end

if TEST
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
end

# Direct solve
if TEST
    x_sol = A \ b
    u = reshape(x_sol, N, N)
end

# Visualization and comparison
if TEST
    figure1 = heatmap(u, title="Approximate solution")
    U = zeros(N, N)
    for (i, xi) in enumerate(x)
        for (j, yi) in enumerate(y)
            U[i,j] = u_func(xi, yi)
        end
    end
    figure2 = heatmap(U, title="Analytical solution")
    figure3 = plot(figure1, figure2)
end

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
function poisson_problem_2d()
    # Parameters
    ndims = 2
    xmin = -π
    xmax = π
    ymin = -π
    ymax = π
    f = (x,y) -> 2*(sin(x)*sin(y) + 16*sin(4x)*sin(4y))
    u_func = (x,y) -> sin(x)*sin(y) + sin(4x)*sin(4y) 
    N = 100

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


if VIS
    A, b = poisson_problem_2d()
    figure1 = heatmap(A, title="A")
    figure2 = plot(eigvals(A), label=false, xlabel="Index", ylabel="Eigenvalues of A")
    plot(figure1, figure2, size=(1000, 300))
    savefig("problems/poisson-2d.png")
end