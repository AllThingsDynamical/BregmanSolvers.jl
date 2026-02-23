include("custom_plots.jl")
using LinearAlgebra
VIS = false

if VIS
    ndims = 2
    xmin = -π
    xmax = π
    ymin = -π
    ymax = π

    u = (x,y) -> sin(x)*sin(y) + sin(4*x)*sin(4*y)
    Lu = (x,y) -> -sin(x)*sin(y) - 16*sin(4*x)*sin(4*y)
    Hu = (x,y) -> -Lu(x,y) + 2*u(x,y)
    f = (x,y) -> Hu(x,y)

    Nx = Ny = 50
    x = LinRange(xmin, xmax, Nx)
    y = LinRange(xmin, xmax, Ny)

    F = [f(xi, yi) for xi in x, yi in y]
    U = [u(xi, yi) for xi in x, yi in y]


    figure1 = heatmap(F, title="f")
    figure2 = heatmap(U, title="u")
    plot(figure1, figure2)

    b = vec(F)

    N = Nx
    dx = x[2]-x[1]
    L1 = (1/dx^2)*diagm(0=> -2*ones(N), 1=> ones(N-1), -1=>ones(N-1))
    L2 = kron(L1, I(N)) + kron(I(N), L1)
    A = -L2 + 2*I 

    x = A \ b
    u_sol = reshape(x, N, N)
    figure3 = heatmap(u_sol)
    plot(figure2, figure3)

    figure4 = spy(A, colorbar=true, title="Modified Helmholtz")
    savefig("problems/figures/modified_helmholtz.png")
end

function helmholtz(N::Int)
    ndims = 2
    xmin = -π
    xmax = π
    ymin = -π
    ymax = π

    u = (x,y) -> sin(x)*sin(y) + sin(4*x)*sin(4*y)
    Lu = (x,y) -> -sin(x)*sin(y) - 16*sin(4*x)*sin(4*y)
    Hu = (x,y) -> -Lu(x,y) + 2*u(x,y)
    f = (x,y) -> Hu(x,y)

    Nx = Ny = N
    x = LinRange(xmin, xmax, Nx)
    y = LinRange(ymin, ymax, Ny)

    F = [f(xi, yi) for xi in x, yi in y]
    b = vec(F)

    dx = x[2]-x[1]
    L1 = (1/dx^2)*diagm(0=> -2*ones(N), 1=> ones(N-1), -1=>ones(N-1))
    L2 = kron(L1, I(N)) + kron(I(N), L1)
    A = -L2 + 2*I
    return A, b 
end
