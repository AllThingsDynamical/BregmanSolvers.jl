using LinearAlgebra
include("custom_plots.jl")
VIS = false

if VIS
    N = 50
    ndims = 2
    xmin = -6π
    xmax = 6π
    ymin = -6π
    ymax = 6π
    u = (x,y) -> sin(x)*cos(y) + sin(3*x)*cos(2*y)
    x = LinRange(xmin, xmax, N)
    y = LinRange(ymin, ymax, N)
    U = [u(xi, yi) for xi in x, yi in y]
    X = [xi for yi in y, xi in x]
    Y = [yi for yi in y, xi in x]

    input = vcat(reshape(X, 1, :), reshape(Y, 1, :))
    output = reshape(U, 1, :)

    function gaussian(x,y)
        exp(-norm(x-y)^2/24^2)
    end

    M = size(input, 2)
    K = zeros(M, M)
    for i=1:M
        for j=1:M
            K[i,j] = gaussian(input[:,i], input[:,j])
        end
    end
    K

    figure = spy(K, colorbar=true, title="Gram matrix")
    savefig("problems/figures/krr.png")
end

function kernel_ridge_regression(N::Int)
    ndims = 2
    xmin = -6π
    xmax = 6π
    ymin = -6π
    ymax = 6π
    u = (x,y) -> sin(x)*cos(y) + sin(3*x)*cos(2*y)
    x = LinRange(xmin, xmax, N)
    y = LinRange(ymin, ymax, N)
    U = [u(xi, yi) for xi in x, yi in y]
    X = [xi for yi in y, xi in x]
    Y = [yi for yi in y, xi in x]

    input = vcat(reshape(X, 1, :), reshape(Y, 1, :))
    output = reshape(U, 1, :)

    function gaussian(x,y)
        exp(-norm(x-y)^2/24^2)
    end

    M = size(input, 2)
    K = zeros(M, M)
    for i=1:M
        for j=1:M
            K[i,j] = gaussian(input[:,i], input[:,j])
        end
    end
    return K, vec(U)
end
