include("../src/flows.jl")
include("../src/exponential.jl")
include("sinusoid.jl")


σ = 1.0
K, F = setup_linear_system(σ)
λ = 1e-12
K = K + λ*I
P = float.(Array(I(length(F))))

begin
    gf = GradientFlow(K, F, P)
    dt = 1.0
    scheme = FullExponentialStep(dt, P*K, P*F)
    iterator = scheme.step_function
    X_new = randn(length(F))
    X_old = randn(length(F))
    res = Inf
    iter = 0
    M = length(F)
    vf = gf.vf
    residuals = Dict()
    for i=1:10
        iterator(X_new, X_old)
        X_old = X_new
        
        iter += 1 
        res = norm(vf(X_old))
        @info iter, res
        residuals[iter] = res
    end
end 

B = P*K
U, S = svd(B)

begin
    B_inv = U*inv(diagm(S))*U'
    B_inv*P*F
    norm(vf(B_inv*P*F))
end


begin
    r = 500
    dt = 10_000
    Σr = diagm(S[1:r])
    diag(exp(-Σr*dt))
    inv(Σr)
    float.(I(length(r))) .- exp.(-Σr*dt)
    Op1 = U[:,1:r]*exp.(-Σr*dt)*U[:,1:r]'
    Op2 = U[:,1:r]*(inv(Σr)*(float.(I(length(r))) .- exp.(-Σr*dt)))*U[:,1:r]'
    x_0 = zero(F)
    x_1 = Op1*x_0 + Op2*P*F
    vf(x_1)
    norm(vf(x_1))
end