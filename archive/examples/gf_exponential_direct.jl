include("../src/flows.jl")
include("../src/exponential.jl")
include("sinusoid.jl")
using RandomizedLinAlg
using LinearAlgebra
using TSVD


σ = 1.0
K, F = setup_linear_system(σ)
λ = 1e-1
K = K + λ*I
P = float.(Array(I(length(F))))
r = 500
B = P*K
Ur, Sr = rsvd(B,r)
U, S = svd(B)
Ut, St = tsvd(B,r)

gf = GradientFlow(K, F, P)
vf = gf.vf


# Direct inverse.
begin
    B_inv = U*inv(diagm(S))*U'
    B_inv*P*F
    norm(vf(B_inv*P*F))
end

# Inverse from truncated singular values
begin
    dt = 100_000
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

# Inverse from randomized singular values
begin
    dt = 100_000
    Σr = diagm(Sr)
    diag(exp(-Σr*dt))
    inv(Σr)
    float.(I(length(r))) .- exp.(-Σr*dt)
    Op1 = Ur*exp.(-Σr*dt)*Ur'
    Op2 = Ur*(inv(Σr)*(float.(I(length(r))) .- exp.(-Σr*dt)))*Ur'
    x_0 = zero(F)
    x_1 = Op1*x_0 + Op2*P*F
    vf(x_1)
    norm(vf(x_1))
end

# Inverse from the upto r basis
begin
    dt = 100_000
    Σt = diagm(St)
    diag(exp(-Σt*dt))
    inv(Σt)
    float.(I(length(r))) .- exp.(-Σt*dt)
    Op1 = Ut*exp.(-Σt*dt)*Ut'
    Op2 = Ut*(inv(Σt)*(float.(I(length(r))) .- exp.(-Σt*dt)))*Ut'
    x_0 = zero(F)
    x_1 = Op1*x_0 + Op2*P*F
    vf(x_1)
    norm(vf(x_1))
end

## Notice that the truncated singular values are competitive to some 
# extent to the direct inverse
# On the other hand, the set of default randomized directions are not competitive.
