using LinearAlgebra
using Zygote
using ForwardDiff
using ReverseDiff
using StaticArrays
using Base.Iterators

## Finite differencing
central_2nd(φp, φ, φm)     = φm - 2φ + φp
Δ²_central(φp, φ, φm, Δx)  = central_2nd(φp, φ, φm) / Δx^2
Δ_central(φp, φm, Δx)      = (φp - φm) / 2Δx
Δ_forward(φ1, φ2, Δt)      = (φ2 - φ1) / Δt

## Stencil generation
neighbours(CI, deg) = CI .+ CartesianIndices(ntuple(x -> -deg:deg, length(CI)))

stencil_depth(CI, depth) = [ CI + CartesianIndex(ntuple(x -> ifelse(x == i, depth, 0), length(CI))) for i in 1:length(CI) ]

stencils(CI, deg, poke = false) = ifelse(poke, [ stencil_depth(CI, -(deg + 1)); neighbours(CI, deg)[:]; stencil_depth(CI, deg + 1) ], neighbours(CI, deg)[:])

stencil_1(CI) = stencils(CI, 0, true)

## FTCS scheme
ftcs_stencil(φ, index) = @view φ[stencil_1(index)[2:end]]

function substitute(φ, index, val)
    if index ∈ CartesianIndices(φ)
        φ[index]
    else
        val
    end
end

function ftcs_stencil(φ, index, φ_boundaries)
    if any(index.I .<= 1) || any(index.I .>= size(φ)) # There have to be more conditions...
        φ_vec = substitute.(Ref(φ), stencil_1(index), φ_boundaries)
    else
        φ_vec = @view φ[stencil_1(index)[2:end]]
    end
end

## Difference operators specific to FTCS stencil

# First derivatives
Δt(φ, dt) = Δ_forward(φ[4], φ[3], dt)
Δx(φ, dx) = Δ_central(φ[5], φ[1], dx)
Δy(φ, dy) = Δ_central(φ[6], φ[2], dy)

# Second derivatives
Δ²x(φ, dx) = Δ²_central(φ[5], φ[3], φ[1], dx)
Δ²y(φ, dy) = Δ²_central(φ[6], φ[3], φ[2], dy)

# Governing equations in residual form
vorticity_residual(ω, ψ, ν, dt, dx, dy) = Δt(ω, dt) + Δy(ψ, dy) * Δx(ω, dx) - Δx(ψ, dx) * Δy(ω, dy) - ν * (Δ²x(ω, dx) + Δ²y(ω, dy))
streamfunction_residual(ω, ψ, dx, dy)   = Δ²x(ψ, dx) + Δ²y(ψ, dy) + ω[3]

# Compute residual equations
function compute_residuals!(R, p, sizer, ω_bound, ψ_bound, ν, dt, dx, dy) 
    ωs = @view p[1:prod(sizer)]
    ψs = @view p[prod(sizer)+1:end]

    ωs = reshape(ωs, sizer)
    ψs = reshape(ψs, sizer)

    # Get indices and bounds
    inds = CartesianIndices(ωs)
    R    = reshape(R, (sizer..., 2))
    # ∂R∂p = reshape(∂R∂p, ((sizer..., sizer...)..., 4)) 

    # Iterate over grid
    for ind in inds
        # Get neighbours of a cell
        ω = ftcs_stencil(ωs, ind, ω_bound)
        ψ = ftcs_stencil(ψs, ind, ψ_bound)

        # Evaluate residuals
        R[ind,1] = vorticity_residual(ω, ψ, ν, dt, dx, dy) # Vorticity equation
        R[ind,2] = streamfunction_residual(ω, ψ, dx, dy)   # Streamfunction equation
    end

    R[:]
    # , reshape(∂R∂p, (2 .* (prod(size(ωs)), prod((reverse ∘ size)(ωs)))...))
end

newton_update(p, δp, α = 1.0) = p - α * δp

# Pre-allocated version
newton_update!(p, δp, α = 1.0) = @. p = p - α * δp

## Initial setup
dt, dx, dy  = 0.1, 0.2, 0.1
ts, xs, ys  = 0:dt:1, -1:dx:1, -1:dy:1
grid        = product(xs, ys, ts)
α           = 1.0
ν           = 0.05
U           = 2.0
ω_bound     = 1.0
ψ_bound     = 0.1

ωs = ω_bound * ones(size(grid))
ψs = ψ_bound * ones(size(grid))

p = [ ωs[:]; ψs[:] ]

R    = similar(p)
∂R∂p = zeros(2 .* (prod(size(ωs)), prod((reverse ∘ size)(ωs)))...)

num_iters = 3
compute_residuals!(R, p) = compute_residuals!(R, p, size(ωs), ω_bound, ψ_bound, ν, dt, dx, dy) 
compute_grad_fwd(R, p) = ForwardDiff.jacobian(compute_residuals!, R, p)

##
# using NLsolve
# prob = nlsolve(compute_residuals!, p)

## NEWTON ITERATIONS
#=================================================================#

function newton_solver!(R, p, num_iters = 3, α = 1.0)
    # Array to store errors
    ε = zeros(num_iters);

    # Newton iteration loop
    for i in 1:num_iters
        # Compute residuals
        R    = compute_residuals!(R, p)

        # Compute Jacobian
        ∂R∂p = compute_grad_fwd(R, p)

        # Compute Newton step
        Δp   = ∂R∂p \ -R

        # Update state with relaxation factor
        p   .= newton_update!(p, Δp, α)

        # Error processing
        ε[i] = maximum(abs.(Δp))
        println("Newton step error L² norm: $(ε[i])")
    end
    R, p, ε
end

##
R, p, ε = newton_solver!(R, p, num_iters, α)

## Plotting
using GLMakie

# function grad_vorticity_residual(ω, ψ, ν, dt, dx, dy)
    
# end

# grad_streamfunc_residual(ω, ψ, dx, dy) = gradient(streamfunction_residual, ω, ψ, dx, dy)

# function grad_residual_update!(∂R∂p, grad, inds)
#     j, k, l = inds.I
#     a, b, c, d, e, f = size(∂R∂p)  
#              ∂R∂p[inds, j  , k  , l  ] = grad[3]
#     if k > 1 ∂R∂p[inds, j  , k-1, l  ] = grad[1] end
#     if l > 1 ∂R∂p[inds, j  , k  , l-1] = grad[2] end
#     if j < a ∂R∂p[inds, j+1, k  , l  ] = grad[4] end
#     if k < b ∂R∂p[inds, j  , k+1, l  ] = grad[5] end
#     if l < c ∂R∂p[inds, j  , k  , l+1] = grad[6] end
# end


# function update_jacobian!(∂R∂p, ind, ω, ψ, ν, dt, dx, dy)
#     # Evaluate Jacobian
#     grad_ω = grad_vorticity_residual(ω, ψ, ν, dt, dx, dy)
#     grad_ψ = grad_streamfunc_residual(ω, ψ, dx, dy)

#     grad_residual_update!(∂R∂p[:,:,:,:,:,:,1], grad_ω[1], ind)
#     grad_residual_update!(∂R∂p[:,:,:,:,:,:,2], grad_ω[2], ind)
#     grad_residual_update!(∂R∂p[:,:,:,:,:,:,3], grad_ψ[1], ind)
#     grad_residual_update!(∂R∂p[:,:,:,:,:,:,4], grad_ψ[2], ind)

#     nothing
# end