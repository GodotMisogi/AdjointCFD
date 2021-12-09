module NewtonMethod

function newton_solver!(R, f!, ∂f∂p!, p, num_iters = 3, α = 1.0)
    # Array to store errors
    ε = zeros(num_iters);

    # Newton iteration loop
    for i in 1:num_iters
        # (Generalize to Jac-vec matrix-free method such as Krylov subspace for large problems.)

        # Compute residuals
        R    = f!(R, p)

        # Compute Jacobian
        ∂R∂p = ∂f∂p!(R, p)

        # Compute Newton step
        Δp   = ∂R∂p \ -R

        # Update state with relaxation factor
        p    = newton_update!(p, reshape(Δp, size(p)), α)

        # Error processing
        ε[i] = maximum(abs.(Δp))
        println("Newton step error L² norm: $(ε[i])")
    end
    R, p, ε
end
