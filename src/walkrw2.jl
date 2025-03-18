



# Rimless wheel state-derivative
"""
    fwalk!(xdot, x, model::DynoLocoModel [, t = 0.])

State-derivative for dynamic locomotion model. Return a state-derivative
vector in-place, using pre-allocated xdot.  See also xdot = fwalk(x, ...).
"""
function fwalk!(xdot, x, w::WalkRW2, t = 0.) # in-place version
    # state-derivative for rimless wheel
    γ = model.γ
    q1, u1 = x
    q1dot = u1
    u1dot = sin(q1 - γ)
    if w.safety && u1 > w.safetyminu # safety keeps from falling backwards
        u1dot = 0
        u1 = w.safetyminu
    end
    xdot[1] = u1
    xdot[2] = u1dot
end


"""
    xdot = fwalk(x, model::Walk [, t = 0.])

State-derivative for dynamic locomotion model. Return a state-derivative
vector. See also pre-allocated version fwalk!(xdot, x, ...).
"""
function fwalk(x, model::Walk, t = 0.)
    xdot = similar(x)
    fwalk!(xdot, x, model, t)
    return xdot
end


function s2stransition(xminus, P, w::WalkRW2)
    (q1, u1) = xminus
    α = w.α
    c2t = cos(2α)
    s2t = sin(2α)

    KEminus = 0.5*u1^2
    pushoffwork = 0.5*P^2
    KEzero = KEminus + pushoffwork # kinetic energy after push-off

    uleadinglegplus = c2t*u1 - P*s2t # collision and push-off

    xplus = [xminus[1], uleadinglegplus] # state after push-off, collision

    # switch legs, assuming model landed at perfect leg angle α
    xnew = [α, uleadinglegplus]

    KEplus = 0.5*uleadinglegplus^2 # kinetic energy after collision
    collisionwork = KEplus - KEzero

    return xnew, S2S(pushoffwork, collisionwork, KEminus, KEzero, KEplus)
end

"""
    heelstrikecondition(x, t [, integrator], model<:DynoLocoModel)

Compute the zero-crossing condition for heelstrike, based on state `x`.
The integrator::DEIntegrator can optionally be supplied as an input
argument (for compatibility with DifferentialEquations.jl).
"""
heelstrikecondition(x, t, integrator::DiffEqBase.DEIntegrator, model::DynoLocoModel) =
    heelstrikecondition(x, t, model)

# As a default behavior, heelstrike is detected when the stance leg
# goes past half-leg angle α. This is appropriate for rimless wheel,
# but should be overriden for other models.
function heelstrikecondition(x, t, model::DynoLocoModel)
    α = model.α
    (q1,q2) = x[1:2]
    q1+α   # heelstrike when q1 = -alpha (negative crossing recommended)
end
