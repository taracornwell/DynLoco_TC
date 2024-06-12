# WalkRW2ls, linearized rimless wheel with step length changing linearly with speed
# WalkRW2lvs, linearzied rimless wheel with step length changing nonlinearly with speed 
#   according to Grieve eqn

export WalkRW2ls, WalkRW2lvs

# variable step lengths abstract type
abstract type AbstractWalkRW2vs <: AbstractWalkRW2 end

@with_kw struct WalkRW2ls <: AbstractWalkRW2vs
    vm = 0.24972422206418757 # initial mid-stance velocity (stance leg upright)
    P = 0.15641662809369772   # push-off impulse (mass-normalized, units of Δv)
    α = 0.3 # angle between legs
    γ = 0.  # downward slope
    L = 1.  # leg length
    M = 1.  # body mass
    g = 1.  # gravitational acceleration
    cstep = 0. # how step length increases with speed
    vmstar = 0.
    parms = (:α, :γ, :L, :M, :g) # a list of the model parameters
    limitcycle = (parms = (:vm,), f = w -> onestep(w).vm - w.vm)
    safety = false # safety keeps model from falling backward if not enough momentum
end

# example of cstep, s = s0 + cstep(v-v0)
# where s = 1*v^β, ds = β*v^(β-1)*dv
# ds = 0.42*(0.4)^(0.42-1) = 0.71
# divide by two to get 0.35

function onestep(w::WalkRW2ls; vm=w.vm, P=w.P, δangle = 0.,
    α=w.α, γ=w.γ,g=w.g,L=w.L,safety=w.safety, cstep=w.cstep, vmstar=w.vmstar)
    mylog = safety ? logshave : log # logshave doesn't blow up on negative numbers
    # Start at mid-stance leg vertical, and find the time tf1
    # to heelstrike, and angular velocity Ωminus
    # δangle is the upward change in slope from nominal γ
    # Phase 1: From mid-stance to just before impact
    # θ is angle ccw from surface normal (which may be at slope)
    newα = α + cstep*(vm-vmstar) # where s = v^0.42, ds = 0.42*v^(0.42-1)*dv, ds/dv = 0.71 for typical
    # and s = 2*L*sin(α), so dα ≈ 0.35*dv
    θf = -newα + δangle # angle of stance leg just before impact, negative means cw
    # angle wrt vertical is -α-γ+δangle
    #Ωminus = -√(2*g*(cos(0)-cos(θf))+L*vm^2)/√L  # pre-impact speed
    Ωminus = -√(2*g/L*(cos(-γ)-cos(θf-γ))+(vm/L)^2)  # pre-impact speed
    tf1 = mylog((L*(γ-θf)+√(vm^2-2γ*θf*L^2+θf^2*L^2))/(vm+L*γ)) # time until impact, phase 1
    # Step-to-step transition: Push-off and collision
    Pwork = 1/2 * P^2
    v0 = √(Ωminus^2 + P^2) # intermediate velocity after push-off
    Ωplus = cos(2newα)*Ωminus - P*sin(2newα) # initial ang vel of next leg
    Cwork = 1/2*(Ωplus^2 - v0^2) # negative collision work
    C = √(-2Cwork)
    # Phase 2: From just after impact to next mid-stance
    θnew = newα + δangle # new stance leg's initial angle
    #    tf2 = mylog((γ + √(2γ*θnew-θnew^2+Ωplus^2))/(γ-θnew-Ωplus))
    gto = γ - θnew - Ωplus # needs to be positive for pendulum to reach next mid-stance
    if safety && gto <= 0
        tf2 = 1e3 - gto # more negative gto is worse, increasing a tf2 time
    else # safety off OR gto positive
        # time to mid-stance, phase 2
        inroot = (2γ*θnew-θnew^2+Ωplus^2)
        if inroot >= 0
#            tf2 = mylog((γ + √(2γ*θnew-θnew^2+Ωplus^2))/(gto)) # time to mid-stance, phase 2
            tf2 = mylog((γ + √inroot)/(gto)) # time to mid-stance, phase 2
        else # inroot negative,
            tf2 = 1e4 - inroot # more negative inroot extends time
        end

    end
    twicenextenergy = (2g*L*cos(θnew-γ)+L^2*Ωplus^2-2g*L*cos(-γ)) # to find next vm
    if twicenextenergy >= 0.
        vmnew = √twicenextenergy
    elseif safety # not enough energy
        vmnew = (1e-3)*exp(twicenextenergy)
    else # no safety, not enough energy
        vmnew = √twicenextenergy # this should fail
    end

    # Step metrics
    steplength = 2L*sin(newα) # rimless wheel step length
    tf = tf1 + tf2 # total time mid-stance to mid-stance
    speed = steplength / tf # average speed mid-stance to mid-stance
    return (vm=vmnew, θnew=θnew, tf=tf, P=P, C=C, Pwork=Pwork,Cwork=Cwork,
        speed=speed, steplength=steplength, stepfrequency=speed/steplength,tf1=tf1, tf2=tf2,
        Ωminus=Ωminus,Ωplus=Ωplus,
        vm0=vm,δ=δangle)
end

# step length varying s = c*v^β
@with_kw struct WalkRW2lvs <: AbstractWalkRW2vs 
    vm = 0.35 # initial mid-stance velocity (stance leg upright)
    P = 0.1   # push-off impulse (mass-normalized, units of Δv)
    α = 0.3 # angle between legs
    γ = 0.  # downward slope
    L = 1.  # leg length
    M = 1.  # body mass
    g = 1.  # gravitational acceleration
    β = 0.42 # s = c*v^β
    c = 1. # how step length increases with speed
    vmstar = 0.
    parms = (:α, :γ, :L, :M, :g) # a list of the model parameters
    limitcycle = (parms = (:vm,), f = w -> onestep(w).vm - w.vm)
    safety = false # safety keeps model from falling backward if not enough momentum
end


function onestep(w::WalkRW2lvs; vm=w.vm, P=w.P, δangle = 0.,
    α=w.α, γ=w.γ,g=w.g,L=w.L,safety=w.safety, c=w.c, vmstar=w.vmstar, β=w.β)
    mylog = safety ? logshave : log # logshave doesn't blow up on negative numbers
    # Start at mid-stance leg vertical, and find the time tf1
    # to heelstrike, and angular velocity Ωminus
    # δangle is the upward change in slope from nominal γ
    # Phase 1: From mid-stance to just before impact
    # θ is angle ccw from surface normal (which may be at slope)
    # newsteplength = c*vm^β
    # newα = newsteplength / (2L)
    newα = (c*vm^β)/(2L)
    θf = -newα + δangle # angle of stance leg just before impact, negative means cw
    # angle wrt vertical is -α-γ+δangle
    #Ωminus = -√(2*g*(cos(0)-cos(θf))+L*vm^2)/√L  # pre-impact speed
    Ωminus = -√(2*g/L*(cos(-γ)-cos(θf-γ))+(vm/L)^2)  # pre-impact speed
    tf1 = mylog((L*(γ-θf)+√(vm^2-2γ*θf*L^2+θf^2*L^2))/(vm+L*γ)) # time until impact, phase 1
    # Step-to-step transition: Push-off and collision
    Pwork = 1/2 * P^2
    v0 = √(Ωminus^2 + P^2) # intermediate velocity after push-off
    Ωplus = cos(2newα)*Ωminus - P*sin(2newα) # initial ang vel of next leg
    Cwork = 1/2*(Ωplus^2 - v0^2) # negative collision work
    C = √(-2Cwork)
    # Phase 2: From just after impact to next mid-stance
    θnew = newα + δangle # new stance leg's initial angle
    #    tf2 = mylog((γ + √(2γ*θnew-θnew^2+Ωplus^2))/(γ-θnew-Ωplus))
    gto = γ - θnew - Ωplus # needs to be positive for pendulum to reach next mid-stance
    if safety && gto <= 0
        tf2 = 1e3 - gto # more negative gto is worse, increasing a tf2 time
    else # safety off OR gto positive
        # time to mid-stance, phase 2
        inroot = (2γ*θnew-θnew^2+Ωplus^2)
        if inroot >= 0
#            tf2 = mylog((γ + √(2γ*θnew-θnew^2+Ωplus^2))/(gto)) # time to mid-stance, phase 2
            tf2 = mylog((γ + √inroot)/(gto)) # time to mid-stance, phase 2
        else # inroot negative,
            tf2 = 1e4 - inroot # more negative inroot extends time
        end

    end
    twicenextenergy = (2g*L*cos(θnew-γ)+L^2*Ωplus^2-2g*L*cos(-γ)) # to find next vm
    if twicenextenergy >= 0.
        vmnew = √twicenextenergy
    elseif safety # not enough energy
        vmnew = (1e-3)*exp(twicenextenergy)
    else # no safety, not enough energy
        vmnew = √twicenextenergy # this should fail
    end

    # Step metrics
    steplength = 2L*sin(newα) # rimless wheel step length
    tf = tf1 + tf2 # total time mid-stance to mid-stance
    speed = steplength / tf # average speed mid-stance to mid-stance
    return (vm=vmnew, θnew=θnew, tf=tf, P=P, C=C, Pwork=Pwork,Cwork=Cwork,
        speed=speed, steplength=steplength, stepfrequency=speed/steplength,tf1=tf1, tf2=tf2,
        Ωminus=Ωminus,Ωplus=Ωplus,
        vm0=vm,δ=δangle)
end