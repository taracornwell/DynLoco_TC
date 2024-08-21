# WalkRW2ls, linearized rimless wheel with step length changing linearly with speed
# WalkRW2lvs, linearzied rimless wheel with step length changing nonlinearly with speed 
#   according to Grieve eqn

export WalkRW2ls, WalkRW2lvs

# variable step lengths abstract type
abstract type AbstractWalkRW2vs <: AbstractWalkRW2 end

@with_kw struct WalkRW2ls <: AbstractWalkRW2vs
    vm = 0.24972422206418757 # initial mid-stance velocity (stance leg upright)
    P = 0.15641662809369772   # push-off impulse (mass-normalized, units of deltav)
    alpha = 0.3 # angle between legs
    gamma = 0.  # downward slope
    L = 1.  # leg length
    M = 1.  # body mass
    g = 1.  # gravitational acceleration
    cstep = 0. # how step length increases with speed
    vmstar = 0.
    parms = (:alpha, :gamma, :L, :M, :g) # a list of the model parameters
    limitcycle = (parms = (:vm,), f = w -> onestep(w).vm - w.vm)
    safety = false # safety keeps model from falling backward if not enough momentum
end

# example of cstep, s = s0 + cstep(v-v0)
# where s = 1*v^beta, ds = beta*v^(beta-1)*dv
# ds = 0.42*(0.4)^(0.42-1) = 0.71
# divide by two to get 0.35

function onestep(w::WalkRW2ls; vm=w.vm, P=w.P, deltaangle = 0.,
    alpha=w.alpha, gamma=w.gamma,g=w.g,L=w.L,safety=w.safety, cstep=w.cstep, vmstar=w.vmstar, pert=1.)
    mylog = safety ? logshave : log # logshave doesn't blow up on negative numbers
    # Start at mid-stance leg vertical, and find the time tf1
    # to heelstrike, and angular velocity omegaminus
    # deltaangle is the upward change in slope from nominal gamma
    # Phase 1: From mid-stance to just before impact
    # theta is angle ccw from surface normal (which may be at slope)
    newalpha = alpha + cstep*(vm-vmstar) # where s = v^0.42, ds = 0.42*v^(0.42-1)*dv, ds/dv = 0.71 for typical
    # and s = 2*L*sin(alpha), so dalpha ≈ 0.35*dv
    thetaf = -newalpha + deltaangle # angle of stance leg just before impact, negative means cw
    # angle wrt vertical is -alpha-gamma+deltaangle
    #omegaminus = -√(2*g*(cos(0)-cos(thetaf))+L*vm^2)/√L  # pre-impact speed
    omegaminus = -√(2*g/L*(cos(-gamma)-cos(thetaf-gamma))+(vm/L)^2)  # pre-impact speed
    tf1 = mylog((L*(gamma-thetaf)+√(vm^2-2gamma*thetaf*L^2+thetaf^2*L^2))/(vm+L*gamma)) # time until impact, phase 1
    # Step-to-step transition: Push-off and collision
    Pwork = 1/2 * P^2
    v0 = √(omegaminus^2 + P^2) # intermediate velocity after push-off
    omegaplus = cos(2newalpha)*omegaminus - P*sin(2newalpha) # initial ang vel of next leg
    Cwork = 1/2*(omegaplus^2 - v0^2) # negative collision work
    C = √(-2Cwork)
    # Phase 2: From just after impact to next mid-stance
    thetanew = newalpha + deltaangle # new stance leg's initial angle
    #    tf2 = mylog((gamma + √(2gamma*thetanew-thetanew^2+omegaplus^2))/(gamma-thetanew-omegaplus))
    gto = gamma - thetanew - omegaplus # needs to be positive for pendulum to reach next mid-stance
    if safety && gto <= 0
        tf2 = 1e3 - gto # more negative gto is worse, increasing a tf2 time
    else # safety off OR gto positive
        # time to mid-stance, phase 2
        inroot = (2gamma*thetanew-thetanew^2+omegaplus^2)
        if inroot >= 0
#            tf2 = mylog((gamma + √(2gamma*thetanew-thetanew^2+omegaplus^2))/(gto)) # time to mid-stance, phase 2
            tf2 = mylog((gamma + √inroot)/(gto)) # time to mid-stance, phase 2
        else # inroot negative,
            tf2 = 1e4 - inroot # more negative inroot extends time
        end
    end

    twicenextenergy = (2g*L*cos(thetanew-gamma)+L^2*omegaplus^2-2g*L*cos(-gamma)) # to find next vm
    if twicenextenergy >= 0.
        vmnew = √twicenextenergy
    elseif safety # not enough energy
        vmnew = (1e-3)*exp(twicenextenergy)
    else # no safety, not enough energy
        vmnew = √twicenextenergy # this should fail
    end

    # multiply new midstance speed vm by perturbation size
    vmnew = pert*vmnew

    # Step metrics
    steplength = 2L*sin(newalpha) # rimless wheel step length
    tf = tf1 + tf2 # total time mid-stance to mid-stance
    speed = steplength / tf # average speed mid-stance to mid-stance
    return (vm=vmnew, thetanew=thetanew, tf=tf, P=P, C=C, Pwork=Pwork,Cwork=Cwork,
        speed=speed, steplength=steplength, stepfrequency=speed/steplength,tf1=tf1, tf2=tf2,
        omegaminus=omegaminus,omegaplus=omegaplus,
        vm0=vm,delta=deltaangle)
end

# step length varying s = c*v^beta
@with_kw struct WalkRW2lvs <: AbstractWalkRW2vs 
    vm = 0.35 # initial mid-stance velocity (stance leg upright)
    P = 0.1   # push-off impulse (mass-normalized, units of deltav)
    alpha = 0.3 # angle between legs
    gamma = 0.  # downward slope
    L = 1.  # leg length
    M = 1.  # body mass
    g = 1.  # gravitational acceleration
    # beta = 0.42 # s = c*v^beta
    beta = 0.54 # value updated from Collins and Kuo 2013
    # c = 1. # how step length increases with speed
    c = 1.22 # value updated from Collins and Kuo 2013
    vmstar = 0.
    parms = (:alpha, :gamma, :L, :M, :g) # a list of the model parameters
    limitcycle = (parms = (:vm,), f = w -> onestep(w).vm - w.vm)
    safety = false # safety keeps model from falling backward if not enough momentum
end


function onestep(w::WalkRW2lvs; vm=w.vm, P=w.P, deltaangle = 0.,
    alpha=w.alpha, gamma=w.gamma,g=w.g,L=w.L,safety=w.safety, c=w.c, vmstar=w.vmstar, beta=w.beta, pert=1.)
    mylog = safety ? logshave : log # logshave doesn't blow up on negative numbers
    # Start at mid-stance leg vertical, and find the time tf1
    # to heelstrike, and angular velocity omegaminus
    # deltaangle is the upward change in slope from nominal gamma
    # Phase 1: From mid-stance to just before impact
    # theta is angle ccw from surface normal (which may be at slope)
    newsteplength = c*vm^beta # Grieve: SL = c*v^beta 
    # newalpha = newsteplength / (2L)
    # newalpha = (c*vm^beta)/(2L)
    newalpha = asin(newsteplength/(2L))
    thetaf = -newalpha + deltaangle # angle of stance leg just before impact, negative means cw
    # angle wrt vertical is -alpha-gamma+deltaangle
    #omegaminus = -√(2*g*(cos(0)-cos(thetaf))+L*vm^2)/√L  # pre-impact speed
    omegaminus = -√(2*g/L*(cos(-gamma)-cos(thetaf-gamma))+(vm/L)^2)  # pre-impact speed
    tf1 = mylog((L*(gamma-thetaf)+√(vm^2-2gamma*thetaf*L^2+thetaf^2*L^2))/(vm+L*gamma)) # time until impact, phase 1
    # Step-to-step transition: Push-off and collision
    Pwork = 1/2 * P^2
    v0 = √(omegaminus^2 + P^2) # intermediate velocity after push-off
    omegaplus = cos(2newalpha)*omegaminus - P*sin(2newalpha) # initial ang vel of next leg
    Cwork = 1/2*(omegaplus^2 - v0^2) # negative collision work
    C = √(-2Cwork)
    # Phase 2: From just after impact to next mid-stance
    thetanew = newalpha + deltaangle # new stance leg's initial angle
    #    tf2 = mylog((gamma + √(2gamma*thetanew-thetanew^2+omegaplus^2))/(gamma-thetanew-omegaplus))
    gto = gamma - thetanew - omegaplus # needs to be positive for pendulum to reach next mid-stance
    if safety && gto <= 0
        tf2 = 1e3 - gto # more negative gto is worse, increasing a tf2 time
    else # safety off OR gto positive
        # time to mid-stance, phase 2
        inroot = (2gamma*thetanew-thetanew^2+omegaplus^2)
        if inroot >= 0
#            tf2 = mylog((gamma + √(2gamma*thetanew-thetanew^2+omegaplus^2))/(gto)) # time to mid-stance, phase 2
            tf2 = mylog((gamma + √inroot)/(gto)) # time to mid-stance, phase 2
        else # inroot negative,
            tf2 = 1e4 - inroot # more negative inroot extends time
        end
    end
    # divide time for second phase by perturbation size (since it correlates to speed change)
    tf2 = tf2/pert
    twicenextenergy = (2g*L*cos(thetanew-gamma)+L^2*omegaplus^2-2g*L*cos(-gamma)) # to find next vm
    if twicenextenergy >= 0.
        vmnew = √twicenextenergy
    elseif safety # not enough energy
        vmnew = (1e-3)*exp(twicenextenergy)
    else # no safety, not enough energy
        vmnew = √twicenextenergy # this should fail
    end

    # multiply new midstance speed vm by perturbation size
    vmnew = pert*vmnew
    
    # Step metrics
    steplength = 2L*sin(newalpha) # rimless wheel step length
    tf = tf1 + tf2 # total time mid-stance to mid-stance
    speed = steplength / tf # average speed mid-stance to mid-stance
    return (vm=vmnew, thetanew=thetanew, tf=tf, P=P, C=C, Pwork=Pwork,Cwork=Cwork,
        speed=speed, steplength=steplength, stepfrequency=speed/steplength,tf1=tf1, tf2=tf2,
        omegaminus=omegaminus,omegaplus=omegaplus,
        vm0=vm,delta=deltaangle,pert=pert)
end
