export optwalktriangle

"""
    optwalktriangle(w::Walk, numsteps=5)

Accelerate at constant rate until a peak and then come right back down.    
Minimizes variance to walk `numsteps` steps. Returns a `MultiStepResults`
struct. As in a "tight regulation" control. Allows slopes to be specified as keyword `δs` array of the slope of each successive
step.

Other keyword arguments: `boundaryvels = (vm,vm)` can specify a tuple of initial and
final speeds, default nominal middle-stance speed `vm`. To start and end at rest, use `(0,0)`.
`boundarywork = true` whether cost includes the work needed to
start and end from `boundaryvels`. `totaltime` is the total time to take the steps, by default
the same time expected of the nominal `w` on level ground.
"""
function optwalktriangle(w::W, numsteps=5; A = 2.5*w.vm/totaltime, boundaryvels::Union{Tuple,Nothing} = nothing,
    boundarywork::Union{Tuple{Bool,Bool},Bool} = (true,true), totaltime=numsteps*onestep(w).tf,
    δs = zeros(numsteps)) where W <: Walk # default to taking the time of regular steady walking

    halfsteps = div(numsteps+2, 2)
    optsteps = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0))
    @variable(optsteps, P[1:numsteps]>=0, start=w.P) # JuMP variables P
    guessv = [[3*w.vm*i/numsteps for i in 1:halfsteps]; 
        [3*w.vm*(-i+numsteps)/numsteps for i in halfsteps+1:numsteps]]
    @variable(optsteps, v[1:numsteps+1]>=0, start=w.vm) # mid-stance speeds
#    @variable(optsteps, A>=0, start=4*w.vm/totaltime) # accelerate rate, which we want to stay near

    if boundaryvels === nothing || isempty(boundaryvels)
        boundaryvels = (w.vm, w.vm) # default to given gait if nothing specified
    end

    if typeof(boundarywork) <: Bool
        boundarywork = (boundarywork, boundarywork)
    end

    if !boundarywork[1] # no hip work at beginning or end; apply boundary velocity constraints
        @constraint(optsteps, v[1] == boundaryvels[1])
    end
    if !boundarywork[2]
        @constraint(optsteps, v[numsteps+1] == boundaryvels[2])
    end

    # Constraints
    # produce separate functions for speeds and step times
    register(optsteps, :onestepv, 3, # velocity after a step
        (v,P,δ)->onestep(w,P=P,vm=v, δangle=δ).vm, autodiff=true) # output vm
    register(optsteps, :onestept, 3, # time after a step
        (v,P,δ)->onestep(w,P=P,vm=v, δangle=δ).tf, autodiff=true)
    @NLexpression(optsteps, summedtime, # add up time of all steps
        sum(onestept(v[i],P[i],δs[i]) for i = 1:numsteps))
    @NLexpression(optsteps, nominaltime, onestept(w.vm,w.P,0)) # nominaltime
    @NLexpression(optsteps, variance,
        sum((onestepv(v[i],P[i],δs[i])-w.vm)^2 for i=1:numsteps))
    for i = 1:numsteps  # step dynamics
        @NLconstraint(optsteps, v[i+1]==onestepv(v[i],P[i],δs[i]))
    end
    @NLconstraint(optsteps, summedtime == totaltime) # total time

    @NLobjective(optsteps, Min, 
        sum(((v[i]-v[i-1])/onestept(v[i-1],P[i-1],δs[i-1]) - A)^2 for i=2:halfsteps) +
        sum(((v[i]-v[i-1])/onestept(v[i-1],P[i-1],δs[i-1]) + A)^2 for i=halfsteps+1:numsteps))

    optimize!(optsteps)
    if termination_status(optsteps) == MOI.LOCALLY_SOLVED || termination_status(optsteps) == MOI.OPTIMAL
        optimal_solution = (vms=value.(v), Ps=value.(P))
    else
        error("The model was not solved correctly.")
        println(termination_status(optsteps))
    end

    return multistep(W(w,vm=value(v[1])), value.(P), δs, vm0=value(v[1]), boundaryvels=boundaryvels) #, optimal_solution
end