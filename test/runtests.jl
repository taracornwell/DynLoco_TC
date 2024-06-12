using NewPackage
using Test

@testset "NewPackage.jl" begin
    # Write your tests here.
    wrw = WalkRW2l(P=0.15, vm = 0.38)
    @test typeof(wrw) == WalkRW2l # basic constructor for Walk

    wstar = findgait(wrw, target=:speed=>0.4, varying=:P) # constraint solver
    @test typeof(wstar) == WalkRW2l

    @test typeof(onestep(wrw)) <: NamedTuple # onestep returns a named tuple

    @test onestep(wstar).vm ≈ wstar.vm # limit cycle
    @test islimitcycle(wstar) # a built-in limit cycle tester


    @test islimitcycle(findgait(wrw, P=0.2, target=:speed=>0.4, varying=:P)) # vary push-off
    @test islimitcycle(findgait(wrw; vm=0.35, P=0.15, α=0.35, :γ=>0.15)) # vary gravity to get limit cycle
    @test islimitcycle(findgait(wrw, target=:speed=>0.45, varying=:γ, P=0)) # gravity only, no push-off
    @test islimitcycle(findgait(wrw, target=:vm=>0.4, varying=:P)) # use vm as target
    @test islimitcycle(findgait(wrw, α=0.32, target=(:speed=>0.4,), varying=(:P,), vm=0.25, P=0.2)) # use tuple of targets
    @test islimitcycle(findgait(wrw, (:speed=>0.4,), (:P,))) # non-keyword version of findgait

    # onestep test safe step with too little momentum
    @test typeof(onestep(wrw, P=0., vm=0.1, safety=true)) <: NamedTuple # should fail if safety=false
    @test_throws DomainError onestep(wrw, P=0., vm=0.1, safety=false)

    @test typeof(multistep(wstar, [0.15 0.17 0.18])) <: MultiStepResults

    # optwalkslope finds an optimal terrain, and optwalk just finds optimal push-offs.
    # Plug one into the other, they should yield the same push-offs and work
    sloperesult = optwalkslope(wstar, 6, boundaryvels = (0., 0.), boundarywork= true, symmetric = false)
    checkresult = optwalk(wstar, 6, boundaryvels=(0.,0.), totaltime=sloperesult.totaltime,
        δs=sloperesult.δangles)
    @test sloperesult.totaltime ≈ onestep(wstar).tf*6 # defaults to N steps of steady walking
    @test checkresult.totaltime ≈ sloperesult.totaltime # walk time should be equal
    @test isapprox(sloperesult.steps.P, checkresult.steps.P, atol=10*sqrt(eps(sloperesult.steps.P[1]))) # push-offs should be approx equal, with a slightly relaxed tolerance
    @test sloperesult.steps.vm0 ≈ checkresult.steps.vm0 # mid-stance velocities should be equal
    msr = multistep(wstar, sloperesult.steps.P, sloperesult.steps.δ, boundaryvels=(0.,0.))
    @test msr.steps.Pwork ≈ sloperesult.steps.Pwork

    # optwalktime

    # logshave
end
