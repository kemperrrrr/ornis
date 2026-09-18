//! Headless audit harness for the unmodified AVBD demo3d core (7701bd4).
//! Calls actual Manifold::updatePrimal/updateDual; this is not a build of Ornis.
#include <cassert>
#include <cstdio>
#include "solver.h"

static Manifold* fixture(Solver& solver) {
    auto* a = new Rigid(&solver, {1, 1, 1}, 1, 0.5f, {0, 0, 0});
    auto* b = new Rigid(&solver, {1, 1, 1}, 0, 0.5f, {0, 0, 0});
    a->initialLin = b->initialLin = {0, 0, 0};
    a->initialAng = b->initialAng = {0, 0, 0, 1};
    auto* m = new Manifold(&solver, a, b);
    m->basis = {0, 1, 0, 1, 0, 0, 0, 0, 1};
    m->numContacts = 1;
    m->friction = 0.5f;
    m->contacts[0] = {};
    return m;
}

int main() {
    {
        Solver solver;
        auto* m = fixture(solver);
        const float theta = -0.001f;
        m->bodyA->positionAng = {0, 0, sinf(theta/2), cosf(theta/2)};
        m->contacts[0].rA = {1, 0, 0};
        m->contacts[0].penalty = {1, 0, 0};
        float3x3 ll{}, la{}, lc{};
        float3 rl{}, ra{};
        m->updatePrimal(m->bodyA, solver.alpha, ll, la, lc, rl, ra);
        const float ornis_scalar = sinf(theta) + cosf(theta) * theta;
        assert(rl.y < -0.000999f && rl.y > -0.001001f);
        assert(ornis_scalar / rl.y > 1.999f && ornis_scalar / rl.y < 2.001f);
        printf("rotation: reference C=%.9g; Ornis scalar formula=%.9g; ratio=%.9g\n", rl.y, ornis_scalar, ornis_scalar/rl.y);
    }
    {
        Solver solver;
        auto* m = fixture(solver);
        m->bodyA->positionLin = {0.1f, -0.01f, 0};
        m->contacts[0].penalty = {100, 100, 100};
        m->contacts[0].lambda = {-9, 4, 0};
        m->updateDual(solver.alpha);
        const auto& p = m->contacts[0];
        assert(fabsf(p.lambda.x + 10) < 1e-5f && fabsf(p.lambda.y - 5) < 1e-5f);
        assert(p.penalty.y == 100);
        printf("sliding: reference lambda=(%.9g,%.9g,%.9g); tangent penalty=%.9g\n", p.lambda.x,p.lambda.y,p.lambda.z,p.penalty.y);
    }
    {
        Solver solver;
        auto* m = fixture(solver);
        m->contacts[0].penalty = {100, 100, 100};
        m->contacts[0].lambda = {-10, 0, 0};
        float3x3 ll{}, la{}, lc{};
        float3 rl{}, ra{};
        m->updatePrimal(m->bodyA, solver.alpha, ll, la, lc, rl, ra);
        assert(rl.y == -10 && ll[1][1] == 100);
        printf("C=0: reference gradient_y=%.9g; hessian_yy=%.9g\n", rl.y, ll[1][1]);
    }
}
