#include "raygen.h"

#include <cmath>
#include <random>

std::vector<Ray> isotropicLines(const Box& box, uint32_t count, uint64_t seed) {
    float center[3];
    float radius2 = 0.0f;
    for (int a = 0; a < 3; ++a) {
        center[a] = 0.5f * (box.lo[a] + box.hi[a]);
        float half = 0.5f * (box.hi[a] - box.lo[a]);
        radius2 += half * half;
    }
    const float radius = std::sqrt(radius2);
    const float kPi = 3.14159265358979f;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::vector<Ray> rays(count);
    for (Ray& r : rays) {
        // Direction uniform on the unit sphere.
        float z = 1.0f - 2.0f * uni(rng);
        float phi = 2.0f * kPi * uni(rng);
        float s = std::sqrt(std::fmax(0.0f, 1.0f - z * z));
        float d[3] = {s * std::cos(phi), s * std::sin(phi), z};

        // Orthonormal basis (u, v) perpendicular to d.
        float u[3];
        if (std::fabs(d[0]) < 0.9f) {
            u[0] = 0.0f; u[1] = d[2]; u[2] = -d[1];  // d x (1,0,0)
        } else {
            u[0] = -d[2]; u[1] = 0.0f; u[2] = d[0];  // d x (0,1,0)
        }
        float un = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        for (float& x : u) x /= un;
        float v[3] = {d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0]};

        // Offset uniform on the disk of radius R perpendicular to d. Lines with
        // uniform direction and uniform offset are uniform in the line measure.
        float rho = radius * std::sqrt(uni(rng));
        float theta = 2.0f * kPi * uni(rng);
        float a = rho * std::cos(theta);
        float b = rho * std::sin(theta);

        // Start where the line enters the sphere.
        float back = std::sqrt(std::fmax(0.0f, radius2 - rho * rho));
        for (int k = 0; k < 3; ++k) {
            r.org[k] = center[k] + a * u[k] + b * v[k] - back * d[k];
            r.dir[k] = d[k];
        }
        r.tmin = 0.0f;
        r.tmax = 2.0f * radius;
    }
    return rays;
}
