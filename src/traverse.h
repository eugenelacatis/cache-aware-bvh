#pragma once

// Ray-box, ray-triangle, and closest-hit traversal for 2-wide and 4-wide trees.
// Plain C++ marked HD, so the same code runs in CUDA kernels and CPU tests.

#include <cmath>
#include <cstdint>

#include "bvh_format.h"
#include "hd.h"

struct Hit {
    float t;
    uint32_t triId;  // Triangle::id; UINT32_MAX when nothing was hit
};

struct RayInv {
    float org[3];
    float inv[3];
};

HD inline RayInv makeRayInv(const Ray& r) {
    RayInv ri;
    for (int a = 0; a < 3; ++a) {
        ri.org[a] = r.org[a];
        ri.inv[a] = 1.0f / r.dir[a];
    }
    return ri;
}

// Slab test. Returns true and the entry distance when the ray crosses `b`
// within [tmin, tmax].
HD inline bool intersectBox(const Box& b, const RayInv& ri, float tmin, float tmax, float& tEnter) {
    for (int a = 0; a < 3; ++a) {
        float t0 = (b.lo[a] - ri.org[a]) * ri.inv[a];
        float t1 = (b.hi[a] - ri.org[a]) * ri.inv[a];
        tmin = fmaxf(tmin, fminf(t0, t1));
        tmax = fminf(tmax, fmaxf(t0, t1));
    }
    tEnter = tmin;
    return tmin <= tmax;
}

// Moller-Trumbore. Returns true and t when the ray hits `tri` within (tmin, tmax).
HD inline bool intersectTriangle(const Triangle& tri, const Ray& r, float tmin, float tmax, float& t) {
    float e1[3], e2[3], s[3];
    for (int a = 0; a < 3; ++a) {
        e1[a] = tri.v1[a] - tri.v0[a];
        e2[a] = tri.v2[a] - tri.v0[a];
        s[a] = r.org[a] - tri.v0[a];
    }
    float p[3] = {r.dir[1] * e2[2] - r.dir[2] * e2[1], r.dir[2] * e2[0] - r.dir[0] * e2[2],
                  r.dir[0] * e2[1] - r.dir[1] * e2[0]};
    float det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
    if (fabsf(det) < 1e-12f) return false;
    float invDet = 1.0f / det;
    float u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * invDet;
    if (u < 0.0f || u > 1.0f) return false;
    float q[3] = {s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]};
    float v = (r.dir[0] * q[0] + r.dir[1] * q[1] + r.dir[2] * q[2]) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;
    t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * invDet;
    return t > tmin && t < tmax;
}

HD inline const Triangle* leafTriangles(const unsigned char* triSection, uint32_t ref) {
    return reinterpret_cast<const Triangle*>(triSection + uint64_t(refLeafSector(ref)) * kSectorBytes);
}

// Closest-hit traversal. Children are pushed far to near so the nearest box is
// popped first. `sceneBox` is the root's box from the file header.
template <int W>
HD inline Hit traceClosest(const typename NodeOf<W>::type* nodes, const unsigned char* triSection, const Box& sceneBox,
                           const Ray& ray) {
    Hit hit{ray.tmax, UINT32_MAX};
    RayInv ri = makeRayInv(ray);
    float tEnter;
    if (!intersectBox(sceneBox, ri, ray.tmin, ray.tmax, tEnter)) return hit;

    uint32_t stackRef[kMaxStack];
    float stackT[kMaxStack];
    int sp = 0;
    stackRef[sp] = 0;  // root
    stackT[sp++] = tEnter;

    while (sp > 0) {
        --sp;
        uint32_t ref = stackRef[sp];
        if (stackT[sp] > hit.t) continue;

        if (refIsLeaf(ref)) {
            const Triangle* tris = leafTriangles(triSection, ref);
            uint32_t n = refLeafCount(ref);
            for (uint32_t i = 0; i < n; ++i) {
                float t;
                if (intersectTriangle(tris[i], ray, ray.tmin, hit.t, t)) {
                    hit.t = t;
                    hit.triId = tris[i].id;
                }
            }
            continue;
        }

        const auto& node = nodes[refNode(ref)];
        uint32_t hitRef[W];
        float hitT[W];
        int count = 0;
        for (int c = 0; c < W; ++c) {
            if (refIsEmpty(node.child[c])) break;
            float t;
            if (!intersectBox(node.box[c], ri, ray.tmin, hit.t, t)) continue;
            // Insertion sort, farthest first.
            int j = count++;
            while (j > 0 && hitT[j - 1] < t) {
                hitT[j] = hitT[j - 1];
                hitRef[j] = hitRef[j - 1];
                --j;
            }
            hitT[j] = t;
            hitRef[j] = node.child[c];
        }
        for (int i = 0; i < count; ++i) {  // validateTree guarantees room
            stackRef[sp] = hitRef[i];
            stackT[sp++] = hitT[i];
        }
    }
    return hit;
}
