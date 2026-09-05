#pragma once

#include "core/geometry.hpp"
#include "core/physics.hpp"
#include "env/contracts.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// Observation featurization for the RL env. This header must stay free of
// rendering and Python dependencies: the Geode mod includes it verbatim so
// the deployed policy sees bit-identical features built from real-game state.
// All normalization constants are fixed at compile time — no running stats.
namespace gdenv {

// Player-anchored fractional occupancy grid: x spans [player.x, player.x+480)
// (~1.5 s of travel at 0.9x speed), y spans the absolute band [90, 450) —
// ground top is 90, ship ceiling 390. Cells are 15x15 (half a block): at
// 30x30 a spike's 6x12 hazard box covered only 0.08 of its cell, an
// amplitude the policy net measurably ignored next to the O(1) scalar
// features; 15x15 keeps sub-cell smoothness (fractional coverage) while a
// spike now registers ~0.32. Two channels, solid and hazard, each cell
// holding covered-area fraction in [0,1].
constexpr int kGridCols = 32;
constexpr int kGridRows = 24;
constexpr float kCellW = 15.0f;
constexpr float kCellH = 15.0f;
constexpr float kGridYBase = 90.0f;
constexpr float kLookahead = kGridCols * kCellW; // 480
constexpr int kGridDim = kGridRows * kGridCols;  // per channel
// Obs v2: scalars 12-17 appended (overhead solid, second hazard, dx,
// reserved). Obs v3 (Back On Track): scalar 17 becomes the gravity flag
// (it was the reserved mode slot, always 0 before) and scalars 18-23 are
// appended — the next two activatable objects ahead (pads / gravity
// portals): distance, dy, kind. Obs v4 (Polargeist): jump rings join the
// specials list (kind +0.25) and scalars 24-25 are appended — "a press now
// fires a ring" and "the held press is still unconsumed" (the two bits of
// PlayerObject state that decide what a click does). Cycles (ball mode)
// adds new VALUES in existing slots, no layout change: scalar 3 (mode) is
// -1 for the ball, scalar 7 (next mode portal) +0.5 for a ball portal, and
// scalar 4 (button) is 0.5 while the physical button is down but
// m_isHolding is already spent (a ball flip / ball ring consumed it, so a
// sustained hold is a no-op until a release + press). Levels 1-8 never
// emit any of these. Obs v5 (Clutterfunk): scalar 26 is appended — the
// mini flag (m_vehicleSize 0.6: 18x18 box, 8.944 jump, 0.85-divided ship
// increment, floor line 99) — and the size portals join the specials list
// (kind +0.9 back to normal size, -0.9 mini); circular hazards (sawblades)
// enter the hazard grid and the hazard scalars as their bounding squares.
// Theory of Everything (UFO mode) again adds only VALUES: scalar 3 (mode)
// is +0.5 for the UFO, scalar 7 (next mode portal) -0.5 for a UFO portal;
// levels 1-11 never emit either. Scalars 0-26 and the grid layout are
// frozen — additions bump kObsVersion and only ever append.
// gdenv_obs_version() exposes this so the Python side can refuse a stale
// DLL.
constexpr int kObsVersion = kObsVersionLegacy; // 5: the frozen layout below
constexpr int kPlayerDim = 27;
constexpr int kObsDim = kPlayerDim + 2 * kGridDim;

// Obs v6 (2026-09-05, docs/rl-improvement-plan.md). Same 24x32 grid of
// 15x15 cells and the same two channels, but the window MOVES WITH THE
// PLAYER: x offsets [-30, 450), y offsets [-180, 180). Theory of
// Everything's UFO corridor sits at y 1011-1155 and every one of its 311
// decisions saw an empty v5 grid (absolute band [90, 450)). Scalars 0-26
// keep their v5 meaning; 27-36 are appended (floor / ceiling clearance,
// which boundary lines apply, collision size, gravity-flip grace, landing
// snap memory, the max-y warning flag). Then an exact geometry block: the
// four nearest hazards, four nearest solids and four nearest interactive
// objects (pads / rings / gravity, size and mode portals) inside the
// window, each as [mask, left, bottom, right, top, cx, cy, radius, kind,
// state] relative to the player, ordered by distance to the player's box
// (ties by object order), zero rows masked. Layout:
//   [0, kPlayerDimV6)              scalars
//   [kPlayerDimV6, +kGeomDim)      geometry rows
//   [.., +2*kGridDim)              solid grid, hazard grid ([row][col])
constexpr float kGridXOffV6 = -30.0f;
constexpr float kGridYOffV6 = -180.0f;
constexpr int kPlayerDimV6 = 37;
constexpr int kGeomPerClass = 4;
constexpr int kGeomRows = 3 * kGeomPerClass; // hazards, solids, interactives
constexpr int kGeomFloats = 10;
constexpr int kGeomDim = kGeomRows * kGeomFloats; // 120
constexpr int kObsDimV6 = kPlayerDimV6 + kGeomDim + 2 * kGridDim; // 1693

inline int obsDimFor(int version) {
    return version == kObsVersionV6 ? kObsDimV6 : kObsDim;
}

namespace detail {

// Accumulate one sorted-by-left box list into a [row][col] cell block whose
// origin (bottom-left) is (ox, oy).
inline void rasterizeAt(const std::vector<gdcore::Box>& boxes, float ox,
                        float oy, float* cells) {
    // Boxes are at most 66 wide (the sawblade 88 bounding square; every
    // rectangle is <= 34), same slack as gdcore::detail::kHazardMaxW.
    auto it = std::lower_bound(
        boxes.begin(), boxes.end(), ox - 66.0f,
        [](const gdcore::Box& b, float v) { return b.left < v; });
    const float gx1 = ox + kLookahead;
    const float gy1 = oy + kGridRows * kCellH;
    for (; it != boxes.end() && it->left < gx1; ++it) {
        float l = std::max(it->left, ox);
        float r = std::min(it->right, gx1);
        float b = std::max(it->bottom, oy);
        float t = std::min(it->top, gy1);
        if (l >= r || b >= t) continue;
        int c0 = static_cast<int>((l - ox) / kCellW);
        int c1 = std::min(kGridCols,
                          static_cast<int>(std::ceil((r - ox) / kCellW)));
        int r0 = static_cast<int>((b - oy) / kCellH);
        int r1 = std::min(kGridRows,
                          static_cast<int>(std::ceil((t - oy) / kCellH)));
        for (int row = r0; row < r1; ++row) {
            float cy0 = oy + row * kCellH;
            float oh = std::min(t, cy0 + kCellH) - std::max(b, cy0);
            for (int col = c0; col < c1; ++col) {
                float cx0 = ox + col * kCellW;
                float ow = std::min(r, cx0 + kCellW) - std::max(l, cx0);
                float& cell = cells[row * kGridCols + col];
                cell = std::min(1.0f, cell + (ow * oh) / (kCellW * kCellH));
            }
        }
    }
}

inline void rasterize(const std::vector<gdcore::Box>& boxes, float px,
                      float* cells) {
    rasterizeAt(boxes, px, kGridYBase, cells);
}

} // namespace detail

// Layout: [0..kPlayerDim) player scalars, then the solid grid [row][col] and
// the hazard grid [row][col], kGridDim floats each. levelEndX is the win
// line (x at which the episode is won), used only for the progress scalar.
// dx is the per-tick x advance (PhysicsConfig::dx) — constant today, an
// input so speed portals only change a value, not the feature contract.
//
// The landing-pattern snap memory (hasPrevLand/prevLandX/...) is deliberately
// not observed: its effect on x is at most +-1 unit and x itself is observed.
inline void buildObs(const gdcore::PlayerState& s,
                     const gdcore::LevelGeometry& g, float levelEndX,
                     float dx, float* out) {
    out[0] = (s.y - 105.0f) / 300.0f; // 0 on ground, 0.9 at ship ceiling
    out[1] = static_cast<float>(s.yvel / 15.0);
    out[2] = s.onGround ? 1.0f : 0.0f;
    out[3] = s.ship ? 1.0f : s.ball ? -1.0f : s.bird ? 0.5f : 0.0f;
    // Previous input state matters: a fresh click and a sustained hold take
    // different paths through gdcore::step. 1 = button down and
    // m_isHolding set (a sustained hold: buffered cube re-jump / ball flip
    // on landing), 0.5 = button down but the press already spent (ball
    // only: the next action must be a release before a press does
    // anything), 0 = button up.
    out[4] = s.inputHeld ? (s.holding ? 1.0f : 0.5f) : 0.0f;
    out[5] = s.x / levelEndX;
    // Next mode portal ahead of the player, to anticipate the yvel-halving
    // mode switch before it enters the grid's coarse cells.
    float portalDist = 1.0f, portalType = 0.0f;
    auto it = std::lower_bound(
        g.portals.begin(), g.portals.end(), s.x,
        [](const gdcore::Portal& p, float v) { return p.box.left < v; });
    if (it != g.portals.end() && it->box.left - s.x < kLookahead) {
        portalDist = (it->box.left - s.x) / kLookahead;
        portalType = it->mode == gdcore::PortalMode::Ship   ? 1.0f
                     : it->mode == gdcore::PortalMode::Ball ? 0.5f
                     : it->mode == gdcore::PortalMode::Ufo  ? -0.5f
                                                            : -1.0f;
    }
    out[6] = portalDist;
    out[7] = portalType;

    // Salient threat scalars. The grid alone proved too diffuse for the MLP
    // to carve out the narrow "last chance to jump" states — a spike's
    // decision boundary is ~5 units wide, and smooth generalization from
    // neighboring don't-care states washes it out. These give the net the
    // critical variables directly; the Geode mod computes them identically.
    // Next two hazards ahead: nearest left edges at or past the player's
    // back edge. The second (obs v2) tells the net whether the landing zone
    // after the first threat is itself threatened.
    float hd = 1.0f, hdy = 0.0f, h2d = 1.0f, h2dy = 0.0f;
    {
        auto hit = std::lower_bound(
            g.hazards.begin(), g.hazards.end(), s.x - 15.0f,
            [](const gdcore::Box& b, float v) { return b.left < v; });
        if (hit != g.hazards.end() && hit->left - s.x < kLookahead) {
            hd = std::max(0.0f, hit->left - s.x) / kLookahead;
            hdy = (hit->bottom - s.y) / 300.0f;
            ++hit;
            if (hit != g.hazards.end() && hit->left - s.x < kLookahead) {
                h2d = std::max(0.0f, hit->left - s.x) / kLookahead;
                h2dy = (hit->bottom - s.y) / 300.0f;
            }
        }
    }
    // Next step-up solid ahead: nearest face whose top is above the player's
    // feet (something to jump onto or into — the stair blocks).
    float wd = 1.0f, wh = 0.0f;
    {
        auto sit = std::lower_bound(
            g.solids.begin(), g.solids.end(), s.x,
            [](const gdcore::Box& b, float v) { return b.left < v; });
        for (; sit != g.solids.end() && sit->left - s.x < kLookahead; ++sit) {
            if (sit->top > s.y + 6.0f) {
                wd = (sit->left - s.x) / kLookahead;
                wh = (sit->top - s.y) / 300.0f;
                break;
            }
        }
    }
    // Next overhead solid ahead (obs v2): nearest face whose bottom is above
    // the player — the down-facing mirror of the step-up scan. This is the
    // ship-corridor ceiling the grid renders too diffusely.
    float cd = 1.0f, ch = 0.0f;
    {
        auto cit = std::lower_bound(
            g.solids.begin(), g.solids.end(), s.x,
            [](const gdcore::Box& b, float v) { return b.left < v; });
        for (; cit != g.solids.end() && cit->left - s.x < kLookahead; ++cit) {
            if (cit->bottom > s.y + 6.0f) {
                cd = (cit->left - s.x) / kLookahead;
                ch = (cit->bottom - s.y) / 300.0f;
                break;
            }
        }
    }
    out[8] = hd;
    out[9] = hdy;
    out[10] = wd;
    out[11] = wh;
    out[12] = cd;
    out[13] = ch;
    out[14] = h2d;
    out[15] = h2dy;
    // Speed as a feature: constant while every level runs at one dx, but the
    // slot means speed portals later change an input, not the network.
    out[16] = dx / 2.0f;
    // Gravity flag (obs v3; 0 in every level without gravity objects).
    out[17] = s.upsideDown ? 1.0f : 0.0f;
    // Next two activatable objects ahead (obs v3): pads relaunch the player
    // and gravity objects invert every vertical rule, so the policy needs
    // them as sharply as it needs hazards. dist = left edge from the player
    // (0 while overlapping), dy = box bottom relative to the player, kind:
    // yellow pad +1, gravity pad -1, normal-gravity portal +0.5,
    // flip-gravity portal -0.5, yellow jump ring +0.25 (obs v4), pink
    // jump pad +0.75, pink jump ring -0.25, gravity ring -0.75 (xStep; new
    // values in the existing kind slot, no layout change; levels 1-5 never
    // emit them).
    // Already-activated pads/rings are still reported (the featurizer has
    // no activation state); they sit behind or under the player only
    // briefly.
    {
        float sd[2] = {1.0f, 1.0f}, sy[2] = {0.0f, 0.0f}, sk[2] = {0.0f, 0.0f};
        auto sit = std::lower_bound(
            g.specials.begin(), g.specials.end(), s.x - 15.0f,
            [](const gdcore::Special& e, float v) { return e.box.left < v; });
        for (int k = 0; k < 2 && sit != g.specials.end() &&
                        sit->box.left - s.x < kLookahead;
             ++k, ++sit) {
            sd[k] = std::max(0.0f, sit->box.left - s.x) / kLookahead;
            sy[k] = (sit->box.bottom - s.y) / 300.0f;
            switch (sit->kind) {
                case gdcore::SpecialKind::YellowPad: sk[k] = 1.0f; break;
                case gdcore::SpecialKind::GravityPad: sk[k] = -1.0f; break;
                case gdcore::SpecialKind::GravityNormal: sk[k] = 0.5f; break;
                case gdcore::SpecialKind::GravityFlip: sk[k] = -0.5f; break;
                case gdcore::SpecialKind::YellowRing: sk[k] = 0.25f; break;
                case gdcore::SpecialKind::PinkPad: sk[k] = 0.75f; break;
                case gdcore::SpecialKind::PinkRing: sk[k] = -0.25f; break;
                case gdcore::SpecialKind::GravityRing: sk[k] = -0.75f; break;
                case gdcore::SpecialKind::SizeNormal: sk[k] = 0.9f; break;
                case gdcore::SpecialKind::SizeMini: sk[k] = -0.9f; break;
            }
        }
        out[18] = sd[0];
        out[19] = sy[0];
        out[20] = sk[0];
        out[21] = sd[1];
        out[22] = sy[1];
        out[23] = sk[1];
    }
    // Ring input state (obs v4). [24]: a fresh press at this decision fires
    // a ring (the player box overlapped an unused ring this tick or last
    // tick — PlayerObject::m_touchingRings). [25]: the current hold still
    // carries an unconsumed press, so the next ring contact fires by
    // itself (m_isHolding && m_hasJustHeld). Both are exact engine state,
    // computable in the Geode mod from PlayerObject fields. (The ball's
    // flip clears both flags, so a ball that flipped reads 0 here even
    // while the button is held — the engine would not fire a ring either.)
    out[24] = gdcore::pressableRing(s) >= 0 ? 1.0f : 0.0f;
    out[25] = (s.holding && s.hasJustHeld) ? 1.0f : 0.0f;
    // Mini flag (obs v5; 0 in every level without size portals).
    out[26] = s.mini ? 1.0f : 0.0f;

    float* grid = out + kPlayerDim;
    std::memset(grid, 0, sizeof(float) * 2 * kGridDim);
    detail::rasterize(g.solids, s.x, grid);
    detail::rasterize(g.hazards, s.x, grid + kGridDim);
}

namespace detail {

// The v5 "kind" code of a special (scalars 20 / 23), shared with v6's
// geometry rows so the two layouts never disagree on a value.
inline float specialKindCode(gdcore::SpecialKind k) {
    switch (k) {
        case gdcore::SpecialKind::YellowPad: return 1.0f;
        case gdcore::SpecialKind::GravityPad: return -1.0f;
        case gdcore::SpecialKind::GravityNormal: return 0.5f;
        case gdcore::SpecialKind::GravityFlip: return -0.5f;
        case gdcore::SpecialKind::YellowRing: return 0.25f;
        case gdcore::SpecialKind::PinkPad: return 0.75f;
        case gdcore::SpecialKind::PinkRing: return -0.25f;
        case gdcore::SpecialKind::GravityRing: return -0.75f;
        case gdcore::SpecialKind::SizeNormal: return 0.9f;
        case gdcore::SpecialKind::SizeMini: return -0.9f;
    }
    return 0.0f;
}

// Mode portals share the interactive rows; codes distinct from every
// special code above.
inline float portalKindCode(gdcore::PortalMode m) {
    switch (m) {
        case gdcore::PortalMode::Cube: return 0.1f;
        case gdcore::PortalMode::Ship: return 0.2f;
        case gdcore::PortalMode::Ball: return 0.3f;
        case gdcore::PortalMode::Ufo: return 0.4f;
    }
    return 0.0f;
}

struct GeomPick {
    float d2 = 0.0f;
    int order = 0;
    const gdcore::Box* box = nullptr;
    float kind = 0.0f;
    float state = 0.0f;
};

// Squared distance between the player's box and `b` (0 when they overlap).
inline float boxDist2(float pl, float pb, float pr, float pt,
                      const gdcore::Box& b) {
    float dx = std::max(0.0f, std::max(b.left - pr, pl - b.right));
    float dy = std::max(0.0f, std::max(b.bottom - pt, pb - b.top));
    return dx * dx + dy * dy;
}

// Keep the kGeomPerClass nearest picks, ordered by (distance, object
// order): a plain insertion into a tiny sorted array.
inline void pickInsert(GeomPick* arr, int& n, const GeomPick& p) {
    int pos = n;
    while (pos > 0 && (p.d2 < arr[pos - 1].d2 ||
                       (p.d2 == arr[pos - 1].d2 && p.order < arr[pos - 1].order)))
        --pos;
    if (pos >= kGeomPerClass) return;
    int last = std::min(n, kGeomPerClass - 1);
    for (int i = last; i > pos; --i) arr[i] = arr[i - 1];
    arr[pos] = p;
    if (n < kGeomPerClass) ++n;
}

inline bool inWindow(const gdcore::Box& b, float x, float y) {
    return b.right >= x + kGridXOffV6 && b.left < x + kGridXOffV6 + kLookahead &&
           b.top >= y + kGridYOffV6 && b.bottom < y + kGridYOffV6 + kGridRows * kCellH;
}

inline void writeGeomRow(float* row, const GeomPick& p, float x, float y) {
    if (!p.box) {
        std::memset(row, 0, sizeof(float) * kGeomFloats);
        return;
    }
    row[0] = 1.0f;
    row[1] = (p.box->left - x) / kLookahead;
    row[2] = (p.box->bottom - y) / 300.0f;
    row[3] = (p.box->right - x) / kLookahead;
    row[4] = (p.box->top - y) / 300.0f;
    if (p.box->radius > 0.0f) {
        row[5] = (p.box->cx - x) / kLookahead;
        row[6] = (p.box->cy - y) / 300.0f;
        row[7] = p.box->radius / 60.0f;
    } else {
        row[5] = row[6] = row[7] = 0.0f;
    }
    row[8] = p.kind;
    row[9] = p.state;
}

inline bool ringTouching(const gdcore::PlayerState& s, int idx) {
    for (int i = 0; i < s.nRingPrev; ++i) if (s.ringPrev[i] == idx) return true;
    for (int i = 0; i < s.nRingNow; ++i) if (s.ringNow[i] == idx) return true;
    return false;
}

} // namespace detail

// Obs v6 builder (layout above). The scalar block starts with v5's 27
// scalars verbatim (buildObs), so the two layouts share one definition of
// every legacy feature.
inline void buildObsV6(const gdcore::PlayerState& s,
                       const gdcore::LevelGeometry& g,
                       const gdcore::PhysicsConfig& c, float levelEndX,
                       float* out) {
    // Scalars 0-26: exactly v5's (built into a scratch row; the grid tail
    // of that call is discarded).
    {
        float tmp[kObsDim];
        buildObs(s, g, levelEndX, c.dx, tmp);
        std::memcpy(out, tmp, sizeof(float) * kPlayerDim);
    }
    const float half = gdcore::playerHalf(s, c);
    // 27/28: clearance to the floor / ceiling line the current mode snaps
    // against (player center). Cube: the floor line (floorLine) and the
    // kMaxGameplayY kill line; flying / ball: the portal-installed band.
    float floorY, ceilY;
    bool floorApplies, ceilApplies;
    if (gdcore::flying(s) || s.ball) {
        const float bandH = s.ship ? 300.0f
                            : s.bird ? gdcore::ufo::kBandHeight
                                     : gdcore::ball::kBandHeight;
        floorY = s.bandFloor + half;
        ceilY = s.bandFloor + bandH - half;
        floorApplies = ceilApplies = true;
    } else {
        floorY = gdcore::floorLine(s, c);
        ceilY = gdcore::kMaxGameplayY + (c.half - half);
        // An upside-down cube only meets the floor line after the flip
        // grace (pad::kFlipFloorGraceTicks); the kill line is not a snap.
        floorApplies = !s.upsideDown ||
                       s.sinceFlip >= gdcore::pad::kFlipFloorGraceTicks;
        ceilApplies = false;
    }
    out[27] = std::max(-2.0f, std::min(2.0f, (s.y - floorY) / 300.0f));
    out[28] = std::max(-2.0f, std::min(2.0f, (ceilY - s.y) / 300.0f));
    out[29] = floorApplies ? 1.0f : 0.0f;
    out[30] = ceilApplies ? 1.0f : 0.0f;
    // 31: collision size — the CACHED rect's half extent (rectMini lags a
    // size portal by one position write), 1 normal / 0.6 mini.
    out[31] = half / c.half;
    // 32: gravity-flip grace, 1 right after a flip -> 0 once it has run out.
    out[32] = s.sinceFlip < gdcore::pad::kFlipFloorGraceTicks
                  ? 1.0f - static_cast<float>(s.sinceFlip) /
                               gdcore::pad::kFlipFloorGraceTicks
                  : 0.0f;
    // 33-35: landing-pattern snap memory (collidedWithObjectInternal tail):
    // whether a previous landing is remembered, the remembered x offset,
    // and where that object sits relative to the player now.
    out[33] = s.hasPrevLand ? 1.0f : 0.0f;
    out[34] = s.hasPrevLand ? static_cast<float>(s.landOffset) / 30.0f : 0.0f;
    out[35] = s.hasPrevLand ? (s.prevLandX - s.x) / kLookahead : 0.0f;
    // 36: the "one more tick above kMaxGameplayY kills" flag.
    out[36] = s.overMaxY ? 1.0f : 0.0f;

    // Geometry rows.
    const float pl = s.x - half, pb = s.y - half, pr = s.x + half,
                pt = s.y + half;
    detail::GeomPick haz[kGeomPerClass], sol[kGeomPerClass],
        act[kGeomPerClass];
    int nh = 0, ns = 0, na = 0;
    const float scanFrom = s.x + kGridXOffV6 - 66.0f;
    const float scanTo = s.x + kGridXOffV6 + kLookahead;
    {
        auto it = std::lower_bound(
            g.hazards.begin(), g.hazards.end(), scanFrom,
            [](const gdcore::Box& b, float v) { return b.left < v; });
        for (; it != g.hazards.end() && it->left < scanTo; ++it) {
            if (!detail::inWindow(*it, s.x, s.y)) continue;
            detail::GeomPick p;
            p.d2 = detail::boxDist2(pl, pb, pr, pt, *it);
            p.order = it->order;
            p.box = &*it;
            p.kind = it->radius > 0.0f ? -1.0f : 1.0f;
            p.state = 0.0f;
            detail::pickInsert(haz, nh, p);
        }
    }
    {
        auto it = std::lower_bound(
            g.solids.begin(), g.solids.end(), scanFrom,
            [](const gdcore::Box& b, float v) { return b.left < v; });
        for (; it != g.solids.end() && it->left < scanTo; ++it) {
            if (!detail::inWindow(*it, s.x, s.y)) continue;
            detail::GeomPick p;
            p.d2 = detail::boxDist2(pl, pb, pr, pt, *it);
            p.order = it->order;
            p.box = &*it;
            p.kind = 1.0f;
            // Contact: the boxes touch or overlap; -1 marks the object the
            // landing-snap memory refers to.
            const bool touching = p.d2 == 0.0f;
            const bool remembered =
                s.hasPrevLand &&
                std::fabs(0.5f * (it->left + it->right) - s.prevLandX) < 0.01f &&
                std::fabs(0.5f * (it->bottom + it->top) - s.prevLandY) < 0.01f;
            p.state = remembered ? -1.0f : touching ? 1.0f : 0.0f;
            detail::pickInsert(sol, ns, p);
        }
    }
    {
        auto it = std::lower_bound(
            g.specials.begin(), g.specials.end(), scanFrom,
            [](const gdcore::Special& e, float v) { return e.box.left < v; });
        const int pressable = gdcore::pressableRing(s);
        for (; it != g.specials.end() && it->box.left < scanTo; ++it) {
            if (!detail::inWindow(it->box, s.x, s.y)) continue;
            const int idx = static_cast<int>(it - g.specials.begin());
            detail::GeomPick p;
            p.d2 = detail::boxDist2(pl, pb, pr, pt, it->box);
            p.order = it->box.order;
            p.box = &it->box;
            p.kind = detail::specialKindCode(it->kind);
            // state: -1 = spent / no effect (one-shot already fired, or an
            // idempotent portal the player already satisfies), 1 = a press
            // now fires this ring, 0.5 = ring in the touching list, 0 =
            // live and untouched.
            float st = 0.0f;
            switch (it->kind) {
                case gdcore::SpecialKind::YellowPad:
                case gdcore::SpecialKind::PinkPad:
                case gdcore::SpecialKind::GravityPad:
                    st = gdcore::specialDone(s, idx) ? -1.0f : 0.0f;
                    break;
                case gdcore::SpecialKind::YellowRing:
                case gdcore::SpecialKind::PinkRing:
                case gdcore::SpecialKind::GravityRing:
                    st = gdcore::specialDone(s, idx) ? -1.0f
                         : pressable == idx          ? 1.0f
                         : detail::ringTouching(s, idx) ? 0.5f
                                                        : 0.0f;
                    break;
                case gdcore::SpecialKind::GravityNormal:
                    st = s.upsideDown ? 0.0f : -1.0f;
                    break;
                case gdcore::SpecialKind::GravityFlip:
                    st = s.upsideDown ? -1.0f : 0.0f;
                    break;
                case gdcore::SpecialKind::SizeNormal:
                    st = s.mini ? 0.0f : -1.0f;
                    break;
                case gdcore::SpecialKind::SizeMini:
                    st = s.mini ? -1.0f : 0.0f;
                    break;
            }
            p.state = st;
            detail::pickInsert(act, na, p);
        }
        auto pit = std::lower_bound(
            g.portals.begin(), g.portals.end(), scanFrom,
            [](const gdcore::Portal& e, float v) { return e.box.left < v; });
        for (; pit != g.portals.end() && pit->box.left < scanTo; ++pit) {
            if (!detail::inWindow(pit->box, s.x, s.y)) continue;
            detail::GeomPick p;
            p.d2 = detail::boxDist2(pl, pb, pr, pt, pit->box);
            p.order = pit->box.order;
            p.box = &pit->box;
            p.kind = detail::portalKindCode(pit->mode);
            const bool same =
                (pit->mode == gdcore::PortalMode::Ship && s.ship) ||
                (pit->mode == gdcore::PortalMode::Ball && s.ball) ||
                (pit->mode == gdcore::PortalMode::Ufo && s.bird) ||
                (pit->mode == gdcore::PortalMode::Cube && !s.ship && !s.ball &&
                 !s.bird);
            p.state = same ? -1.0f : 0.0f;
            detail::pickInsert(act, na, p);
        }
    }
    float* rows = out + kPlayerDimV6;
    for (int k = 0; k < kGeomPerClass; ++k) {
        detail::writeGeomRow(rows + (0 * kGeomPerClass + k) * kGeomFloats,
                             k < nh ? haz[k] : detail::GeomPick{}, s.x, s.y);
        detail::writeGeomRow(rows + (1 * kGeomPerClass + k) * kGeomFloats,
                             k < ns ? sol[k] : detail::GeomPick{}, s.x, s.y);
        detail::writeGeomRow(rows + (2 * kGeomPerClass + k) * kGeomFloats,
                             k < na ? act[k] : detail::GeomPick{}, s.x, s.y);
    }

    float* grid = out + kPlayerDimV6 + kGeomDim;
    std::memset(grid, 0, sizeof(float) * 2 * kGridDim);
    detail::rasterizeAt(g.solids, s.x + kGridXOffV6, s.y + kGridYOffV6, grid);
    detail::rasterizeAt(g.hazards, s.x + kGridXOffV6, s.y + kGridYOffV6,
                        grid + kGridDim);
}

// Dispatch on the layout version (gdenv_create's obs_version).
inline void buildObsVersioned(int version, const gdcore::PlayerState& s,
                              const gdcore::LevelGeometry& g,
                              const gdcore::PhysicsConfig& c, float levelEndX,
                              float* out) {
    if (version == kObsVersionV6) buildObsV6(s, g, c, levelEndX, out);
    else buildObs(s, g, levelEndX, c.dx, out);
}

} // namespace gdenv
