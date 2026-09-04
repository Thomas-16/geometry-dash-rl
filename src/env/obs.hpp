#pragma once

#include "core/geometry.hpp"
#include "core/physics.hpp"

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
// emit any of these. Scalars 0-23 and the grid layout are frozen —
// additions bump kObsVersion and only ever append.
// gdenv_obs_version() exposes this so the Python side can refuse a stale
// DLL.
constexpr int kObsVersion = 4;
constexpr int kPlayerDim = 26;
constexpr int kObsDim = kPlayerDim + 2 * kGridDim;

namespace detail {

// Accumulate one sorted-by-left box list into a [row][col] cell block.
inline void rasterize(const std::vector<gdcore::Box>& boxes, float px,
                      float* cells) {
    // Boxes are at most 34 wide (portal width bounds the table), same slack
    // as gdcore::detail::forCandidates.
    auto it = std::lower_bound(
        boxes.begin(), boxes.end(), px - 34.0f,
        [](const gdcore::Box& b, float v) { return b.left < v; });
    const float gx1 = px + kLookahead;
    const float gy1 = kGridYBase + kGridRows * kCellH;
    for (; it != boxes.end() && it->left < gx1; ++it) {
        float l = std::max(it->left, px);
        float r = std::min(it->right, gx1);
        float b = std::max(it->bottom, kGridYBase);
        float t = std::min(it->top, gy1);
        if (l >= r || b >= t) continue;
        int c0 = static_cast<int>((l - px) / kCellW);
        int c1 = std::min(kGridCols,
                          static_cast<int>(std::ceil((r - px) / kCellW)));
        int r0 = static_cast<int>((b - kGridYBase) / kCellH);
        int r1 = std::min(kGridRows,
                          static_cast<int>(std::ceil((t - kGridYBase) / kCellH)));
        for (int row = r0; row < r1; ++row) {
            float cy0 = kGridYBase + row * kCellH;
            float oh = std::min(t, cy0 + kCellH) - std::max(b, cy0);
            for (int col = c0; col < c1; ++col) {
                float cx0 = px + col * kCellW;
                float ow = std::min(r, cx0 + kCellW) - std::max(l, cx0);
                float& cell = cells[row * kGridCols + col];
                cell = std::min(1.0f, cell + (ow * oh) / (kCellW * kCellH));
            }
        }
    }
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
    out[3] = s.ship ? 1.0f : s.ball ? -1.0f : 0.0f;
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

    float* grid = out + kPlayerDim;
    std::memset(grid, 0, sizeof(float) * 2 * kGridDim);
    detail::rasterize(g.solids, s.x, grid);
    detail::rasterize(g.hazards, s.x, grid + kGridDim);
}

} // namespace gdenv
