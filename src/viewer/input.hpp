// Input wrappers gated on the ImGui panel: when a text field has focus or the
// mouse is over the panel, the game modes must not see the key/click (typing
// a run name must not make the cube jump). Every mode input site goes through
// these instead of raw raylib calls.
#pragma once

#include "viewer/world.hpp"

#include "raylib.h"

namespace gdv {

inline bool keyPressed(const App& a, int key) { return !a.guiWantKeyboard && IsKeyPressed(key); }
inline bool keyDown(const App& a, int key) { return !a.guiWantKeyboard && IsKeyDown(key); }
inline bool mousePressed(const App& a, int b) { return !a.guiWantMouse && IsMouseButtonPressed(b); }
inline bool mouseDown(const App& a, int b) { return !a.guiWantMouse && IsMouseButtonDown(b); }
inline float mouseWheel(const App& a) { return a.guiWantMouse ? 0.0f : GetMouseWheelMove(); }

}  // namespace gdv
