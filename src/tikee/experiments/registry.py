"""Definición declarativa de los brazos (ARCHITECTURE.md §8.1).

QAOA (C3) queda fuera del registro multi-semilla: corre una sola vez (semilla 42,
F4) por su costo (~25 min por profundidad p); no se replica x10 semillas — el propio
presupuesto de cómputo de PLAN.md §13 ya lo trata como costo fijo, no por semilla."""

from __future__ import annotations

LEVEL_A_ARMS = ["A0", "A1", "B0", "B0x", "B1", "C0", "C1", "C2", "R"]
LEVEL_B_ARMS = ["A0b", "A1b", "B0b", "B1b", "C0b", "C2b", "C4b", "Rb"]
ALL_MULTISEED_ARMS = LEVEL_A_ARMS + LEVEL_B_ARMS

QUBO_ARMS_LEVEL_A = ["C0", "C1", "C2"]
QUBO_ARMS_LEVEL_B = ["C0b", "C2b", "C4b"]
