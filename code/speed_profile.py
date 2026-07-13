"""
speed_profile.py
-----------------
Single source of truth for the time-of-day travel-speed table used across
the standard heuristic, the learnheuristic and the dynamic heuristic.

Speed profile from Rudy (2025), Table 5 — Beijing in-day traffic averages,
index = hour of day (0 = midnight).
"""

from __future__ import annotations

SPEED_PROFILE_KMH: list[float] = [
    38.9,  # 00h (12AM)
    39.5,  # 01h
    40.2,  # 02h
    40.9,  # 03h
    41.0,  # 04h  <- peak
    40.0,  # 05h
    35.6,  # 06h
    30.9,  # 07h
    30.2,  # 08h
    30.8,  # 09h
    31.1,  # 10h
    31.7,  # 11h
    32.4,  # 12h (12PM)
    32.1,  # 13h
    31.2,  # 14h
    30.9,  # 15h
    30.2,  # 16h
    28.4,  # 17h  <- evening rush
    28.4,  # 18h  <- evening rush
    31.1,  # 19h
    32.5,  # 20h
    33.6,  # 21h
    37.0,  # 22h
    38.0,  # 23h
]

V_MAX_MS: float = max(SPEED_PROFILE_KMH) / 3.6   # 41.0 km/h in m/s
AVG_SPEED_MS: float = (sum(SPEED_PROFILE_KMH) / 24.0) / 3.6   # ~33.8 km/h in m/s


def speed_ms(time_s: float) -> float:
    """Travel speed [m/s] for the hour corresponding to time_s (seconds from midnight)."""
    hour_idx = int(time_s / 3600.0) % 24
    return SPEED_PROFILE_KMH[hour_idx] / 3.6
