BLUE = "#1559A6"
RED = "#C7262E"
GREEN = "#2B8A3E"
ORANGE = "#B7472A"
DARK = "#243447"
GRID = "rgba(70, 86, 103, 0.18)"
STABLE_FILL = "rgba(43, 138, 62, 0.06)"


def frequency_label(unit: str) -> str:
    return "Frequência (Hz)" if unit == "Hz" else "Frequência angular (rad/s)"


def frequency_values(omega: tuple[float, ...], unit: str) -> list[float]:
    if unit == "Hz":
        from math import pi

        return [value / (2.0 * pi) for value in omega]
    return list(omega)


def convert_frequency(omega: float | None, unit: str) -> float | None:
    if omega is None:
        return None
    if unit == "Hz":
        from math import pi

        return omega / (2.0 * pi)
    return omega
