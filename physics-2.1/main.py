from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

C1 = 1.0  # uF
C2 = 1.0  # uF
C3 = 1.0  # uF
C4 = 0.5  # uF
C5 = 1.0  # uF

U0 = 200.0  # V


@dataclass(frozen=True)
class CapacitorBatteryState:
    c_equiv_uf: float
    q_total_uc: float
    voltage_v: float
    energy_j: float


@dataclass(frozen=True)
class NodePotentialSolution:
    va_v: float
    vb_v: float
    q_left_uc: float
    c_equiv_uf: float


@dataclass(frozen=True)
class PhysicsTaskResult:
    initial: CapacitorBatteryState
    node_solution: NodePotentialSolution
    final: CapacitorBatteryState
    delta_energy_j: float


def capacitors_series(left_uf: float, right_uf: float) -> float:
    return left_uf * right_uf / (left_uf + right_uf)


def energy_from_capacitance_and_voltage(c_uf: float, u_v: float) -> float:
    return 0.5 * c_uf * 1e-6 * u_v**2


def energy_from_charge_and_capacitance(q_uc: float, c_uf: float) -> float:
    return 0.5 * q_uc**2 / (c_uf * 1e-6) * 1e-12


def compute_initial_state() -> CapacitorBatteryState:
    c12 = capacitors_series(C1, C2)
    c34 = capacitors_series(C3, C4)
    c_initial = c12 + c34
    q_initial = c_initial * U0
    e_initial = energy_from_capacitance_and_voltage(c_initial, U0)
    return CapacitorBatteryState(
        c_equiv_uf=c_initial,
        q_total_uc=q_initial,
        voltage_v=U0,
        energy_j=e_initial,
    )


def solve_node_potentials() -> NodePotentialSolution:
    vm = 1.0
    vn = 0.0

    a11 = C1 + C2 + C5
    a12 = -C5
    b1 = C1 * vm + C2 * vn
    a21 = -C5
    a22 = C3 + C4 + C5
    b2 = C3 * vm + C4 * vn

    det = a11 * a22 - a12 * a21
    va = (b1 * a22 - a12 * b2) / det
    vb = (a11 * b2 - b1 * a21) / det

    q_left = C1 * (vm - va) + C3 * (vm - vb)
    c_final = q_left / vm

    return NodePotentialSolution(
        va_v=va,
        vb_v=vb,
        q_left_uc=q_left,
        c_equiv_uf=c_final,
    )


def solve_final_state(q_initial_uc: float, c_final_uf: float) -> CapacitorBatteryState:
    u_final = q_initial_uc / c_final_uf
    e_final = energy_from_charge_and_capacitance(q_initial_uc, c_final_uf)

    return CapacitorBatteryState(
        c_equiv_uf=c_final_uf,
        q_total_uc=q_initial_uc,
        voltage_v=u_final,
        energy_j=e_final,
    )


def format_report(
    result: PhysicsTaskResult,
) -> str:
    initial = result.initial
    node_solution = result.node_solution
    final = result.final
    delta_e_j = result.delta_energy_j

    lines = [
        "Physics task 2.1: capacitor battery",
        "",
        "Input data:",
        f"  C1 = C2 = C3 = C5 = {C1} uF",
        f"  C4 = {C4} uF",
        f"  U0 = {U0} V",
        "",
        "--- Open switch ---",
        f"  C12 = C1 * C2 / (C1 + C2) = {capacitors_series(C1, C2):.6f} uF",
        f"  C34 = C3 * C4 / (C3 + C4) = {capacitors_series(C3, C4):.6f} uF",
        f"  C_initial = C12 + C34 = {initial.c_equiv_uf:.6f} uF",
        f"  Q_initial = C_initial * U0 = {initial.q_total_uc:.6f} uC",
        f"  E_initial = C_initial * U0^2 / 2 = {initial.energy_j:.6f} J",
        "",
        "--- Closed switch: node potentials ---",
        "  3.0 * Va - 1.0 * Vb = 1.0",
        " -1.0 * Va + 2.5 * Vb = 1.0",
        f"  Va = {node_solution.va_v:.6f} V",
        f"  Vb = {node_solution.vb_v:.6f} V",
        f"  Q_left = {node_solution.q_left_uc:.6f} uC at 1 V",
        f"  C_final = {final.c_equiv_uf:.6f} uF",
        "",
        "--- Charge conservation after disconnecting the source ---",
        f"  Q_final = Q_initial = {final.q_total_uc:.6f} uC",
        f"  U_final = Q_initial / C_final = {final.voltage_v:.6f} V",
        "",
        "--- Energy ---",
        f"  E_final = Q^2 / (2 * C_final) = {final.energy_j:.6f} J",
        f"  Delta E = E_final - E_initial = {delta_e_j:.6f} J",
        "",
        f"Energy decreases by approximately {abs(delta_e_j) * 1000:.3f} mJ.",
    ]
    return "\n".join(lines)


def write_artifacts(result: PhysicsTaskResult, console_text: str) -> None:
    (ARTIFACTS_DIR / "console_output.txt").write_text(console_text + "\n", encoding="utf-8")
    (ARTIFACTS_DIR / "results.json").write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def make_screenshot(console_text: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/Library/Fonts/Courier New.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ]
    font_path = None
    for candidate in candidates:
        if Path(candidate).exists():
            font_path = candidate
            break

    font = ImageFont.truetype(font_path, 16) if font_path else ImageFont.load_default()
    title_font = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default()
    lines = console_text.splitlines()
    padding = 24
    line_gap = 5
    title = "python3 physics-2.1/main.py"

    probe = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe)
    text_width = max(int(draw.textlength(line, font=font)) for line in lines)
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + line_gap
    title_height = title_font.getbbox("Ag")[3] - title_font.getbbox("Ag")[1] + 14
    width = text_width + padding * 2
    height = padding * 2 + title_height + line_height * len(lines)

    image = Image.new("RGB", (width, height), "#111827")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (12, 12, width - 12, height - 12),
        radius=10,
        outline="#374151",
        width=2,
        fill="#111827",
    )
    draw.text((padding, padding), title, fill="#93c5fd", font=title_font)

    y = padding + title_height
    for line in lines:
        draw.text((padding, y), line, fill="#e5e7eb", font=font)
        y += line_height

    image.save(ASSETS_DIR / "program_output.png")


def make_scheme() -> None:
    from PIL import Image, ImageDraw, ImageFont

    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    font_path = next((candidate for candidate in candidates if Path(candidate).exists()), None)
    font = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()
    small_font = ImageFont.truetype(font_path, 16) if font_path else ImageFont.load_default()

    width, height = 900, 420
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)

    black = "#111827"
    blue = "#2563eb"
    gray = "#6b7280"

    left_x = 95
    right_x = 805
    top_y = 115
    bottom_y = 305
    middle_y = (top_y + bottom_y) // 2
    a_x = 450
    b_x = 450
    cap_gap = 20
    plate_half = 42

    def line(points, fill=black, width_value=4):
        draw.line(points, fill=fill, width=width_value)

    def capacitor(x: int, y: int, label: str, value: str) -> None:
        line([(x - 70, y), (x - cap_gap, y)])
        line([(x + cap_gap, y), (x + 70, y)])
        line([(x - cap_gap, y - plate_half), (x - cap_gap, y + plate_half)], blue, 5)
        line([(x + cap_gap, y - plate_half), (x + cap_gap, y + plate_half)], blue, 5)
        draw.text((x - 22, y - 86), label, fill=black, font=font)
        draw.text((x - 30, y + 54), value, fill=gray, font=small_font)

    line([(left_x, top_y), (left_x, bottom_y)])
    line([(right_x, top_y), (right_x, bottom_y)])

    line([(left_x, top_y), (170, top_y)])
    capacitor(240, top_y, "C1", "1 uF")
    line([(310, top_y), (a_x, top_y)])
    line([(a_x, top_y), (550, top_y)])
    capacitor(620, top_y, "C2", "1 uF")
    line([(690, top_y), (right_x, top_y)])

    line([(left_x, bottom_y), (170, bottom_y)])
    capacitor(240, bottom_y, "C3", "1 uF")
    line([(310, bottom_y), (b_x, bottom_y)])
    line([(b_x, bottom_y), (550, bottom_y)])
    capacitor(620, bottom_y, "C4", "0.5 uF")
    line([(690, bottom_y), (right_x, bottom_y)])

    line([(a_x, top_y), (a_x, 165)])
    line([(a_x - 42, 165), (a_x + 42, 165)], blue, 5)
    line([(a_x - 42, 215), (a_x + 42, 215)], blue, 5)
    line([(a_x, 215), (a_x, 248)])
    line([(a_x, 280), (a_x, bottom_y)])
    line([(a_x, 248), (a_x - 34, 276)], gray, 3)
    draw.text((a_x + 55, 180), "C5", fill=black, font=font)
    draw.text((a_x + 55, 207), "1 uF", fill=gray, font=small_font)
    draw.text((a_x - 72, 253), "K", fill=black, font=font)

    draw.ellipse((left_x - 8, middle_y - 8, left_x + 8, middle_y + 8), fill=black)
    draw.ellipse((right_x - 8, middle_y - 8, right_x + 8, middle_y + 8), fill=black)
    draw.ellipse((a_x - 7, top_y - 7, a_x + 7, top_y + 7), fill=black)
    draw.ellipse((b_x - 7, bottom_y - 7, b_x + 7, bottom_y + 7), fill=black)
    draw.text((a_x + 14, top_y - 34), "a", fill=black, font=font)
    draw.text((b_x + 14, bottom_y + 8), "b", fill=black, font=font)
    draw.text((left_x - 52, middle_y - 13), "m", fill=black, font=font)
    draw.text((right_x + 24, middle_y - 13), "n", fill=black, font=font)
    draw.text((58, 54), "U0 = 200 V", fill=black, font=font)

    image.save(ASSETS_DIR / "scheme.png")


def main() -> None:
    initial = compute_initial_state()
    node_solution = solve_node_potentials()
    final = solve_final_state(initial.q_total_uc, node_solution.c_equiv_uf)
    delta_e_j = final.energy_j - initial.energy_j
    result = PhysicsTaskResult(
        initial=initial,
        node_solution=node_solution,
        final=final,
        delta_energy_j=delta_e_j,
    )
    report = format_report(result)
    print(report)
    write_artifacts(result, report)
    make_screenshot(report)
    make_scheme()


if __name__ == "__main__":
    main()
