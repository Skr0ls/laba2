from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias


Vector: TypeAlias = tuple[float, ...]
Matrix: TypeAlias = tuple[tuple[float, ...], ...]

ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

EPSILON = 0.01
MAX_ITERATIONS = 100
INITIAL_APPROXIMATION: Vector = (0.0, 0.0, 0.0)

A: Matrix = (
    (5.0, 0.0, 1.0),
    (1.0, 3.0, -1.0),
    (-3.0, 2.0, 10.0),
)
B: Vector = (11.0, 4.0, 6.0)


@dataclass(frozen=True)
class DominanceRow:
    row_number: int
    diagonal_abs: float
    off_diagonal_sum: float
    is_strictly_dominant: bool


@dataclass(frozen=True)
class IterationRow:
    n: int
    x: Vector
    max_diff: float
    error_estimate: float
    residual: Vector
    residual_norm: float


@dataclass(frozen=True)
class SolverResult:
    iteration_matrix: Matrix
    free_terms: Vector
    contraction_norm: float
    iterations: list[IterationRow]
    solution: Vector
    rounded_solution: Vector


def matrix_vector_product(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(sum(value * vector[index] for index, value in enumerate(row)) for row in matrix)


def vector_add(left: Vector, right: Vector) -> Vector:
    return tuple(left[index] + right[index] for index in range(len(left)))


def vector_subtract(left: Vector, right: Vector) -> Vector:
    return tuple(left[index] - right[index] for index in range(len(left)))


def max_abs(vector: Vector) -> float:
    return max(abs(value) for value in vector)


def format_number(value: float, digits: int = 6) -> str:
    rounded = round(value, digits)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{digits}f}"


def format_vector(vector: Vector, digits: int = 6) -> str:
    return "[" + ", ".join(format_number(value, digits) for value in vector) + "]"


def format_matrix(matrix: Matrix, digits: int = 6) -> str:
    rows = []
    for row in matrix:
        rows.append("  [" + ", ".join(f"{format_number(value, digits):>10}" for value in row) + "]")
    return "\n".join(rows)


def validate_system(matrix: Matrix, vector: Vector) -> None:
    if len(matrix) == 0:
        raise ValueError("The coefficient matrix must not be empty.")
    if len(matrix) != len(vector):
        raise ValueError("The coefficient matrix and free-term vector dimensions do not match.")
    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError("The coefficient matrix must be square.")


def check_diagonal_dominance(matrix: Matrix) -> list[DominanceRow]:
    rows: list[DominanceRow] = []
    for row_index, row in enumerate(matrix):
        diagonal_abs = abs(row[row_index])
        off_diagonal_sum = sum(abs(value) for index, value in enumerate(row) if index != row_index)
        rows.append(
            DominanceRow(
                row_number=row_index + 1,
                diagonal_abs=diagonal_abs,
                off_diagonal_sum=off_diagonal_sum,
                is_strictly_dominant=diagonal_abs > off_diagonal_sum,
            )
        )
    return rows


def build_iteration_system(matrix: Matrix, vector: Vector) -> tuple[Matrix, Vector]:
    validate_system(matrix, vector)

    iteration_rows: list[tuple[float, ...]] = []
    free_terms: list[float] = []

    for row_index, row in enumerate(matrix):
        diagonal = row[row_index]
        if abs(diagonal) < 1e-12:
            raise ZeroDivisionError(f"Zero diagonal coefficient in row {row_index + 1}.")

        iteration_row = []
        for column_index, value in enumerate(row):
            if row_index == column_index:
                iteration_row.append(0.0)
            else:
                iteration_row.append(-value / diagonal)

        iteration_rows.append(tuple(iteration_row))
        free_terms.append(vector[row_index] / diagonal)

    return tuple(iteration_rows), tuple(free_terms)


def matrix_infinity_norm(matrix: Matrix) -> float:
    return max(sum(abs(value) for value in row) for row in matrix)


def solve_by_simple_iteration(
    matrix: Matrix,
    vector: Vector,
    initial: Vector,
    epsilon: float,
    max_iterations: int = MAX_ITERATIONS,
) -> SolverResult:
    iteration_matrix, free_terms = build_iteration_system(matrix, vector)
    contraction_norm = matrix_infinity_norm(iteration_matrix)
    if contraction_norm >= 1:
        raise ValueError("The iteration matrix is not contractive in the infinity norm.")

    iterations: list[IterationRow] = []
    previous = initial
    multiplier = contraction_norm / (1.0 - contraction_norm)

    for n in range(1, max_iterations + 1):
        current = vector_add(matrix_vector_product(iteration_matrix, previous), free_terms)
        max_diff = max_abs(vector_subtract(current, previous))
        error_estimate = multiplier * max_diff
        residual = vector_subtract(matrix_vector_product(matrix, current), vector)
        residual_norm = max_abs(residual)

        iterations.append(
            IterationRow(
                n=n,
                x=current,
                max_diff=max_diff,
                error_estimate=error_estimate,
                residual=residual,
                residual_norm=residual_norm,
            )
        )

        if error_estimate <= epsilon:
            rounded_solution = tuple(round(value, 2) for value in current)
            return SolverResult(
                iteration_matrix=iteration_matrix,
                free_terms=free_terms,
                contraction_norm=contraction_norm,
                iterations=iterations,
                solution=current,
                rounded_solution=rounded_solution,
            )

        previous = current

    raise RuntimeError("Simple iteration did not converge within the iteration limit.")


def format_dominance_report(rows: list[DominanceRow]) -> list[str]:
    lines = ["Проверка диагонального преобладания:"]
    for row in rows:
        sign = ">" if row.is_strictly_dominant else "<="
        status = "да" if row.is_strictly_dominant else "нет"
        lines.append(
            f"  строка {row.row_number}: |a{row.row_number}{row.row_number}| = "
            f"{format_number(row.diagonal_abs)} {sign} "
            f"{format_number(row.off_diagonal_sum)} -> {status}"
        )
    return lines


def format_iteration_table(iterations: list[IterationRow]) -> list[str]:
    lines = [
        "Таблица итераций:",
        " n |         x1 |         x2 |         x3 |   max diff |    err est |  max |AX-B|",
        "---+------------+------------+------------+------------+------------+------------",
    ]
    for row in iterations:
        x1, x2, x3 = row.x
        lines.append(
            f"{row.n:2d} | {x1:10.6f} | {x2:10.6f} | {x3:10.6f} | "
            f"{row.max_diff:10.6f} | {row.error_estimate:10.6f} | {row.residual_norm:10.6f}"
        )
    return lines


def format_console_report(result: SolverResult, dominance_rows: list[DominanceRow]) -> str:
    final_iteration = result.iterations[-1]
    lines = [
        "Лабораторная работа 2: решение СЛАУ",
        "Вариант: 1",
        "Метод: метод простой итерации",
        f"Точность: epsilon = {EPSILON}",
        f"Начальное приближение: X0 = {format_vector(INITIAL_APPROXIMATION)}",
        "",
        "Система: A * X = B",
        "Матрица A:",
        format_matrix(A),
        "Вектор B:",
        "  " + format_vector(B),
        "",
        *format_dominance_report(dominance_rows),
        "",
        "Итерационный вид: X(n+1) = C * X(n) + d",
        "Матрица C:",
        format_matrix(result.iteration_matrix),
        "Вектор d:",
        "  " + format_vector(result.free_terms),
        f"Норма ||C||_inf = q = {format_number(result.contraction_norm)}",
        f"Критерий остановки: q / (1 - q) * max diff <= {format_number(EPSILON)}",
        "",
        *format_iteration_table(result.iterations),
        "",
        "Финальный ответ:",
        f"  X = {format_vector(result.solution)}",
        f"  После округления до 0.01: X = {format_vector(result.rounded_solution, 2)}",
        f"  Невязка AX - B = {format_vector(final_iteration.residual)}",
        f"  ||AX - B||_inf = {format_number(final_iteration.residual_norm)}",
    ]
    return "\n".join(lines)


def write_markdown_table(result: SolverResult) -> None:
    lines = [
        "# Таблица итераций",
        "",
        "| n | x1 | x2 | x3 | max diff | error estimate | max residual |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.iterations:
        x1, x2, x3 = row.x
        lines.append(
            f"| {row.n} | {x1:.6f} | {x2:.6f} | {x3:.6f} | "
            f"{row.max_diff:.6f} | {row.error_estimate:.6f} | {row.residual_norm:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Final solution: `{format_vector(result.solution)}`.",
            f"Rounded solution: `{format_vector(result.rounded_solution, 2)}`.",
        ]
    )
    (ARTIFACTS_DIR / "iteration_table.md").write_text("\n".join(lines), encoding="utf-8")


def write_latex_table(result: SolverResult) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Таблица итераций метода простой итерации}",
        "\\small",
        "\\begin{tabular}{rrrrrr}",
        "\\toprule",
        "$n$ & $x_1$ & $x_2$ & $x_3$ & $\\Delta_n$ & $R_n$ \\\\",
        "\\midrule",
    ]
    for row in result.iterations:
        x1, x2, x3 = row.x
        lines.append(
            f"{row.n} & {x1:.6f} & {x2:.6f} & {x3:.6f} & "
            f"{row.max_diff:.6f} & {row.error_estimate:.6f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\normalsize",
            "\\end{table}",
            "",
        ]
    )
    (ARTIFACTS_DIR / "iteration_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_results_json(result: SolverResult, dominance_rows: list[DominanceRow]) -> None:
    data = {
        "variant": 1,
        "method": "simple_iteration",
        "epsilon": EPSILON,
        "initial_approximation": INITIAL_APPROXIMATION,
        "matrix_a": A,
        "vector_b": B,
        "diagonal_dominance": [asdict(row) for row in dominance_rows],
        "result": asdict(result),
    }
    (ARTIFACTS_DIR / "results.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def find_monospace_font() -> str | None:
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/Library/Fonts/Courier New.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def make_console_screenshot(console_text: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font_path = find_monospace_font()
    font = ImageFont.truetype(font_path, 17) if font_path else ImageFont.load_default()
    title_font = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()
    lines = console_text.splitlines()
    padding = 28
    line_gap = 6
    title = "python3 main.py"

    probe = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe)
    text_width = max(int(draw.textlength(line, font=font)) for line in lines)
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + line_gap
    title_height = title_font.getbbox("Ag")[3] - title_font.getbbox("Ag")[1] + 18
    width = text_width + padding * 2
    height = padding * 2 + title_height + line_height * len(lines)

    image = Image.new("RGB", (width, height), "#111827")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (14, 14, width - 14, height - 14),
        radius=12,
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


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    dominance_rows = check_diagonal_dominance(A)
    if not all(row.is_strictly_dominant for row in dominance_rows):
        raise ValueError("The matrix does not have strict row diagonal dominance.")

    result = solve_by_simple_iteration(A, B, INITIAL_APPROXIMATION, EPSILON)
    console_report = format_console_report(result, dominance_rows)

    print(console_report)
    (ARTIFACTS_DIR / "console_output.txt").write_text(console_report, encoding="utf-8")
    write_markdown_table(result)
    write_latex_table(result)
    write_results_json(result, dominance_rows)
    make_console_screenshot(console_report)


if __name__ == "__main__":
    main()
