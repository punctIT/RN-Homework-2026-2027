
import pathlib
import math



def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    A = list()
    B = list()
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip();
            line = line.replace("=", " = ").replace("+ ", " +").replace("- ", " -")
            data = line.strip().split()
            coeffs = []
            for term in data:
                if "x" in term or "y" in term or "z" in term:
                    num = term[:-1]
                    if num in ("", "+"):
                        coeffs.append(1.0)
                    elif num == "-":
                        coeffs.append(-1.0)
                    else:
                        coeffs.append(float(num))
                elif term == "=":
                    continue
                else:
                    try:
                        B.append(float(term))
                    except ValueError:
                        pass
            A.append(coeffs)
            #print(data)
    return [A, B]


def determinant(matrix: list[list[float]]) -> float:
    return  matrix[0][0] * (matrix[1][1]*matrix[2][2] - matrix[1][2]*matrix[2][1])- matrix[0][1] * (matrix[1][0]*matrix[2][2] - matrix[1][2]*matrix[2][0])+ matrix[0][2] * (matrix[1][0]*matrix[2][1] - matrix[1][1]*matrix[2][0])
    

def trace(matrix: list[list[float]]) -> float:
    return sum(matrix[i][i] for i in range(0, len(matrix)))

def norm(vector: list[float]) -> float:
    return math.sqrt(sum(v**2 for v in vector))

def transpose(matrix: list[list[float]]) -> list[list[float]]:
    rows = len(matrix)
    cols = len(matrix[0])
    result = []
    for j in range(cols):
        new_row = []
        for i in range(rows):
            new_row.append(matrix[i][j])
        result.append(new_row)
    return result


def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    result = []
    for i in range(len(matrix)):
        total = 0.0
        for j in range(len(vector)):
            total += matrix[i][j] * vector[j]
        result.append(total)
    return result

def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det = determinant(matrix)
    if det == 0:
        raise ValueError("Sistemul nu are solutie unica (determinant = 0)")

    n = len(matrix)
    result = []
    for k in range(n):
        modified = []
        for i in range(n):
            row = []
            for j in range(n):
                if j == k:
                    row.append(vector[i])
                else:
                    row.append(matrix[i][j])
            modified.append(row)
        result.append(determinant(modified) / det)
    return result


def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    result = []
    for r in range(len(matrix)):
        if r == i:
            continue
        row = []
        for c in range(len(matrix[0])):
            if c == j:
                continue
            row.append(matrix[r][c])
        result.append(row)
    return result


def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            m = minor(matrix, i, j)
            det_minor = m[0][0] * m[1][1] - m[0][1] * m[1][0]
            sign = (-1) ** (i + j)
            row.append(sign * det_minor)
        result.append(row)
    return result


def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det = determinant(matrix)
    if det == 0:
        raise ValueError("Matricea nu este inversabila")
    adj = adjoint(matrix)
    n = len(matrix)
    result = []
    for i in range(n):
        total = 0.0
        for j in range(n):
            total += (adj[i][j] / det) * vector[j]
        result.append(total)
    return result


if __name__ == "__main__":
    A, B = load_system(pathlib.Path("system.txt"))
    print(f"{A=} {B=}")
    print(f"{determinant(A)=}")
    print(f"{trace(A)=}")
    print(f"{norm(B)=}")
    print(f"{transpose(A)=}")
    print(f"{multiply(A, B)=}")
    print(f"{solve_cramer(A, B)=}")
    print(f"{solve(A, B)=}")