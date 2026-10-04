"""이산수학 Report 1
실행: python3 inverse_matrix.py [--trace]
외부 패키지 없이 행렬식과 가우스–조던으로 역행렬을 계산한다.
"""
from fractions import Fraction
import argparse


class SingularMatrixError(ValueError):
    """역행렬이 없는 정방행렬."""


def read_matrix():
    """양의 정수 차수와 행별 원소를 받아 2차원 리스트로 저장."""
    while True:
        try:
            n = int(input("정방행렬의 차수 n: "))
            if n <= 0:
                raise ValueError
            break
        except ValueError:
            print("입력 오류: n은 양의 정수여야 합니다.")
    matrix = []
    for i in range(n):
        while True:
            try:
                values = input(f"{i + 1}행 ({n}개 원소): ").split()
                if len(values) != n:
                    raise ValueError(f"원소를 {n}개 입력하세요.")
                row = [Fraction(value) for value in values]
                matrix.append(row)
                break
            except (ValueError, ZeroDivisionError) as error:
                print(f"입력 오류: 정수·소수·분수로 입력하세요. ({error})")
    return matrix


def minor(matrix, row, col):
    """지정한 행과 열을 제거한 소행렬."""
    return [[value for j, value in enumerate(values) if j != col]
            for i, values in enumerate(matrix) if i != row]


def determinant(matrix):
    """첫째 행의 여인수 전개로 행렬식을 재귀 계산."""
    n = len(matrix)
    if n == 0:
        return Fraction(1)  # 1×1 행렬의 여인수에 필요한 빈 행렬식
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return sum(((-1) ** j * matrix[0][j]
                * determinant(minor(matrix, 0, j)) for j in range(n)), Fraction(0))


def inverse_by_determinant(matrix, trace=False):
    """A^-1 = 여인수행렬의 전치 / det(A)."""
    det = determinant(matrix)
    if trace:
        print(f"det(A) = {det}")
    if det == 0:
        raise SingularMatrixError("행렬식이 0이므로 역행렬이 존재하지 않습니다.")
    n = len(matrix)
    cofactors = [[(-1) ** (i + j) * determinant(minor(matrix, i, j))
                  for j in range(n)] for i in range(n)]
    if trace:
        print_matrix("여인수 행렬", cofactors)
        print_matrix("수반행렬 (여인수 행렬의 전치)",
                     [[cofactors[j][i] for j in range(n)] for i in range(n)])
    return [[cofactors[j][i] / det for j in range(n)] for i in range(n)]


def identity(n):
    return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]


def inverse_by_gauss_jordan(matrix, trace=False):
    """확대행렬 [A | I]의 왼쪽을 I로 바꾸어 오른쪽을 반환."""
    n = len(matrix)
    unit = identity(n)
    augmented = [matrix[i][:] + unit[i] for i in range(n)]
    if trace:
        print_matrix("초기 확대행렬 [A | I]", augmented, split=n)
    for col in range(n):
        pivot_row = next((r for r in range(col, n) if augmented[r][col] != 0), None)
        if pivot_row is None:
            raise SingularMatrixError(f"{col + 1}열에 피벗이 없어 역행렬이 존재하지 않습니다.")
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
            if trace:
                print_matrix(f"행 교환 R{col + 1} <-> R{pivot_row + 1}", augmented, n)
        pivot = augmented[col][col]
        augmented[col] = [value / pivot for value in augmented[col]]
        if trace:
            print_matrix(f"피벗 정규화 R{col + 1} / ({pivot})", augmented, n)
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0:
                continue
            augmented[row] = [augmented[row][j] - factor * augmented[col][j]
                              for j in range(2 * n)]
            if trace:
                print_matrix(f"소거 R{row + 1} - ({factor}) * R{col + 1}", augmented, n)
    return [row[n:] for row in augmented]


def multiply(left, right):
    n = len(left)
    return [[sum((left[i][k] * right[k][j] for k in range(n)), Fraction(0))
             for j in range(n)] for i in range(n)]


def print_matrix(title, matrix, split=None):
    print(title + ":")
    cells = [[str(value) for value in row] for row in matrix]
    width = max(len(value) for row in cells for value in row)
    for row in cells:
        formatted = [value.rjust(width) for value in row]
        if split is not None:
            formatted.insert(split, "|")
        print("  [ " + "  ".join(formatted) + " ]")


def main():
    parser = argparse.ArgumentParser(description="두 방법으로 역행렬 계산 및 비교")
    parser.add_argument("--trace", action="store_true", help="여인수와 행 연산 과정을 출력")
    args = parser.parse_args()
    try:
        matrix = read_matrix()
        print()
        print_matrix("입력 행렬 A (2차원 리스트)", matrix)
        results = []
        for name, method in [("행렬식", inverse_by_determinant),
                             ("가우스–조던", inverse_by_gauss_jordan)]:
            print(f"\n[{name} 방법]")
            try:
                inverse = method(matrix, args.trace)
                print_matrix(f"{name}으로 구한 역행렬", inverse)
                results.append(inverse)
            except SingularMatrixError as error:
                print(f"오류: {error}")
                results.append(None)
        first, second = results
        if first is not None and second is not None:
            print("\n두 방법의 결과가 동일합니다." if first == second
                  else "\n두 방법의 결과가 다릅니다.")
            unit = identity(len(matrix))
            for name, inverse in zip(["행렬식", "가우스–조던"], results):
                right_product = multiply(matrix, inverse)
                left_product = multiply(inverse, matrix)
                print_matrix(f"{name} 검증 A * A^-1", right_product)
                valid = right_product == unit and left_product == unit
                print(f"{name} 양방향 단위행렬 검증: {'통과' if valid else '실패'}")
        elif first is None and second is None:
            print("\n두 방법 모두 역행렬이 없다고 판단했습니다. 결과 비교는 생략합니다.")
        else:
            print("\n주의: 두 방법의 역행렬 존재 여부 판단이 다릅니다.")
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 종료되어 프로그램을 마칩니다.")


if __name__ == "__main__":
    main()
