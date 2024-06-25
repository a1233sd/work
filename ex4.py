import numpy as np
from scipy.linalg import solve

def solve_circuit(R, V_source):
    # Разбиваем R на отдельные переменные для удобства
    R1, R2, R3, R4, R5, R6, R7, R8, R9, R10 = R
    
    # Матрица коэффициентов (с учетом R)
    A = np.array([
        [1/R1 + 1/R2 + 1/R3, -1/R2, 0, -1/R3, 0],
        [-1/R2, 1/R2 + 1/R4 + 1/R5 + 1/R6, -1/R5, -1/R6, 0],
        [0, -1/R5, 1/R5 + 1/R7 + 1/R8, 0, -1/R8],
        [-1/R3, -1/R6, 0, 1/R3 + 1/R6 + 1/R9, -1/R9],
        [0, 0, -1/R8, -1/R9, 1/R8 + 1/R9 + 1/R10]
    ])
    
    # Вектор правых частей (с учетом источника напряжения)
    B = np.array([V_source / R1, 0, 0, 0, 0])
    
    # Решаем систему линейных уравнений
    V = solve(A, B)
    
    return V


if __name__ == "__main__":
    # Ввод значений сопротивлений (в омах)
    R = []
    for i in range(1, 11):
        R_value = float(input(f"Введите значение сопротивления R{i} (в омах): "))
        R.append(R_value)
    
    # Ввод значения источника напряжения (в вольтах)
    V_source = float(input("Введите значение источника напряжения (в вольтах): "))
    
    # Решаем задачу
    voltages = solve_circuit(R, V_source)
    
    # Выводим результаты
    for i, V in enumerate(voltages, start=1):
        print(f"Напряжение в узле V{i}: {V:.2f} В")

