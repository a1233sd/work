import re
import numpy as np
from scipy.linalg import solve


# Функция для парсинга входного файла, который содержит описание элементов схемы
def parse_input_file(file_path):
    elements = []
    with open(file_path, 'r') as file:
        for line in file:
            elements.append(line.strip())
    return elements


# Функция для построения матрицы проводимостей (G) и вектора токов (I) по методу узловых потенциалов
def build_mna_matrix(elements):
    nodes = set()  # Множество для хранения всех узлов
    resistors = []  # Список для хранения резисторов
    voltage_sources = []  # Список для хранения источников напряжения

    # Регулярное выражение для парсинга строк с резисторами
    resistor_pattern = re.compile(r'^R:(\w+)\s+(\w+)\s+(\w+)\s+R=(\d+\.?\d*)$')
    # Регулярное выражение для парсинга строк с источниками напряжения
    voltage_source_pattern = re.compile(r'^Vsrc:(\w+)\s+(\w+)\s+(\w+)\s+[UV]=(\d+\.?\d*)$')

    # Парсинг элементов схемы
    for element in elements:
        resistor_match = resistor_pattern.match(element)
        voltage_source_match = voltage_source_pattern.match(element)

        if resistor_match:
            _, node1, node2, resistance = resistor_match.groups()
            resistance = float(resistance)
            resistors.append((node1, node2, resistance))
            nodes.update([node1, node2])
        elif voltage_source_match:
            _, node1, node2, voltage = voltage_source_match.groups()
            voltage = float(voltage)
            voltage_sources.append((node1, node2, voltage))
            nodes.update([node1, node2])
        else:
            raise ValueError(f"Неверный формат строки: {element}")

    # Исключаем узел 'gnd' (земля) из множества узлов
    nodes.discard('gnd')
    node_list = list(nodes)
    # Создаем индекс для каждого узла
    node_index = {node: i for i, node in enumerate(node_list)}

    num_nodes = len(node_list)
    num_volt_sources = len(voltage_sources)

    # Инициализация матрицы проводимостей (G) и вектора токов (I)
    G = np.zeros((num_nodes + num_volt_sources, num_nodes + num_volt_sources))
    I = np.zeros(num_nodes + num_volt_sources)

    # Заполнение матрицы проводимостей для резисторов
    for (node1, node2, resistance) in resistors:
        if node1 != 'gnd' and node2 != 'gnd':
            n1, n2 = node_index[node1], node_index[node2]
            G[n1][n1] += 1 / resistance
            G[n2][n2] += 1 / resistance
            G[n1][n2] -= 1 / resistance
            G[n2][n1] -= 1 / resistance
        elif node1 == 'gnd':
            n2 = node_index[node2]
            G[n2][n2] += 1 / resistance
        elif node2 == 'gnd':
            n1 = node_index[node1]
            G[n1][n1] += 1 / resistance

    # Заполнение матрицы проводимостей и вектора токов для источников напряжения
    for idx, (node1, node2, voltage) in enumerate(voltage_sources):
        if node1 == 'gnd':
            n2 = node_index[node2]
            voltage_index = num_nodes + idx
            G[voltage_index][n2] = -1
            G[n2][voltage_index] = -1
            I[voltage_index] = -voltage
        elif node2 == 'gnd':
            n1 = node_index[node1]
            voltage_index = num_nodes + idx
            G[voltage_index][n1] = 1
            G[n1][voltage_index] = 1
            I[voltage_index] = voltage
        else:
            n1, n2 = node_index[node1], node_index[node2]
            voltage_index = num_nodes + idx
            G[voltage_index][n1] = -1
            G[voltage_index][n2] = 1
            G[n1][voltage_index] = -1
            G[n2][voltage_index] = 1
            I[voltage_index] = voltage

    return G, I, node_index



# Функция для записи результатов в выходной файл
def write_output_file(file_path, node_voltages):
    with open(file_path, 'w') as file:
        for node, voltage in node_voltages.items():
            file.write(f"{node} {voltage}\n")


# Главная функция
def main():
    input_file = 'input_example.txt'
    output_file = 'output_example.txt'
    # Парсинг входного файла
    elements = parse_input_file(input_file)
    # Построение матрицы проводимостей и вектора токов
    G, I, node_index = build_mna_matrix(elements)

    # Вывод матрицы проводимостей и вектора токов для отладки
    print("Матрица G:")
    print(G)
    print("Вектор I:")
    print(I)

    # Решение системы уравнений
    V = solve(G, I)

    # Создание словаря узловых напряжений
    node_voltages = {node: V[index] for node, index in node_index.items()}
    # Запись результатов в выходной файл
    write_output_file(output_file, node_voltages)
    print("Напряжения на узлах:", node_voltages)


if __name__ == "__main__":
    main()
