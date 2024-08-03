import re
import numpy as np
from scipy.linalg import solve


def parse_input_file(file_path):
    elements = []
    with open(file_path, 'r') as file:
        for line in file:
            elements.append(line.strip())
    return elements


def build_mna_matrix(elements):
    nodes = set()
    resistors = []
    voltage_sources = []

    resistor_pattern = re.compile(r'^R:(\w+)\s+(\w+)\s+(\w+)\s+R=(\d+\.?\d*)$')
    voltage_source_pattern = re.compile(r'^Vsrc:(\w+)\s+(\w+)\s+(\w+)\s+[UV]=(\d+\.?\d*)$')

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

    nodes.discard('gnd')
    node_list = list(nodes)
    node_index = {node: i for i, node in enumerate(node_list)}

    num_nodes = len(node_list)
    num_voltage_sources = len(voltage_sources)

    G = np.zeros((num_nodes, num_nodes))
    I = np.zeros(num_nodes)

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

    for (node1, node2, voltage) in voltage_sources:
        if node1 == 'gnd':
            n2 = node_index[node2]
            I[n2] -= voltage
        elif node2 == 'gnd':
            n1 = node_index[node1]
            I[n1] += voltage
        else:
            n1, n2 = node_index[node1], node_index[node2]
            G = np.pad(G, ((0, 1), (0, 1)), mode='constant')
            I = np.pad(I, (0, 1), mode='constant')
            G[n1][-1] = 1
            G[-1][n1] = 1
            G[n2][-1] = -1
            G[-1][n2] = -1
            I[-1] = voltage

    # Удаление последней строки и столбца, если они добавлены для источников напряжения
    G = G[:num_nodes, :num_nodes]
    I = I[:num_nodes]

    return G, I, node_index


def solve_mna(G, I):
    V = solve(G, I)
    return V


def write_output_file(file_path, node_voltages):
    with open(file_path, 'w') as file:
        for node, voltage in node_voltages.items():
            file.write(f"{node} {voltage}\n")


def main():
    input_file = 'input_example.txt'
    output_file = 'output_example.txt'

    elements = parse_input_file(input_file)
    G, I, node_index = build_mna_matrix(elements)

    # Проверка матрицы G и вектора I перед решением
    print("Матрица G:")
    print(G)
    print("Вектор I:")
    print(I)

    V = solve_mna(G, I)

    node_voltages = {node: V[index] for node, index in node_index.items()}
    write_output_file(output_file, node_voltages)


if __name__ == "__main__":
    main()
