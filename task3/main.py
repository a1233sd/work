import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# Чтение данных из файла
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Преобразуем строку в комплексное число
    data['Input'] = data['Input'].apply(complex)
    data['Output'] = data['Output'].apply(complex)
    return data


# Функция для создания полиномиальных признаков с памятью
def create_memory_polynomial_features(x_real, x_imag, memory_depth=3, degree=5):
    features = []
    for i in range(memory_depth):
        # Сдвиг реальной части
        shifted_real = np.roll(x_real, i)
        if i > 0:
            shifted_real[:i] = 0  # Обнуляем сдвинутые значения
        for d in range(1, degree + 1):
            features.append(np.power(shifted_real, d))

        # Сдвиг мнимой части
        shifted_imag = np.roll(x_imag, i)
        if i > 0:
            shifted_imag[:i] = 0  # Обнуляем сдвинутые значения
        for d in range(1, degree + 1):
            features.append(np.power(shifted_imag, d))

    return np.column_stack(features)


# Обучение модели
def train_model(X, y):
    model = Ridge(alpha=1.0)  # Используем Ridge регрессию
    model.fit(X, y)
    return model


# Функция для предсказания
def predict(model, X):
    return model.predict(X)


# Основная функция
def main():
    file_path = 'D:/python/work/num2/Amp_C_train.txt'  # Укажите правильный путь к файлу

    # Проверяем существование файла
    if not os.path.exists(file_path):
        print(f"Файл не найден по указанному пути: {file_path}")
        return  # Завершаем программу, если файл не найден
    else:
        print(f"Файл найден: {file_path}")

    # Загрузка данных
    data = load_data(file_path)

    # Разделение входных сигналов на реальные и мнимые части
    input_real = np.array(data['Input'].apply(lambda z: z.real))
    input_imag = np.array(data['Input'].apply(lambda z: z.imag))

    # Разделение выходных сигналов на реальные и мнимые части
    output_real = np.array(data['Output'].apply(lambda z: z.real))
    output_imag = np.array(data['Output'].apply(lambda z: z.imag))

    # Параметры модели
    memory_depth = 10  # Глубина памяти
    degree = 10  # Степень полинома

    # Создание признаков
    X = create_memory_polynomial_features(input_real, input_imag, memory_depth, degree)

    # Убираем начальные значения, которые были сдвинуты
    X = X[memory_depth:]
    y_real_trimmed = output_real[memory_depth:]
    y_imag_trimmed = output_imag[memory_depth:]

    # Нормализация признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Обучение модели для реальной части
    model_real = train_model(X_scaled, y_real_trimmed)

    # Обучение модели для мнимой части
    model_imag = train_model(X_scaled, y_imag_trimmed)

    # Предсказание
    predicted_real = predict(model_real, X_scaled)
    predicted_imag = predict(model_imag, X_scaled)
    predicted_output = predicted_real + 1j * predicted_imag

    # Расчет RMSE для реальной и мнимой частей
    rmse_real = np.sqrt(mean_squared_error(y_real_trimmed, predicted_real))
    rmse_imag = np.sqrt(mean_squared_error(y_imag_trimmed, predicted_imag))
    rmse = np.sqrt(rmse_real ** 2 + rmse_imag ** 2)
    print(f'RMSE (Real): {rmse_real}')
    print(f'RMSE (Imag): {rmse_imag}')
    print(f'Общий RMSE: {rmse}')



if __name__ == "__main__":
    main()
