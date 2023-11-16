"""
Created by Elias Obreque
Date: 16-11-2023
email: els.obrq@gmail.com
"""
import numba
import numpy as np
import time
from multiprocessing import Pool


# Función en Python puro que realiza una operación en un array
def my_function(x):
    result = 0
    for i in range(len(x)):
        result += x[i] ** 2
    return result


# Utilizando Numba para acelerar la función
@numba.jit(nopython=True)  # El decorador @jit indica que queremos compilar la función con Numba
def my_function_numba(x):
    result = 0
    for i in range(len(x)):
        result += x[i] ** 2
    return result


# Función que realiza cálculos usando Numba dentro de un proceso
def process_function(data):
    return my_function_numba(data)


if __name__ == '__main__':
    # Crear un array para probar la función
    arr = np.random.rand(100000000)

    # Medir el tiempo de ejecución sin Numba
    start_time = time.time()
    result_python = my_function(arr)
    print("Tiempo sin Numba:", time.time() - start_time)

    # Medir el tiempo de ejecución con Numba
    start_time = time.time()
    result_numba = my_function_numba(arr)
    print("Tiempo con Numba:", time.time() - start_time)

    # Dividir el array en partes para procesar en paralelo
    num_processes = 4
    split_data = np.array_split(arr, num_processes)
    # Inicializar el Pool de procesos
    start_time = time.time()
    with Pool(num_processes) as pool:
        # Aplicar la función en paralelo utilizando el Pool
        results = pool.map(process_function, split_data)
    # Combinar los resultados de los diferentes procesos
    final_result = sum(results)
    print("Tiempo con Numba y Pool:", time.time() - start_time)
    # Verificar que los resultados coincidan
    assert np.allclose(result_python, result_numba)
