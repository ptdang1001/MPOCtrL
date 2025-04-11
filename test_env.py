# this is a python file to test if the libraies are installed correctly


def test_python_libraries():
    import numpy as np

    print("Numpy Installed Successfully!")
    import pandas as pd

    print("Pandas Installed Successfully!")
    import matplotlib.pyplot as plt

    print("Matplotlib Installed Successfully!")
    import sklearn

    print("Sklearn Installed Successfully!")
    import torch

    print("Pytorch Installed Successfully!")
    import lightning

    print("Pytorch Lightning Installed Successfully!")
    import tqdm

    print("Tqdm Installed Successfully!")
    import numba

    print("Numba Installed Successfully!")

    import fireducks.pandas as pd

    print("Fireducks Installed Successfully!")

    return True


if __name__ == "__main__":
    result = test_python_libraries()
    if result:
        print("All libraries are installed correctly")
    else:
        print("Some libraries are not installed correctly")
