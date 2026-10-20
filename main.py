import sys

import cv2
import numpy as np
import tensorflow as tf


def main():
    print("Sprawdzenie środowiska projektu")
    print(f"Interpreter: {sys.executable}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"TensorFlow: {tf.__version__}")
    print("Biblioteki zostały poprawnie zaimportowane.")


if __name__ == "__main__":
    main()