import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

# --- 1. Pedir la imagen al usuario ---
imagen_path = input("Escribe la ruta o nombre del archivo de imagen: ").strip()

# Verifica si existe el archivo
if not os.path.exists(imagen_path):
    print(f"\n No se encontró la imagen en: {imagen_path}")
    print("Verifica el nombre o la ruta completa (por ejemplo /home/usuario/imagen.png)\n")
    exit()

# --- 2. Cargar y convertir a escala de grises ---
img = Image.open(imagen_path).convert('L')
print("Imagen cargada correctamente.")

# --- 3. Convertir a blanco y negro ---
# Puedes ajustar el umbral (100 → más claro, 120 → más oscuro)
_, binary = cv2.threshold(np.array(img), 100, 255, cv2.THRESH_BINARY_INV)

plt.imshow(binary, cmap='gray')
plt.title("Imagen binarizada (fractal)")
plt.axis('off')
plt.show()

# --- 4. Función para contar cuadros ocupados ---
def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                       np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where((S > 0) & (S < k*k*255))[0])

# --- 5. Calcular N(ε) para distintos tamaños de caja ---
Z = binary
sizes = 2**np.arange(1, int(np.log2(min(Z.shape))), 1)
counts = []

for size in sizes:
    counts.append(boxcount(Z, size))

# --- 6. Calcular la pendiente (dimensión fractal) ---
coeffs = np.polyfit(np.log(1/sizes), np.log(counts), 1)
D = coeffs[0]

# --- 7. Mostrar resultado ---
plt.plot(np.log(1/sizes), np.log(counts), 'o-', color='blue')
plt.xlabel("log(1/ε)")
plt.ylabel("log(N(ε))")
plt.title(f"Dimensión fractal estimada: D = {D:.3f}")
plt.show()

print(f"\n Dimensión fractal estimada: D = {D:.3f}\n")

