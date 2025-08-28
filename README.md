# Amoniaco

Herramienta gráfica para estimar el coste nivelado de producción de amoníaco.
Permite comparar distintas rutas electroquímicas en una sola interfaz.

## Uso

Ejecuta `NH3.py` para abrir la interfaz. El programa permite calcular el LCOA y
realizar distintos análisis de sensibilidad. Desde el menú superior se puede
seleccionar la tecnología (**NRR Directa** o **NRR Li-mediada**) y utilizar el
botón *Comparar tecnologías* para obtener el LCOA de ambas rutas de forma
simultánea.

### Nuevas funciones

- Guardar y cargar conjuntos de parámetros en formato JSON desde el menú
  **Archivo**.
- Restablecer rápidamente los valores por defecto de los parámetros.
- En la pestaña **Sensibilidades 2D** se pueden generar curvas η_FE vs j_tot
  para distintos valores de LCOA (por defecto 1, 2 y 6 €/kg) y exportarlas a CSV
  mediante el botón "Guardar Datos de Curvas" presente en la ventana del
  gráfico.
