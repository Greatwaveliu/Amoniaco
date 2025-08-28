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
- En la pestaña **Sensibilidades 2D** se puede guardar la curva de paridad
  mostrada en un archivo CSV mediante el botón "Guardar Datos de Curvas" que
  aparece en la ventana del gráfico generado. Si el nivel de paridad no cruza el
  mapa calculado, se mostrará un aviso de "Sin datos".
