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
- La pestaña **Curvas U_cell vs η_FE** traza la eficiencia faradaica frente al
  potencial de celda para energías específicas seleccionadas (por defecto
  60, 100 y 150 kWh/kg) y permite guardar los datos mostrados a CSV mediante el
  botón "Guardar Datos de Curvas" en la figura.
- El análisis económico incorpora el consumo de H₂ y Li en la ruta mediada,
  con valores por defecto basados en datos experimentales (U_cell = 4 V,
  η_FE = 0.30, j_tot = 500 A/m²).
