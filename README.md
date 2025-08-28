# Amoniaco

Herramienta gráfica para estimar el coste nivelado de producción de amoníaco.

## Uso

Ejecuta `NH3.py` para abrir la interfaz. El programa permite calcular el LCOA y
realizar distintos análisis de sensibilidad.

### Ruta mediada con litio

El script `NH3_lithium.py` incorpora un modelo económico simplificado para la
síntesis electroquímica de amoníaco mediada con litio. Al ejecutarlo se generan
dos gráficas:

- `lcoa_vs_j.png`: LCOA en función de la densidad de corriente total.
- `parity_vs_price.png`: Densidad de corriente requerida para alcanzar un LCOA
  objetivo en función del precio de la electricidad.

### Nuevas funciones

- Guardar y cargar conjuntos de parámetros en formato JSON desde el menú
  **Archivo**.
- Restablecer rápidamente los valores por defecto de los parámetros.
