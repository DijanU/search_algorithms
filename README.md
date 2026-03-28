# 🧩 Maze Solver — Proyecto #2

Implementación y comparación de algoritmos de búsqueda informada y no informada para resolver laberintos de 128×128 en Python, con interfaz interactiva en Streamlit.

## Algoritmos implementados

| Algoritmo | Tipo | Heurística | Óptimo |
|-----------|------|------------|--------|
| BFS | No informado | — | ✅ |
| DFS | No informado | — | ❌ |
| Greedy Best-First | Informado | Manhattan / Euclidiana | ❌ |
| A* | Informado | Manhattan / Euclidiana | ✅ |

## Estructura del repositorio

```
.
├── mazeApp.py           # Interfaz interactiva (Streamlit)
├── maze_solver_v2.ipynb # Notebook de análisis y experimentos
├── reporte.md           # Reporte escrito del proyecto
├── Fong_runs/           # Laberintos de prueba (.txt)
│   ├── Laberinto1-2.txt
│   ├── Laberinto2-2.txt
│   └── Laberinto3-2.txt
└── README.md
```

## Formato del laberinto

Los laberintos se representan en archivos `.txt` de 128×128 caracteres:

| Símbolo | Significado |
|---------|-------------|
| `0` | Camino libre |
| `1` | Pared |
| `2` | Punto de partida |
| `3` | Punto de salida |

La jerarquía de movimientos es: **Arriba → Derecha → Abajo → Izquierda**.

## Interfaz interactiva (Streamlit)

La app `mazeApp.py` permite cargar cualquier laberinto `.txt` y visualizar la resolución con todos los algoritmos.

### Instalación de dependencias

```bash
pip install streamlit numpy matplotlib pandas
```

### Correr la app

```bash
streamlit run mazeApp.py
```

La app abrirá en `http://localhost:8501` con las siguientes funcionalidades:

- **Carga de laberinto** desde archivo `.txt`
- **Selección de algoritmo** (o ejecutar todos a la vez)
- **Visualización del laberinto** con el camino solución marcado
- **Métricas por algoritmo:** nodos visitados, largo del camino y tiempo de ejecución
- **Simulación aleatoria** con N puntos de inicio aleatorios configurables
- **Opción de invertir** pasillos/paredes para laberintos con convención opuesta

## Métricas evaluadas

- **Longitud del camino:** número de pasos desde inicio hasta la meta
- **Nodos explorados:** cantidad de estados expandidos hasta encontrar la solución
- **Tiempo de ejecución:** medido en segundos con `time.time()`
- **Branching factor efectivo:** calculado como `nodos^(1/profundidad)`

## Resultados principales

En los tres laberintos evaluados:

- **BFS y A\*** encontraron siempre el camino óptimo
- **Greedy Manhattan** exploró consistentemente el menor número de nodos
- **DFS** produjo caminos de hasta 285% del óptimo
- La heurística **Manhattan** resultó más informativa que Euclidiana para movimientos ortogonales
- Algunos laberintos presentan **componentes conexas aisladas** sin solución desde ciertos puntos

Para el análisis completo, ver [`reporte.md`](./reporte.md).

---

*Proyecto #2 · Búsqueda y Heurísticas · Universidad del Valle de Guatemala*