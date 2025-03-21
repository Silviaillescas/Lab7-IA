# Conecta 4 con TD Learning y Minimax
Michelle Mejía 22596 - Silvia Illescas 22376

Este proyecto implementa el juego **Connect Four** donde un agente entrenado con **Temporal Difference Learning (Q-Learning)** se enfrenta a jugadores basados en el algoritmo **Minimax**, con y sin **poda alfa-beta**. También permite que el usuario juegue contra la IA y que dos agentes IA compitan entre sí.

## Requisitos

- Python 3.x
- Bibliotecas:
  - `numpy`
  - `matplotlib`
  - `pickle` (incluida por defecto)

Instala las dependencias con:

```bash
pip install numpy matplotlib
```

## Cómo ejecutar

Ejecuta el archivo principal:

```bash
python Task2.py
```

Se mostrará un menú:

```
1. Jugar contra la IA (TD Learning)
2. Ejecutar torneo y generar gráfico
3. Jugar IA vs IA (TD Learning vs Minimax)
```

## ¿Qué hace el agente con TD Learning?

El agente con TD Learning utiliza **Q-Learning** para aprender a jugar Connect Four mediante autojuego. Aprende a partir de las recompensas recibidas en cada partida, ajustando su política con base en una función de valor. Se entrena con 5000 partidas antes de participar en los torneos.

### Recompensas

- +100 por ganar
- -100 por perder
- +1 por jugada propia
- -1 por jugada del oponente

## Torneo y evaluación

El torneo consiste en 150 partidas divididas así:

- 50 partidas: **TD Learning vs Minimax (sin poda)**
- 50 partidas: **TD Learning vs Minimax (con poda alfa-beta)**
- 50 partidas: **TD1 vs TD2** (ambos usan Q-Learning)

Al final se genera el gráfico `resultados_torneo.pdf` con las victorias de cada agente y empates, para evaluar el desempeño.

## Video Funcionamiento
https://www.youtube.com/watch?v=xFCI82oS-Nc
