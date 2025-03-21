import numpy as np
import random
import matplotlib.pyplot as plt

# Parámetros del juego
ROWS = 6
COLS = 7
PLAYER = 1  # Jugador humano
AI_TD = 2   # IA con TD Learning
ALPHA = 0.05  # Tasa de aprendizaje (ajustada para mejorar el rendimiento)
GAMMA = 0.9  # Factor de descuento
EPSILON = 0.1  # Probabilidad de exploración (reducida después de entrenamiento)

# Inicialización de la tabla Q (estado -> acción)
Q_table = {}

def crear_tablero():
    return np.zeros((ROWS, COLS), dtype=int)

def obtener_movimientos_validos(tablero):
    return [col for col in range(COLS) if tablero[0][col] == 0]

def obtener_fila_valida(tablero, col):
    for fila in range(ROWS-1, -1, -1):
        if tablero[fila, col] == 0:
            return int(fila)
    return None

def hacer_movimiento(tablero, col, pieza):
    fila = obtener_fila_valida(tablero, col)
    if fila is not None:
        tablero[fila, col] = pieza
        return True
    return False

def verificar_victoria(tablero, pieza):
    for c in range(COLS-3):
        for r in range(ROWS):
            if all(tablero[r, c+i] == pieza for i in range(4)):
                return True
    for c in range(COLS):
        for r in range(ROWS-3):
            if all(tablero[r+i, c] == pieza for i in range(4)):
                return True
    for c in range(COLS-3):
        for r in range(ROWS-3):
            if all(tablero[r+i, c+i] == pieza for i in range(4)):
                return True
    for c in range(COLS-3):
        for r in range(3, ROWS):
            if all(tablero[r-i, c+i] == pieza for i in range(4)):
                return True
    return False

def elegir_accion(tablero):
    """Elige una acción usando ε-greedy."""
    movimientos_validos = obtener_movimientos_validos(tablero)
    if not movimientos_validos:
        return None

    estado = str(tablero)
    if estado not in Q_table:
        Q_table[estado] = np.zeros(COLS)

    if random.uniform(0, 1) < EPSILON:
        return random.choice(movimientos_validos)

    return max(movimientos_validos, key=lambda col: Q_table[estado][col])

def entrenar_td_learning(num_partidas=5000):
    """TD Learning juega contra sí mismo antes del torneo."""
    print(f"Entrenando TD Learning con {num_partidas} partidas...")

    for _ in range(num_partidas):
        tablero = crear_tablero()
        turno = random.choice([AI_TD, PLAYER])

        while True:
            col = elegir_accion(tablero)
            if col is None:
                break

            hacer_movimiento(tablero, col, turno)

            if verificar_victoria(tablero, turno):
                recompensa = 100 if turno == AI_TD else -100
                break
            else:
                recompensa = 0

            estado = str(tablero)
            if estado not in Q_table:
                Q_table[estado] = np.zeros(COLS)

            Q_table[estado][col] += ALPHA * (recompensa + GAMMA * np.max(Q_table.get(str(tablero), np.zeros(COLS))) - Q_table[estado][col])

            turno = PLAYER if turno == AI_TD else AI_TD

    print("Entrenamiento completado.")

def minimax(tablero, profundidad, alpha, beta, maximizando, poda=True):
    movimientos_validos = obtener_movimientos_validos(tablero)
    es_terminal = verificar_victoria(tablero, PLAYER) or verificar_victoria(tablero, AI_TD) or len(movimientos_validos) == 0

    if profundidad == 0 or es_terminal:
        if verificar_victoria(tablero, AI_TD):
            return None, 1000000
        elif verificar_victoria(tablero, PLAYER):
            return None, -1000000
        elif len(movimientos_validos) == 0:
            return None, 0
        return None, 0

    if maximizando:
        valor_max = -np.inf
        mejor_col = random.choice(movimientos_validos)
        for col in movimientos_validos:
            copia_tablero = tablero.copy()
            hacer_movimiento(copia_tablero, col, AI_TD)
            _, nuevo_valor = minimax(copia_tablero, profundidad-1, alpha, beta, False, poda)
            if nuevo_valor > valor_max:
                valor_max = nuevo_valor
                mejor_col = col
            if poda:
                alpha = max(alpha, valor_max)
                if alpha >= beta:
                    break
        return mejor_col, valor_max
    else:
        valor_min = np.inf
        mejor_col = random.choice(movimientos_validos)
        for col in movimientos_validos:
            copia_tablero = tablero.copy()
            hacer_movimiento(copia_tablero, col, PLAYER)
            _, nuevo_valor = minimax(copia_tablero, profundidad-1, alpha, beta, True, poda)
            if nuevo_valor < valor_min:
                valor_min = nuevo_valor
                mejor_col = col
            if poda:
                beta = min(beta, valor_min)
                if alpha >= beta:
                    break
        return mejor_col, valor_min

def jugar_humano_vs_ia():
    """Permite que un humano juegue contra la IA TD Learning."""
    tablero = crear_tablero()
    turno = random.choice([PLAYER, AI_TD])

    while True:
        print(tablero)

        if turno == PLAYER:
            col = int(input("Elige una columna (0-6): "))
            if col not in obtener_movimientos_validos(tablero):
                print("Movimiento inválido. Inténtalo de nuevo.")
                continue
        else:
            col = elegir_accion(tablero)

        hacer_movimiento(tablero, col, turno)

        if verificar_victoria(tablero, turno):
            print(tablero)
            print("¡Ganaste!" if turno == PLAYER else "La IA ganó.")
            break

        turno = AI_TD if turno == PLAYER else PLAYER

def jugar_ia_vs_ia():
    """Simula partidas entre dos IAs TD Learning."""
    victorias_td = 0

    for _ in range(50):
        tablero = crear_tablero()
        turno = AI_TD

        while True:
            col = elegir_accion(tablero)
            if col is None:
                break

            hacer_movimiento(tablero, col, turno)

            if verificar_victoria(tablero, turno):
                victorias_td += 1
                break

            turno = AI_TD

    print(f"IA TD Learning ganó {victorias_td} de 50 partidas.")

def jugar_td_vs_minimax(num_juegos=50, poda=False):
    """Ejecuta partidas de TD Learning vs Minimax y devuelve el número de victorias."""
    victorias_td = 0
    victorias_minimax = 0

    for _ in range(num_juegos):
        tablero = crear_tablero()
        turno = random.choice([AI_TD, PLAYER])

        while True:
            if turno == AI_TD:
                col = elegir_accion(tablero)
                if col is None:
                    break
            else:
                col, _ = minimax(tablero, 3, -np.inf, np.inf, True, poda)

            hacer_movimiento(tablero, col, turno)

            if verificar_victoria(tablero, turno):
                if turno == AI_TD:
                    victorias_td += 1
                else:
                    victorias_minimax += 1
                break

            turno = PLAYER if turno == AI_TD else AI_TD

    return victorias_td, victorias_minimax

def jugar_torneo():
    """Ejecuta 150 partidas entre diferentes combinaciones de agentes y genera gráficos."""

    entrenar_td_learning(num_partidas=5000)  # Entrenar TD Learning antes del torneo

    print("Jugando TD vs Minimax (SIN poda alfa-beta)...")
    victorias_td_sin_poda, victorias_minimax_sin_poda = jugar_td_vs_minimax(50, poda=False)
    print(f"TD Learning ganó {victorias_td_sin_poda} partidas, Minimax ganó {victorias_minimax_sin_poda} partidas.")

    print("Jugando TD vs Minimax (CON poda alfa-beta)...")
    victorias_td_con_poda, victorias_minimax_con_poda = jugar_td_vs_minimax(50, poda=True)
    print(f"TD Learning ganó {victorias_td_con_poda} partidas, Minimax ganó {victorias_minimax_con_poda} partidas.")

    print("Jugando TD vs TD...")
    victorias_td_vs_td, empates = jugar_td_vs_minimax(50)
    print(f"TD1 ganó {victorias_td_vs_td} partidas, TD2 ganó {50 - victorias_td_vs_td - empates} partidas, Empates: {empates}.")

    categorias = ["TD vs Minimax (sin poda)", "TD vs Minimax (con poda)", "TD1 vs TD2"]
    victorias_td_lista = [victorias_td_sin_poda, victorias_td_con_poda, victorias_td_vs_td]
    victorias_oponente_lista = [victorias_minimax_sin_poda, victorias_minimax_con_poda, 50 - victorias_td_vs_td - empates]
    empates_lista = [0, 0, empates]

    plt.figure(figsize=(8, 6))
    plt.bar(categorias, victorias_td_lista, label="TD Learning / TD1", color="blue")
    plt.bar(categorias, victorias_oponente_lista, bottom=victorias_td_lista, label="Minimax / TD2", color="red")
    plt.bar(categorias, empates_lista, bottom=np.array(victorias_td_lista) + np.array(victorias_oponente_lista), label="Empates", color="gray")
    plt.ylabel("Número de Partidas")
    plt.xlabel("Tipo de Partida")
    plt.title("Resultados del Torneo de Connect Four")
    plt.legend()
    plt.savefig("resultados_torneo.pdf")
    plt.show()


def main():
    print("1. Jugar contra la IA (TD Learning)")
    print("2. Ejecutar torneo y generar gráfico")
    print("3. Jugar IA vs IA (TD Learning vs Minimax)")

    opcion = input("Elige una opción (1, 2 o 3): ")

    if opcion == "1":
        jugar_humano_vs_ia()
    elif opcion == "2":
        jugar_torneo()
    elif opcion == "3":
        jugar_ia_vs_ia()
    else:
        print("Opción inválida.")

if __name__ == "__main__":
    main()
