import numpy as np
import random

# Parámetros del juego
ROWS = 6
COLS = 7
PLAYER = 1  # Jugador humano
AI = 2      # IA

def crear_tablero():
    """Crea un tablero vacío de Connect Four."""
    return np.zeros((ROWS, COLS), dtype=int)

def movimiento_valido(tablero, col):
    """Verifica si una columna tiene espacio disponible."""
    return tablero[0][col] == 0

def obtener_fila_valida(tablero, col):
    """Devuelve la fila más baja disponible en la columna seleccionada."""
    for fila in range(ROWS-1, -1, -1):
        if tablero[fila][col] == 0:
            return fila
    return None

def hacer_movimiento(tablero, col, pieza):
    """Coloca una pieza en el tablero si es posible."""
    fila = obtener_fila_valida(tablero, col)
    if fila is not None:
        tablero[fila][col] = pieza
        return True
    return False

def imprimir_tablero(tablero):
    """Imprime el tablero en formato visual."""
    print(np.flip(tablero, 0))  # Invierte para que la fila 0 esté abajo

def verificar_victoria(tablero, pieza):
    """Verifica si un jugador ha ganado."""
    # Horizontal
    for c in range(COLS-3):
        for r in range(ROWS):
            if all(tablero[r, c+i] == pieza for i in range(4)):
                return True
    # Vertical
    for c in range(COLS):
        for r in range(ROWS-3):
            if all(tablero[r+i, c] == pieza for i in range(4)):
                return True
    # Diagonal positiva
    for c in range(COLS-3):
        for r in range(ROWS-3):
            if all(tablero[r+i, c+i] == pieza for i in range(4)):
                return True
    # Diagonal negativa
    for c in range(COLS-3):
        for r in range(3, ROWS):
            if all(tablero[r-i, c+i] == pieza for i in range(4)):
                return True
    return False

def obtener_movimientos_validos(tablero):
    """Devuelve una lista de columnas con espacio disponible."""
    return [col for col in range(COLS) if movimiento_valido(tablero, col)]

def evaluar_tablero(tablero, pieza):
    """Evalúa el tablero desde la perspectiva de la IA."""
    oponente = PLAYER if pieza == AI else AI
    score = 0

    # Puntuación para el centro del tablero
    centro_columna = [int(i) for i in list(tablero[:, COLS//2])]
    score += centro_columna.count(pieza) * 3

    # Evaluación de líneas
    for r in range(ROWS):
        fila_array = [int(i) for i in list(tablero[r, :])]
        score += evaluar_linea(fila_array, pieza, oponente)

    for c in range(COLS):
        col_array = [int(i) for i in list(tablero[:, c])]
        score += evaluar_linea(col_array, pieza, oponente)

    for r in range(ROWS-3):
        for c in range(COLS-3):
            diag_pos = [tablero[r+i][c+i] for i in range(4)]
            score += evaluar_linea(diag_pos, pieza, oponente)

            diag_neg = [tablero[r+3-i][c+i] for i in range(4)]
            score += evaluar_linea(diag_neg, pieza, oponente)

    return score

def evaluar_linea(linea, pieza, oponente):
    """Evalúa una línea de 4 casillas."""
    score = 0
    if linea.count(pieza) == 4:
        score += 100
    elif linea.count(pieza) == 3 and linea.count(0) == 1:
        score += 5
    elif linea.count(pieza) == 2 and linea.count(0) == 2:
        score += 2
    if linea.count(oponente) == 3 and linea.count(0) == 1:
        score -= 4
    return score

def minimax(tablero, profundidad, alpha, beta, maximizando, poda=True):
    """Algoritmo Minimax con opción de poda alfa-beta."""
    movimientos_validos = obtener_movimientos_validos(tablero)
    es_terminal = verificar_victoria(tablero, PLAYER) or verificar_victoria(tablero, AI) or len(movimientos_validos) == 0

    if profundidad == 0 or es_terminal:
        if verificar_victoria(tablero, AI):
            return (None, 1000000)
        elif verificar_victoria(tablero, PLAYER):
            return (None, -1000000)
        elif len(movimientos_validos) == 0:
            return (None, 0)
        return (None, evaluar_tablero(tablero, AI))

    if maximizando:
        valor_max = -np.inf
        mejor_col = random.choice(movimientos_validos)
        for col in movimientos_validos:
            copia_tablero = tablero.copy()
            hacer_movimiento(copia_tablero, col, AI)
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

def jugar():
    """Función para jugar contra la IA."""
    tablero = crear_tablero()
    imprimir_tablero(tablero)
    turno = random.choice([PLAYER, AI])
    poda = True  # Se puede cambiar a False para desactivar alpha-beta pruning

    while True:
        if turno == PLAYER:
            try:
                col = int(input("Elige una columna (0-6): "))
                if col not in range(COLS) or not movimiento_valido(tablero, col):
                    print("Movimiento inválido. Inténtalo de nuevo.")
                    continue
                hacer_movimiento(tablero, col, PLAYER)
                if verificar_victoria(tablero, PLAYER):
                    imprimir_tablero(tablero)
                    print("¡Ganaste!")
                    break
                turno = AI
            except ValueError:
                print("Entrada no válida. Introduce un número entre 0 y 6.")
        else:
            print("Turno de la IA...")
            col, _ = minimax(tablero, 3, -np.inf, np.inf, True, poda)  # Reducir profundidad para mejorar tiempos
            if col is not None:
                hacer_movimiento(tablero, col, AI)
                if verificar_victoria(tablero, AI):
                    imprimir_tablero(tablero)
                    print("La IA ganó.")
                    break
                turno = PLAYER
            else:
                print("Empate.")
                break

        imprimir_tablero(tablero)

def jugar_ia_vs_ia(num_juegos=10):
    """Simula partidas entre IA sin poda alfa-beta vs. IA con poda alfa-beta."""
    victorias_sin_poda = 0
    victorias_con_poda = 0
    empates = 0

    for i in range(num_juegos):
        print(f"\nJuego {i+1}/{num_juegos}")
        tablero = crear_tablero()
        turno = random.choice([AI, PLAYER])  # Elegir aleatoriamente quién comienza
        poda = False  # Alternar poda alfa-beta en cada turno

        while True:
            imprimir_tablero(tablero)

            if turno == AI:
                col, _ = minimax(tablero, 3, -np.inf, np.inf, True, poda)
                hacer_movimiento(tablero, col, AI)
                if verificar_victoria(tablero, AI):
                    imprimir_tablero(tablero)
                    if poda:
                        victorias_con_poda += 1
                        print("¡IA con poda alfa-beta ganó!")
                    else:
                        victorias_sin_poda += 1
                        print("¡IA sin poda alfa-beta ganó!")
                    break
            else:
                col, _ = minimax(tablero, 3, -np.inf, np.inf, True, not poda)
                hacer_movimiento(tablero, col, PLAYER)
                if verificar_victoria(tablero, PLAYER):
                    imprimir_tablero(tablero)
                    if not poda:
                        victorias_sin_poda += 1
                        print("¡IA sin poda alfa-beta ganó!")
                    else:
                        victorias_con_poda += 1
                        print("¡IA con poda alfa-beta ganó!")
                    break

            # Alternar poda alfa-beta en cada turno
            poda = not poda
            turno = AI if turno == PLAYER else PLAYER

            if len(obtener_movimientos_validos(tablero)) == 0:
                empates += 1
                print("¡Empate!")
                break

    print("\nResultados Finales:")
    print(f"IA sin poda alfa-beta ganó {victorias_sin_poda} veces")
    print(f"IA con poda alfa-beta ganó {victorias_con_poda} veces")
    print(f"Empates: {empates}")

if __name__ == "__main__":
    modo = input("Elige el modo de juego: 1 = Humano vs IA, 2 = IA vs IA: ")
    if modo == "1":
        jugar()
    elif modo == "2":
        jugar_ia_vs_ia(10)
    else:
        print("Opción inválida.")
