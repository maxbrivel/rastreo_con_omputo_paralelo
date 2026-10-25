import cv2 as cv
import numpy as np
import time
import multiprocessing


# ==================================================
# 1. PROCESAMIENTO DE CADA CUADRANTE
# ==================================================

def procesar_cuadrante(argumentos):
    """
    Calcula la media de intensidad y detecta contornos
    de las regiones que cambiaron entre dos fotogramas.

    Esta función se ejecuta en los procesos del Pool.
    """

    fragmento_actual, fragmento_referencia = argumentos

    # ----------------------------------------------
    # Calcular media de intensidad
    # ----------------------------------------------

    media_actual = float(np.mean(fragmento_actual))

    # ----------------------------------------------
    # Calcular diferencia entre los fotogramas
    # ----------------------------------------------

    diferencia = cv.absdiff(
        fragmento_actual,
        fragmento_referencia
    )

    # ----------------------------------------------
    # Convertir la diferencia en una máscara binaria
    # ----------------------------------------------

    _, mascara = cv.threshold(
        diferencia,
        25,
        255,
        cv.THRESH_BINARY
    )

    # ----------------------------------------------
    # Eliminar pequeñas variaciones y ruido
    # ----------------------------------------------

    kernel = np.ones((3, 3), np.uint8)

    mascara = cv.morphologyEx(
        mascara,
        cv.MORPH_OPEN,
        kernel
    )

    # Unir regiones cercanas que pertenezcan al mismo objeto
    mascara = cv.dilate(
        mascara,
        kernel,
        iterations=2
    )

    # ----------------------------------------------
    # Encontrar los contornos de las regiones cambiadas
    # ----------------------------------------------

    contornos, _ = cv.findContours(
        mascara,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    # ----------------------------------------------
    # Filtrar contornos demasiado pequeños
    # ----------------------------------------------

    contornos_validos = []

    AREA_MINIMA = 700

    for contorno in contornos:

        area = cv.contourArea(contorno)

        if area >= AREA_MINIMA:
            contornos_validos.append(contorno)

    return media_actual, contornos_validos


# ==================================================
# 2. DIVIDIR LA IMAGEN EN OCHO CUADRANTES
# ==================================================

def obtener_cuadrantes(frame):

    alto, ancho = frame.shape[:2]

    h_m = alto // 2
    w_c = ancho // 4

    coordenadas = [
        # Fila superior
        (0, 0, w_c, h_m),
        (w_c, 0, w_c * 2, h_m),
        (w_c * 2, 0, w_c * 3, h_m),
        (w_c * 3, 0, ancho, h_m),

        # Fila inferior
        (0, h_m, w_c, alto),
        (w_c, h_m, w_c * 2, alto),
        (w_c * 2, h_m, w_c * 3, alto),
        (w_c * 3, h_m, ancho, alto)
    ]

    fragmentos = []

    for x1, y1, x2, y2 in coordenadas:

        fragmento = frame[y1:y2, x1:x2]

        fragmentos.append(fragmento)

    return fragmentos, coordenadas


# ==================================================
# 3. PROGRAMA PRINCIPAL
# ==================================================

def main():

    # ----------------------------------------------
    # Configurar la fuente de video
    # ----------------------------------------------

    RUTA_VIDEO = None

    if RUTA_VIDEO:

        camara = cv.VideoCapture(RUTA_VIDEO)

        fps_video = camara.get(cv.CAP_PROP_FPS)

        if fps_video <= 0:
            fps_video = 30.0

        espera_ms = max(1, round(1000 / fps_video))

    else:

        camara = cv.VideoCapture(0, cv.CAP_V4L2)

        if not camara.isOpened():
            camara = cv.VideoCapture(1, cv.CAP_V4L2)

        if not camara.isOpened():
            camara = cv.VideoCapture(0)

        espera_ms = 1

    if not camara.isOpened():

        origen = RUTA_VIDEO if RUTA_VIDEO else "la cámara"

        raise RuntimeError(
            f"No se pudo abrir {origen}. "
            "Verifica la ruta o conexión."
        )

    print("Fuente lista. Presiona 'q' para salir.")

    # ----------------------------------------------
    # Parámetros de detección
    # ----------------------------------------------

    UMBRAL_CAMBIO = 10.0

    INTERVALO_SEGUNDOS = 3

    medias_anteriores = None

    frame_referencia = None

    tiempo_ultima_actualizacion = (
        0.0 if RUTA_VIDEO else time.monotonic()
    )

    numero_fotograma = 0

    # ----------------------------------------------
    # Crear los ocho procesos
    # ----------------------------------------------

    with multiprocessing.Pool(processes=8) as pool:

        try:

            while True:

                ret, frame = camara.read()

                if not ret:
                    break

                numero_fotograma += 1

                if RUTA_VIDEO:

                    tiempo_actual = (
                        numero_fotograma / fps_video
                    )

                else:

                    tiempo_actual = time.monotonic()

                # ----------------------------------
                # Convertir el fotograma a escala de grises
                # ----------------------------------

                gris = cv.cvtColor(
                    frame,
                    cv.COLOR_BGR2GRAY
                )

                # ----------------------------------
                # Inicializar el fotograma de referencia
                # ----------------------------------

                if frame_referencia is None:

                    frame_referencia = gris.copy()

                # ----------------------------------
                # Dividir ambas imágenes en cuadrantes
                # ----------------------------------

                fragmentos_actuales, coordenadas = (
                    obtener_cuadrantes(gris)
                )

                fragmentos_referencia, _ = (
                    obtener_cuadrantes(frame_referencia)
                )

                # ----------------------------------
                # Preparar las tareas para los procesos
                # ----------------------------------

                tareas = []

                for i in range(len(fragmentos_actuales)):

                    tarea = (
                        fragmentos_actuales[i],
                        fragmentos_referencia[i]
                    )

                    tareas.append(tarea)

                # ----------------------------------
                # EJECUCIÓN PARALELA CON POOL.MAP
                # ----------------------------------

                resultados = pool.map(
                    procesar_cuadrante,
                    tareas
                )

                # ----------------------------------
                # Separar medias y contornos
                # ----------------------------------

                medias_actuales = []
                contornos_cuadrantes = []

                for media, contornos in resultados:

                    medias_actuales.append(media)

                    contornos_cuadrantes.append(contornos)

                medias_actuales = np.array(
                    medias_actuales
                )

                # ----------------------------------
                # Inicializar las medias de referencia
                # ----------------------------------

                if medias_anteriores is None:

                    medias_anteriores = (
                        medias_actuales.copy()
                    )

                # ----------------------------------
                # Comparar las medias de intensidad
                # ----------------------------------

                diferencias = np.abs(
                    medias_actuales - medias_anteriores
                )

                # ----------------------------------
                # Dibujar cuadrantes y contornos
                # ----------------------------------

                for i in range(len(coordenadas)):

                    x1, y1, x2, y2 = coordenadas[i]

                    dif = diferencias[i]

                    # Cuadrícula de referencia
                    cv.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (60, 60, 60),
                        1
                    )

                    # ----------------------------------
                    # Mostrar la diferencia de intensidad
                    # ----------------------------------

                    if dif > UMBRAL_CAMBIO:

                        cv.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            2
                        )

                        texto = f"Dif: {dif:.1f}"

                        cv.putText(
                            frame,
                            texto,
                            (x1 + 10, y1 + 30),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )

                    # ----------------------------------
                    # Dibujar los contornos detectados
                    # ----------------------------------

                    for contorno in contornos_cuadrantes[i]:

                        # Los contornos están expresados
                        # respecto al origen del cuadrante.
                        # Se trasladan a las coordenadas
                        # originales del fotograma.

                        contorno_global = contorno.copy()

                        contorno_global[:, 0, 0] += x1
                        contorno_global[:, 0, 1] += y1

                        # Dibujar el contorno del objeto
                        cv.drawContours(
                            frame,
                            [contorno_global],
                            -1,
                            (0, 255, 0),
                            2
                        )

                        # Calcular un rectángulo delimitador
                        rx, ry, rw, rh = cv.boundingRect(
                            contorno_global
                        )

                        # Dibujar el rectángulo del objeto
                        cv.rectangle(
                            frame,
                            (rx, ry),
                            (rx + rw, ry + rh),
                            (255, 0, 0),
                            2
                        )

                # ----------------------------------
                # Actualizar la referencia cada 1.5 s
                # ----------------------------------

                if (
                    tiempo_actual - tiempo_ultima_actualizacion
                ) >= INTERVALO_SEGUNDOS:

                    frame_referencia = gris.copy()

                    medias_anteriores = (
                        medias_actuales.copy()
                    )

                    tiempo_ultima_actualizacion = (
                        tiempo_actual
                    )

                # ----------------------------------
                # Mostrar el resultado
                # ----------------------------------

                cv.imshow(
                    "Deteccion Paralela de Cambios y Contornos",
                    frame
                )

                if cv.waitKey(espera_ms) & 0xFF == ord('q'):
                    break

        finally:

            camara.release()

            cv.destroyAllWindows()


# ==================================================
# 4. PUNTO DE ENTRADA
# ==================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()