import cv2 as cv
import numpy as np
import multiprocessing as mp
import os
import time


# ==========================================================
# 1. CONFIGURACIÓN
# ==========================================================

CARPETA_OBJETOS = "objetos_detectados"
os.makedirs(CARPETA_OBJETOS, exist_ok=True)

# Se limita el número de procesos para no crear más de los necesarios.
NUM_PROCESOS = min(8, mp.cpu_count())

UMBRAL_CAMBIO = 35
AREA_MINIMA = 450
ANCHO_MINIMO = 15
ALTO_MINIMO = 15

INTERVALO_REFERENCIA = 8.0
DISTANCIA_MAXIMA = 90
MAX_FOTOGRAMAS_PERDIDOS = 5
LONGITUD_TRAYECTORIA = 40

# Reducir ruido antes de comparar las imágenes.
TAMANO_SUAVIZADO = (5, 5)


# ==========================================================
# 2. DIVIDIR LA IMAGEN EN OCHO CUADRANTES
# ==========================================================

def obtener_cuadrantes(frame):
    alto, ancho = frame.shape[:2]
    mitad_alto = alto // 2
    cuarto_ancho = ancho // 4

    coordenadas = [
        (0, 0, cuarto_ancho, mitad_alto),
        (cuarto_ancho, 0, 2 * cuarto_ancho, mitad_alto),
        (2 * cuarto_ancho, 0, 3 * cuarto_ancho, mitad_alto),
        (3 * cuarto_ancho, 0, ancho, mitad_alto),
        (0, mitad_alto, cuarto_ancho, alto),
        (cuarto_ancho, mitad_alto, 2 * cuarto_ancho, alto),
        (2 * cuarto_ancho, mitad_alto, 3 * cuarto_ancho, alto),
        (3 * cuarto_ancho, mitad_alto, ancho, alto),
    ]

    fragmentos = []

    for x1, y1, x2, y2 in coordenadas:
        fragmentos.append(frame[y1:y2, x1:x2])

    return fragmentos, coordenadas


# ==========================================================
# 3. PROCESAMIENTO PARALELO DE CADA CUADRANTE
# ==========================================================

def procesar_cuadrante(argumentos):
    """
    Recibe la imagen actual y la referencia de un cuadrante.
    Devuelve la media de intensidad y cajas delimitadoras locales.
    Devolver cajas, en lugar de contornos completos, reduce
    la cantidad de información que se comunica entre procesos.
    """
    actual, referencia = argumentos

    if actual.size == 0 or referencia.size == 0:
        return 0.0, []

    media = float(np.mean(actual))

    diferencia = cv.absdiff(actual, referencia)
    _, mascara = cv.threshold(
        diferencia, UMBRAL_CAMBIO, 255, cv.THRESH_BINARY
    )

    kernel = np.ones((3, 3), np.uint8)
    mascara = cv.morphologyEx(mascara, cv.MORPH_OPEN, kernel)
    mascara = cv.morphologyEx(mascara, cv.MORPH_CLOSE, kernel)
    mascara = cv.dilate(mascara, kernel, iterations=1)

    contornos, _ = cv.findContours(
        mascara, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    cajas = []

    for contorno in contornos:
        area = cv.contourArea(contorno)

        if area < AREA_MINIMA:
            continue

        x, y, ancho, alto = cv.boundingRect(contorno)

        if ancho < ANCHO_MINIMO or alto < ALTO_MINIMO:
            continue

        area_rectangulo = ancho * alto
        if area_rectangulo == 0:
            continue

        extension = area / area_rectangulo
        if extension < 0.30:
            continue

        # Se devuelve una caja (x, y, ancho, alto), no el contorno.
        cajas.append((x, y, ancho, alto))

    return media, cajas


# ==========================================================
# 4. FUNCIONES AUXILIARES PARA SEGUIMIENTO
# ==========================================================

def calcular_centro(caja):
    x, y, ancho, alto = caja
    return x + ancho // 2, y + alto // 2


def guardar_recorte(frame, caja, objeto_id):
    alto_frame, ancho_frame = frame.shape[:2]
    x, y, ancho, alto = caja

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(ancho_frame, x + ancho)
    y2 = min(alto_frame, y + alto)

    if x2 <= x1 or y2 <= y1:
        return

    recorte = frame[y1:y2, x1:x2]

    if recorte.size == 0:
        return

    ruta = os.path.join(CARPETA_OBJETOS, f"objeto_ID_{objeto_id}.png")

    if cv.imwrite(ruta, recorte):
        print(f"Objeto nuevo ID {objeto_id}: guardado en {ruta}")


def actualizar_seguimiento(detecciones, objetos, siguiente_id, frame):
    """
    Asocia cada detección con el objeto no utilizado más cercano.
    Cada objeto solo puede asociarse a una detección por fotograma.
    """
    for objeto in objetos.values():
        objeto["detectado"] = False

    ids_utilizados = set()

    # Ordenar las detecciones por área aproximada: primero las mayores.
    detecciones = sorted(
        detecciones,
        key=lambda caja: caja[2] * caja[3],
        reverse=True
    )

    for caja in detecciones:
        centro = calcular_centro(caja)
        mejor_id = None
        mejor_distancia = DISTANCIA_MAXIMA

        for objeto_id, objeto in objetos.items():
            if objeto_id in ids_utilizados:
                continue

            if objeto["perdidos"] > MAX_FOTOGRAMAS_PERDIDOS:
                continue

            centro_anterior = objeto["centro"]
            dx = centro[0] - centro_anterior[0]
            dy = centro[1] - centro_anterior[1]
            distancia = (dx * dx + dy * dy) ** 0.5

            if distancia < mejor_distancia:
                mejor_distancia = distancia
                mejor_id = objeto_id

        if mejor_id is not None:
            objeto = objetos[mejor_id]
            objeto["caja"] = caja
            objeto["centro"] = centro
            objeto["perdidos"] = 0
            objeto["detectado"] = True
            objeto["trayectoria"].append(centro)

            if len(objeto["trayectoria"]) > LONGITUD_TRAYECTORIA:
                objeto["trayectoria"] = objeto["trayectoria"][-LONGITUD_TRAYECTORIA:]

            ids_utilizados.add(mejor_id)

        else:
            objeto_id = siguiente_id
            siguiente_id += 1

            objetos[objeto_id] = {
                "caja": caja,
                "centro": centro,
                "trayectoria": [centro],
                "perdidos": 0,
                "detectado": True,
            }

            ids_utilizados.add(objeto_id)
            guardar_recorte(frame, caja, objeto_id)

    # Incrementar el contador de pérdidas y eliminar objetos antiguos.
    ids_eliminar = []

    for objeto_id, objeto in objetos.items():
        if not objeto["detectado"]:
            objeto["perdidos"] += 1

            if objeto["perdidos"] > MAX_FOTOGRAMAS_PERDIDOS:
                ids_eliminar.append(objeto_id)

    for objeto_id in ids_eliminar:
        del objetos[objeto_id]

    return objetos, siguiente_id


# ==========================================================
# 5. DIBUJAR CAJAS, IDENTIFICADORES Y TRAYECTORIAS
# ==========================================================

def dibujar_objetos(frame, objetos):
    for objeto_id, objeto in objetos.items():
        if not objeto["detectado"]:
            continue

        x, y, ancho, alto = objeto["caja"]
        cx, cy = objeto["centro"]

        cv.rectangle(
            frame, (x, y), (x + ancho, y + alto), (255, 80, 0), 2
        )
        cv.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        cv.putText(
            frame,
            f"ID: {objeto_id}",
            (x, max(20, y - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 80, 0),
            2,
        )

        trayectoria = objeto["trayectoria"]

        for i in range(1, len(trayectoria)):
            cv.line(
                frame,
                trayectoria[i - 1],
                trayectoria[i],
                (0, 255, 0),
                2,
            )


# ==========================================================
# 6. PROGRAMA PRINCIPAL
# ==========================================================

def main():
    # Deja None para usar la cámara o escribe la ruta de un video.
    RUTA_VIDEO = None

    if RUTA_VIDEO is not None:
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
        raise RuntimeError("No se pudo abrir la cámara o el archivo de video.")

    # Limitar el búfer ayuda a evitar procesar imágenes demasiado antiguas.
    camara.set(cv.CAP_PROP_BUFFERSIZE, 1)

    print(f"Cámara iniciada. Procesos: {NUM_PROCESOS}. Presiona 'q' para salir.")

    objetos = {}
    siguiente_id = 1
    frame_referencia = None
    medias_anteriores = None
    tiempo_ultima_actualizacion = time.monotonic()
    tiempo_ultimo_fps = time.monotonic()
    contador_fps = 0
    fps_estimado = 0.0

    # Crear el Pool una sola vez, no en cada fotograma.
    with mp.Pool(processes=NUM_PROCESOS) as pool:
        try:
            while True:
                correcto, frame = camara.read()

                if not correcto:
                    break

                gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                gris = cv.GaussianBlur(gris, TAMANO_SUAVIZADO, 0)

                if frame_referencia is None:
                    frame_referencia = gris.copy()

                fragmentos_actuales, coordenadas = obtener_cuadrantes(gris)
                fragmentos_referencia, _ = obtener_cuadrantes(frame_referencia)

                tareas = []

                for i in range(len(fragmentos_actuales)):
                    tareas.append((
                        fragmentos_actuales[i],
                        fragmentos_referencia[i]
                    ))

                # Cada proceso analiza un cuadrante.
                resultados = pool.map(procesar_cuadrante, tareas, chunksize=1)

                detecciones = []

                for i in range(len(resultados)):
                    media, cajas_locales = resultados[i]
                    x1, y1, x2, y2 = coordenadas[i]

                    if medias_anteriores is not None:
                        diferencia_media = abs(media - medias_anteriores[i])

                        if diferencia_media > 25:
                            cv.putText(
                                frame,
                                f"Cambio: {diferencia_media:.1f}",
                                (x1 + 8, y1 + 25),
                                cv.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 0, 255),
                                2,
                            )

                    cv.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)

                    # Convertir las cajas locales a coordenadas globales.
                    for x, y, ancho, alto in cajas_locales:
                        caja_global = (x + x1, y + y1, ancho, alto)
                        detecciones.append(caja_global)

                objetos, siguiente_id = actualizar_seguimiento(
                    detecciones, objetos, siguiente_id, frame
                )

                dibujar_objetos(frame, objetos)

                # Actualizar la referencia en intervalos regulares.
                tiempo_actual = time.monotonic()

                if tiempo_actual - tiempo_ultima_actualizacion >= INTERVALO_REFERENCIA:
                    frame_referencia = gris.copy()
                    medias_anteriores = np.array(
                        [resultado[0] for resultado in resultados],
                        dtype=np.float32,
                    )
                    tiempo_ultima_actualizacion = tiempo_actual

                elif medias_anteriores is None:
                    medias_anteriores = np.array(
                        [resultado[0] for resultado in resultados],
                        dtype=np.float32,
                    )

                # Mostrar FPS aproximados para comprobar el rendimiento.
                contador_fps += 1
                if tiempo_actual - tiempo_ultimo_fps >= 1.0:
                    fps_estimado = contador_fps / (tiempo_actual - tiempo_ultimo_fps)
                    contador_fps = 0
                    tiempo_ultimo_fps = tiempo_actual

                cv.putText(
                    frame,
                    f"FPS: {fps_estimado:.1f} | Objetos: {sum(o['detectado'] for o in objetos.values())}",
                    (10, 25),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )

                cv.imshow("Seguimiento paralelo de objetos", frame)

                if cv.waitKey(espera_ms) & 0xFF == ord("q"):
                    break

        finally:
            camara.release()
            cv.destroyAllWindows()

    print("\nSeguimiento finalizado.")
    print(f"Imágenes guardadas en: {CARPETA_OBJETOS}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
