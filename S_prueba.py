import cv2 as cv
import numpy as np
import time
import multiprocessing
import os
import csv


# ==========================================================
# 1. CONFIGURACIÓN
# ==========================================================

CARPETA_OBJETOS = "objetos_detectados"

# Crear la carpeta si no existe
os.makedirs(CARPETA_OBJETOS, exist_ok=True)

NUM_PROCESOS = 8

UMBRAL_CAMBIO = 10

UMBRAL_MEDIA = 5.0

AREA_MINIMA = 100

INTERVALO_REFERENCIA = 3.0

DISTANCIA_MAXIMA = 70

MAX_FOTOGRAMAS_PERDIDOS = 10

# Poner True para comparar los tiempos, False para ver el seguimiento
COMPARAR_TIEMPOS = False
VIDEO_PRUEBA = "video.mp4"
FRAMES_PRUEBA = 90
REPETICIONES = 3
PROCESOS_PRUEBA = [1, 2, 4, 8]


# ==========================================================
# 2. PROCESAMIENTO PARALELO DE CADA CUADRANTE
# ==========================================================

def procesar_cuadrante(argumentos):
    """
    Calcula la media de intensidad y detecta contornos
    de regiones que cambiaron respecto a la referencia.

    Esta función se ejecuta en los procesos del Pool.
    """

    fragmento_actual, fragmento_referencia = argumentos

    # Calcular la media de intensidad
    media = float(np.mean(fragmento_actual))

    # Comparar el fragmento actual con el de referencia
    diferencia = cv.absdiff(
        fragmento_actual,
        fragmento_referencia
    )

    # Crear una máscara binaria
    _, mascara = cv.threshold(
        diferencia,
        UMBRAL_CAMBIO,
        255,
        cv.THRESH_BINARY
    )

    # Eliminar ruido pequeño
    kernel = np.ones((3, 3), np.uint8)

    mascara = cv.morphologyEx(
        mascara,
        cv.MORPH_OPEN,
        kernel
    )

    # Conectar regiones próximas
    mascara = cv.dilate(
        mascara,
        kernel,
        iterations=2
    )

    # Encontrar contornos
    contornos, _ = cv.findContours(
        mascara,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    contornos_validos = []

    for contorno in contornos:

        area = cv.contourArea(contorno)

        if area >= AREA_MINIMA:
            contornos_validos.append(contorno)

    return media, contornos_validos


# ==========================================================
# 3. DIVIDIR LA IMAGEN EN OCHO CUADRANTES
# ==========================================================

def obtener_cuadrantes(frame):

    alto, ancho = frame.shape[:2]

    h_m = alto // 2

    w_c = ancho // 4

    coordenadas = [
        # Fila superior
        (0, 0, w_c, h_m),
        (w_c, 0, 2 * w_c, h_m),
        (2 * w_c, 0, 3 * w_c, h_m),
        (3 * w_c, 0, ancho, h_m),

        # Fila inferior
        (0, h_m, w_c, alto),
        (w_c, h_m, 2 * w_c, alto),
        (2 * w_c, h_m, 3 * w_c, alto),
        (3 * w_c, h_m, ancho, alto)
    ]

    fragmentos = []

    for x1, y1, x2, y2 in coordenadas:

        fragmento = frame[y1:y2, x1:x2]

        fragmentos.append(fragmento)

    return fragmentos, coordenadas


# ==========================================================
# 4. CALCULAR EL CENTRO DE UN RECTÁNGULO
# ==========================================================

def calcular_centro(caja):

    x, y, ancho, alto = caja

    cx = x + ancho // 2

    cy = y + alto // 2

    return (cx, cy)


# ==========================================================
# 5. ASOCIAR DETECCIONES CON OBJETOS EXISTENTES
# ==========================================================

def actualizar_seguimiento(
    detecciones,
    objetos,
    siguiente_id,
    frame
):
    """
    Asocia las nuevas detecciones con los objetos existentes.

    Si una detección no coincide con ningún objeto conocido,
    crea un ID y guarda un recorte de la imagen.

    Devuelve el diccionario actualizado y el siguiente ID.
    """

    # Marcar los objetos como no detectados inicialmente
    for objeto_id in objetos:

        objetos[objeto_id]["detectado"] = False

    # Registrar las asociaciones realizadas en este fotograma
    ids_utilizados = set()

    # Procesar las detecciones
    for caja in detecciones:

        centro = calcular_centro(caja)

        mejor_id = None

        mejor_distancia = DISTANCIA_MAXIMA

        # Buscar el objeto conocido más cercano
        for objeto_id, objeto in objetos.items():

            if objeto_id in ids_utilizados:
                continue

            # Solo comparar objetos que aún pueden recuperarse
            if objeto["perdidos"] > MAX_FOTOGRAMAS_PERDIDOS:
                continue

            centro_anterior = objeto["centro"]

            distancia = np.sqrt(
                (centro[0] - centro_anterior[0]) ** 2
                +
                (centro[1] - centro_anterior[1]) ** 2
            )

            if distancia < mejor_distancia:

                mejor_distancia = distancia

                mejor_id = objeto_id

        # --------------------------------------------------
        # CASO 1: SE ENCONTRÓ UN OBJETO CONOCIDO
        # --------------------------------------------------

        if mejor_id is not None:

            objeto = objetos[mejor_id]

            objeto["caja"] = caja

            objeto["centro"] = centro

            objeto["perdidos"] = 0

            objeto["detectado"] = True

            objeto["trayectoria"].append(centro)

            # Limitar el historial de posiciones
            if len(objeto["trayectoria"]) > 50:

                objeto["trayectoria"] = (
                    objeto["trayectoria"][-50:]
                )

            ids_utilizados.add(mejor_id)

        # --------------------------------------------------
        # CASO 2: SE DETECTÓ UN OBJETO NUEVO
        # --------------------------------------------------

        else:

            objeto_id = siguiente_id

            siguiente_id += 1

            x, y, ancho, alto = caja

            alto_frame, ancho_frame = frame.shape[:2]

            # Limitar las coordenadas a la imagen
            x1 = max(0, x)

            y1 = max(0, y)

            x2 = min(ancho_frame, x + ancho)

            y2 = min(alto_frame, y + alto)

            # Crear el recorte del objeto
            recorte = frame[y1:y2, x1:x2]

            # Guardar el primer recorte de este ID
            if recorte.size > 0:

                ruta = os.path.join(
                    CARPETA_OBJETOS,
                    f"objeto_ID_{objeto_id}.png"
                )

                cv.imwrite(ruta, recorte)

                print(
                    f"Nuevo objeto ID {objeto_id}: "
                    f"imagen guardada en {ruta}"
                )

            # Registrar el objeto
            objetos[objeto_id] = {
                "caja": caja,
                "centro": centro,
                "trayectoria": [centro],
                "perdidos": 0,
                "detectado": True
            }

            ids_utilizados.add(objeto_id)

    # ------------------------------------------------------
    # ACTUALIZAR OBJETOS QUE NO SE DETECTARON
    # ------------------------------------------------------

    for objeto_id in list(objetos.keys()):

        objeto = objetos[objeto_id]

        if not objeto["detectado"]:

            objeto["perdidos"] += 1

            # Eliminar el seguimiento si desaparece demasiado tiempo
            if objeto["perdidos"] > MAX_FOTOGRAMAS_PERDIDOS:

                del objetos[objeto_id]

    return objetos, siguiente_id


# ==========================================================
# 6. DIBUJAR OBJETOS Y TRAYECTORIAS
# ==========================================================

def dibujar_objetos(frame, objetos):

    for objeto_id, objeto in objetos.items():

        # Dibujar solo objetos detectados en este fotograma
        if not objeto["detectado"]:
            continue

        x, y, ancho, alto = objeto["caja"]

        cx, cy = objeto["centro"]

        # Dibujar el rectángulo delimitador
        cv.rectangle(
            frame,
            (x, y),
            (x + ancho, y + alto),
            (255, 0, 0),
            2
        )

        # Dibujar el centro
        cv.circle(
            frame,
            (cx, cy),
            4,
            (0, 255, 255),
            -1
        )

        # Mostrar el identificador
        cv.putText(
            frame,
            f"ID: {objeto_id}",
            (x, max(20, y - 10)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        # Dibujar la trayectoria
        trayectoria = objeto["trayectoria"]

        for i in range(1, len(trayectoria)):

            punto_anterior = trayectoria[i - 1]

            punto_actual = trayectoria[i]

            cv.line(
                frame,
                punto_anterior,
                punto_actual,
                (0, 255, 0),
                2
            )


# ==========================================================
# DATOS DEL MOVIMIENTO
# ==========================================================

def mostrar_movimiento(frame, objetos, posiciones, segundo, fotograma, datos):
    activos = 0

    for objeto_id, objeto in objetos.items():
        if not objeto["detectado"]:
            continue

        activos += 1
        cx, cy = objeto["centro"]
        velocidad = 0
        direccion = "Inicio"

        if objeto_id in posiciones:
            x_anterior, y_anterior, segundo_anterior = posiciones[objeto_id]
            dx = cx - x_anterior
            dy = cy - y_anterior
            tiempo = segundo - segundo_anterior

            if tiempo > 0:
                velocidad = np.sqrt(dx**2 + dy**2) / tiempo

            if dx == 0 and dy == 0:
                direccion = "Quieto"
            elif abs(dx) > abs(dy):
                direccion = "Derecha" if dx > 0 else "Izquierda"
            else:
                direccion = "Abajo" if dy > 0 else "Arriba"

        posiciones[objeto_id] = (cx, cy, segundo)

        x, y, ancho, alto = objeto["caja"]
        cv.putText(frame, f"{direccion}  {velocidad:.0f} px/s",
                   (max(5, x), min(frame.shape[0] - 10, y + alto + 18)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        datos.append([fotograma, objeto_id, cx, cy, round(velocidad, 2), direccion])

    # Borramos posiciones de objetos que ya salieron del seguimiento
    for objeto_id in list(posiciones):
        if objeto_id not in objetos:
            del posiciones[objeto_id]

    cv.putText(frame, f"Objetos detectados: {activos}", (10, 25),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)


# ==========================================================
# 7. PROGRAMA PRINCIPAL
# ==========================================================

def main():

    # None utiliza la cámara.
    # También puedes indicar la ruta de un archivo de video.
    RUTA_VIDEO = None

    # ------------------------------------------------------
    # Abrir cámara o video
    # ------------------------------------------------------

    if RUTA_VIDEO:

        camara = cv.VideoCapture(RUTA_VIDEO)

        fps_video = camara.get(cv.CAP_PROP_FPS)

        if fps_video <= 0:
            fps_video = 30.0

        espera_ms = max(
            1,
            round(1000 / fps_video)
        )

    else:

        camara = cv.VideoCapture(0, cv.CAP_V4L2)

        if not camara.isOpened():

            camara = cv.VideoCapture(1, cv.CAP_V4L2)

        if not camara.isOpened():

            camara = cv.VideoCapture(0)

        espera_ms = 1

    if not camara.isOpened():

        raise RuntimeError(
            "No se pudo abrir la cámara o el archivo de video."
        )

    print("Cámara iniciada. Presiona 'q' para salir.")

    # ------------------------------------------------------
    # Variables del seguimiento
    # ------------------------------------------------------

    objetos = {}

    siguiente_id = 1

    posiciones = {}
    datos_movimiento = []

    medias_anteriores = None

    frame_referencia = None

    numero_fotograma = 0

    video_salida = None

    tiempo_ultima_actualizacion = (
        0.0 if RUTA_VIDEO else time.monotonic()
    )

    # ------------------------------------------------------
    # Crear el Pool una sola vez
    # ------------------------------------------------------

    with multiprocessing.Pool(
        processes=NUM_PROCESOS
    ) as pool:

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

                # Convertir a escala de grises
                gris = cv.cvtColor(
                    frame,
                    cv.COLOR_BGR2GRAY
                )

                # Inicializar la referencia
                if frame_referencia is None:

                    frame_referencia = gris.copy()

                # Dividir imagen actual y referencia
                fragmentos_actuales, coordenadas = (
                    obtener_cuadrantes(gris)
                )

                fragmentos_referencia, _ = (
                    obtener_cuadrantes(frame_referencia)
                )

                # Crear las ocho tareas
                tareas = []

                for i in range(len(fragmentos_actuales)):

                    tareas.append(
                        (
                            fragmentos_actuales[i],
                            fragmentos_referencia[i]
                        )
                    )

                # ==========================================
                # PROCESAMIENTO PARALELO
                # ==========================================

                resultados = pool.map(
                    procesar_cuadrante,
                    tareas
                )

                # ==========================================
                # REUNIR CONTORNOS DE TODOS LOS CUADRANTES
                # ==========================================

                detecciones = []

                for i in range(len(resultados)):

                    media, contornos = resultados[i]

                    x1, y1, x2, y2 = coordenadas[i]

                    # Obtener la diferencia de medias
                    if medias_anteriores is not None:

                        diferencia_media = abs(
                            media - medias_anteriores[i]
                        )

                    else:

                        diferencia_media = 0.0

                    # Dibujar límites del cuadrante
                    cv.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (60, 60, 60),
                        1
                    )

                    # Mostrar diferencia de intensidad
                    if diferencia_media > UMBRAL_MEDIA:

                        cv.putText(
                            frame,
                            f"Dif: {diferencia_media:.1f}",
                            (x1 + 10, y1 + 30),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )

                    # Convertir contornos locales a globales
                    for contorno in contornos:

                        contorno_global = contorno.copy()

                        contorno_global[:, 0, 0] += x1

                        contorno_global[:, 0, 1] += y1

                        # Dibujar el contorno detectado
                        cv.drawContours(
                            frame,
                            [contorno_global],
                            -1,
                            (0, 255, 0),
                            2
                        )

                        # Obtener caja delimitadora global
                        caja = cv.boundingRect(
                            contorno_global
                        )

                        detecciones.append(caja)

                # ==========================================
                # ACTUALIZAR EL SEGUIMIENTO
                # ==========================================

                objetos, siguiente_id = actualizar_seguimiento(
                    detecciones,
                    objetos,
                    siguiente_id,
                    frame
                )

                # ==========================================
                # DIBUJAR IDs Y TRAYECTORIAS
                # ==========================================

                dibujar_objetos(
                    frame,
                    objetos
                )

                # Mostrar velocidad aparente y dirección de los objetos
                mostrar_movimiento(frame, objetos, posiciones,
                                   tiempo_actual, numero_fotograma, datos_movimiento)

                # ==========================================
                # ACTUALIZAR FOTOGRAMA DE REFERENCIA
                # ==========================================

                if (
                    tiempo_actual - tiempo_ultima_actualizacion
                ) >= INTERVALO_REFERENCIA:

                    frame_referencia = gris.copy()

                    medias_anteriores = np.array(
                        [
                            resultado[0]
                            for resultado in resultados
                        ]
                    )

                    tiempo_ultima_actualizacion = (
                        tiempo_actual
                    )

                # Inicializar medias si todavía no existen
                if medias_anteriores is None:

                    medias_anteriores = np.array(
                        [
                            resultado[0]
                            for resultado in resultados
                        ]
                    )

                # ==========================================
                # MOSTRAR RESULTADO
                # ==========================================

                # Guardar el video con las detecciones, si abrimos un archivo
                if RUTA_VIDEO:
                    if video_salida is None:
                        alto, ancho = frame.shape[:2]
                        video_salida = cv.VideoWriter(
                            "video_analizado.mp4", cv.VideoWriter_fourcc(*"mp4v"),
                            fps_video, (ancho, alto)
                        )
                    video_salida.write(frame)

                cv.imshow(
                    "Seguimiento Paralelo de Objetos",
                    frame
                )

                if cv.waitKey(espera_ms) & 0xFF == ord('q'):
                    break

        finally:

            if video_salida is not None:
                video_salida.release()
            camara.release()

            cv.destroyAllWindows()

    # Guardamos los movimientos para revisarlos después
    with open("movimientos.csv", "w", newline="", encoding="utf-8") as archivo:
        tabla = csv.writer(archivo)
        tabla.writerow(["fotograma", "id", "centro_x", "centro_y", "velocidad_px_s", "direccion"])
        tabla.writerows(datos_movimiento)

    print("Datos de movimiento guardados en movimientos.csv")
    if RUTA_VIDEO:
        print("Video con los movimientos guardado en video_analizado.mp4")
    print("\nSeguimiento finalizado.")

    print(
        f"Imágenes guardadas en: {CARPETA_OBJETOS}"
    )


# ==========================================================
# COMPARAR TIEMPOS SECUENCIALES Y PARALELOS
# ==========================================================

def cargar_video(ruta, limite):
    video = cv.VideoCapture(ruta)
    if not video.isOpened():
        raise RuntimeError("No se pudo abrir el video: " + ruta)

    fps = video.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frames = []
    while len(frames) < limite:
        ok, imagen = video.read()
        if not ok:
            break
        frames.append(cv.cvtColor(imagen, cv.COLOR_BGR2GRAY))
    video.release()

    if len(frames) < 2:
        raise ValueError("El video necesita al menos dos fotogramas.")

    # Preparamos las mismas ocho regiones para todas las pruebas.
    referencia = frames[0].copy()
    ultima_actualizacion = 0
    tareas_video = []

    for numero, gris in enumerate(frames, start=1):
        actuales, _ = obtener_cuadrantes(gris)
        anteriores, _ = obtener_cuadrantes(referencia)
        tareas_video.append(list(zip(actuales, anteriores)))

        tiempo_video = numero / fps
        if tiempo_video - ultima_actualizacion >= INTERVALO_REFERENCIA:
            referencia = gris.copy()
            ultima_actualizacion = tiempo_video

    return tareas_video


def limitar_hilos():
    # Evitamos que OpenCV cree más hilos dentro de cada proceso.
    cv.setNumThreads(1)


def medir(tareas_video, procesos, repeticiones):
    tiempos = []
    primeras_detecciones = []

    for repeticion in range(1, repeticiones + 1):
        pool = None
        if procesos > 0:
            pool = multiprocessing.Pool(procesos, initializer=limitar_hilos)

        try:
            # Una prueba corta de calentamiento, sin contar su tiempo.
            if pool:
                pool.map(procesar_cuadrante, tareas_video[0])
            else:
                [procesar_cuadrante(t) for t in tareas_video[0]]

            for numero, tareas in enumerate(tareas_video, start=1):
                inicio = time.perf_counter()
                if pool:
                    resultados = pool.map(procesar_cuadrante, tareas)
                else:
                    resultados = [procesar_cuadrante(t) for t in tareas]
                ms = (time.perf_counter() - inicio) * 1000

                tiempos.append({
                    "procesos": procesos,
                    "repeticion": repeticion,
                    "fotograma": numero,
                    "tiempo_ms": round(ms, 4)
                })

                # Guardamos una muestra para revisar que ambos modos
                # encuentren las mismas regiones de movimiento.
                if repeticion == 1:
                    cajas = [sorted(cv.boundingRect(c) for c in contornos)
                             for _, contornos in resultados]
                    primeras_detecciones.append(cajas)
        finally:
            if pool:
                pool.close()
                pool.join()

    promedio = sum(d["tiempo_ms"] for d in tiempos) / len(tiempos)
    return tiempos, promedio, primeras_detecciones


def guardar_tabla(archivo, filas):
    with open(archivo, "w", newline="", encoding="utf-8") as f:
        tabla = csv.DictWriter(f, fieldnames=filas[0].keys())
        tabla.writeheader()
        tabla.writerows(filas)


def guardar_graficas(resumen, carpeta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nombres = [fila["modo"] for fila in resumen]

    plt.figure(figsize=(8, 4))
    plt.bar(nombres, [fila["tiempo_ms"] for fila in resumen])
    plt.ylabel("Tiempo promedio (ms por fotograma)")
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, "tiempos.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(nombres, [fila["speedup"] for fila in resumen], marker="o")
    plt.axhline(1, linestyle="--")
    plt.ylabel("Speedup")
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, "speedup.png"))
    plt.close()


def comparar():
    # Misma imagen dividida en 8 regiones en todos los experimentos
    limitar_hilos()
    tareas = cargar_video(VIDEO_PRUEBA, FRAMES_PRUEBA)
    carpeta = "resultados_rendimiento"
    os.makedirs(carpeta, exist_ok=True)

    _, base, detecciones_base = medir(tareas, 0, REPETICIONES)
    resumen = [{"modo": "Secuencial", "procesos": 0,
                "tiempo_ms": round(base, 4), "speedup": 1,
                "fps": round(1000 / base, 2) if base > 0 else 0,
                "eficiencia_pct": 0}]

    for procesos in PROCESOS_PRUEBA:
        _, tiempo, detecciones = medir(tareas, procesos, REPETICIONES)
        if detecciones != detecciones_base:
            print("Las detecciones no coincidieron con", procesos, "procesos")
            continue

        speedup = base / tiempo if tiempo > 0 else 0
        eficiencia = speedup / procesos * 100
        resumen.append({"modo": str(procesos) + " procesos", "procesos": procesos,
                        "tiempo_ms": round(tiempo, 4), "speedup": round(speedup, 3),
                        "fps": round(1000 / tiempo, 2) if tiempo > 0 else 0,
                        "eficiencia_pct": round(eficiencia, 2)})
        print(procesos, "procesos:", round(tiempo, 2), "ms, Speedup:", round(speedup, 2))

    guardar_tabla(os.path.join(carpeta, "resumen.csv"), resumen)
    guardar_graficas(resumen, carpeta)
    print("Resultados en", carpeta)
    print("Solo medimos el procesamiento de las regiones, no todo el seguimiento.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if COMPARAR_TIEMPOS:
        comparar()
    else:
        main()
