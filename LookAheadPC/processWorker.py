"""
Worker PyQt5 que integra la lógica avanzada de filtros (FiltroFacial, FiltroCabello,
triangulación Delaunay y blend modes) con la interfaz de Qt.
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from scipy.spatial import Delaunay

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage

# --- Ruta base del proyecto ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Configuración de múltiples filtros ---
FILTROS = {
    "corte": [
        {
            "type": "cabello",
            "imagen": os.path.join(BASE_DIR, "Filtros", "filtro_cabello_corto1.png"),
            "blend_mode": "multiply",
            "escala": 1.4,  # 1.4x el ancho de la cara
            "offset_y": -20  # 20 píxeles hacia arriba
        },
        {
            "type": "cabello",
            "imagen": os.path.join(BASE_DIR, "Filtros", "filtro_cabello_largo1.png"),
            "blend_mode": "multiply",
            "escala": 1.4,  # 1.4x el ancho de la cara
            "offset_y": -20  # 20 píxeles hacia arriba
        }
    ],
    "delineador": [
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_delineador1.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_eyes.csv"),
            "blend_mode": "normal"
        },
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_delineador2.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_eyes.csv"),
            "blend_mode": "normal"
        },
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_delineador3.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_eyes.csv"),
            "blend_mode": "normal"
        },
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_delineador4.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_eyes.csv"),
            "blend_mode": "normal"
        },
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_delineador5.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_eyes.csv"),
            "blend_mode": "normal"
        }
    ],
    "labios": [
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_labial1.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_lips.csv"),
            "blend_mode": "soft_light"
        },
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_labial2.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_lips.csv"),
            "blend_mode": "soft_light"
        },
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_labial3.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_lips.csv"),
            "blend_mode": "soft_light"
        },
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_labial4.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_lips.csv"),
            "blend_mode": "soft_light"
        },
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_labial5.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_lips.csv"),
            "blend_mode": "soft_light"
        }
    ],
    "sombras": [
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_sombra1.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_eyes.csv"),
            "blend_mode": "multiply"
        },
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_sombra2.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_eyes.csv"),
            "blend_mode": "multiply"
        },
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_sombra3.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_eyes.csv"),
            "blend_mode": "multiply"
        },
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_sombra4.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_eyes.csv"),
            "blend_mode": "multiply"
        },
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_sombra5.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_eyes.csv"),
            "blend_mode": "multiply"
        }
    ],
    "rubor":[
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_rubor1.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_cheeks.csv"),
            "blend_mode": "multiply"
        },
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_rubor2.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_cheeks.csv"),
            "blend_mode": "multiply"
        }
    ],
    "vello": [
        {
            "type": "mask",
            "mask_path": os.path.join(BASE_DIR, "Filtros", "filtro_barba1.png"),
            "csv_path": os.path.join(BASE_DIR, "Filtros", "points_jaw.csv"),
            "blend_mode": "multiply"
        }
    ]
}

filtros_por_categoria = {}
# ---------------------- Clases de filtros (FiltroFacial, FiltroCabello) ----------------------

class FiltroFacial:
    """
    Filtro basado en máscara con puntos en CSV y triangulación Delaunay.
    CSV: filas [id, x, y] con coordenadas relativas a la imagen de la máscara.
    """
    def __init__(self, name, mask_path, csv_path, blend_mode='normal'):
        self.name = name
        self.mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if self.mask is None:
            raise FileNotFoundError(f"No se pudo cargar: {mask_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No se pudo cargar CSV: {csv_path}")
        self.puntos = pd.read_csv(csv_path, header=None, names=["id", "x", "y"])
        self.puntos['id'] = self.puntos['id'].astype(int)
        self.src_pts = np.float32(self.puntos[['x', 'y']].values)
        
        # triangulación Delaunay (indices)
        if len(self.src_pts) >= 3:
            self.triangulos = Delaunay(self.src_pts).simplices
        else:
            self.triangulos = np.array([], dtype=np.int32)

        self.blend_mode = blend_mode

        if self.mask.shape[2] == 4:
            self.mask_alpha = (self.mask[:, :, 3] / 255.0).astype(np.float32)
            self.mask_rgb = self.mask[:, :, :3].astype(np.uint8)
        else:
            self.mask_alpha = np.ones((self.mask.shape[0], self.mask.shape[1]), dtype=np.float32)
            self.mask_rgb = self.mask.astype(np.uint8)

    def _aplicar_blend_mode(self, base, overlay, alpha):
        base_norm = base.astype(np.float32) / 255.0
        overlay_norm = overlay.astype(np.float32) / 255.0

        if self.blend_mode == 'multiply':
            blended = base_norm * overlay_norm
        elif self.blend_mode == 'screen':
            blended = 1 - (1 - base_norm) * (1 - overlay_norm)
        elif self.blend_mode == 'overlay':
            mask = base_norm < 0.5
            blended = np.where(mask, 2 * base_norm * overlay_norm,
                               1 - 2 * (1 - base_norm) * (1 - overlay_norm))
        elif self.blend_mode == 'soft_light':
            blended = np.where(overlay_norm < 0.5,
                              base_norm - (1 - 2 * overlay_norm) * base_norm * (1 - base_norm),
                              base_norm + (2 * overlay_norm - 1) * (np.sqrt(base_norm) - base_norm))
        else:  # normal
            blended = overlay_norm

        alpha_3 = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        result = base_norm * (1 - alpha_3) + blended * alpha_3
        return np.clip(result * 255, 0, 255).astype(np.uint8)

    def _warp_triangle(self, dst, t_src, t_dst):
        r1 = cv2.boundingRect(np.float32([t_src]))
        r2 = cv2.boundingRect(np.float32([t_dst]))
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2

        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return

        # recortes
        img1_rect = self.mask_rgb[y1:y1+h1, x1:x1+w1]
        alpha1_rect = self.mask_alpha[y1:y1+h1, x1:x1+w1]

        if img1_rect.size == 0:
            return

        t1_rect = np.float32([(t_src[i][0] - x1, t_src[i][1] - y1) for i in range(3)])
        t2_rect = np.float32([(t_dst[i][0] - x2, t_dst[i][1] - y2) for i in range(3)])

        M = cv2.getAffineTransform(t1_rect, t2_rect)
        warped = cv2.warpAffine(img1_rect, M, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        warped_alpha = cv2.warpAffine(alpha1_rect, M, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask_tri = np.zeros((h2, w2), dtype=np.float32)
        cv2.fillConvexPoly(mask_tri, np.int32(t2_rect), 1.0, lineType=cv2.LINE_AA)
        warped_alpha = warped_alpha * mask_tri
        warped_alpha = cv2.GaussianBlur(warped_alpha, (3, 3), 0)

        y_end = min(y2 + h2, dst.shape[0])
        x_end = min(x2 + w2, dst.shape[1])
        h_valid = y_end - y2
        w_valid = x_end - x2
        if h_valid <= 0 or w_valid <= 0:
            return

        warped = warped[:h_valid, :w_valid]
        warped_alpha = warped_alpha[:h_valid, :w_valid]

        roi = dst[y2:y2+h_valid, x2:x2+w_valid]
        dst[y2:y2+h_valid, x2:x2+w_valid] = self._aplicar_blend_mode(roi, warped, warped_alpha)

    def get_name(self) -> str:
        return self.name

    def aplicar(self, frame, landmarks_px):
        # landmarks_px: array Nx2 (coordenadas absolutas en frame)
        if self.triangulos.size == 0:
            return frame

        # dst_pts se forma tomando la lista de ids en CSV y buscando en landmarks_px
        try:
            dst_pts = landmarks_px[self.puntos['id'].values]
        except Exception:
            # indices no disponibles
            return frame

        for tri in self.triangulos:
            t_src = self.src_pts[tri]
            t_dst = dst_pts[tri]
            self._warp_triangle(frame, t_src, t_dst)

        return frame

class FiltroCabello:
    """
    Filtro de cabello que posiciona una imagen RGBA encima de la cabeza,
    escalándola y posicionándola en base a landmarks (sienes / frente).
    """
    def __init__(self, name, imagen_path, blend_mode='normal', escala=1.0, offset_y=0):
        self.name = name
        self.imagen = cv2.imread(imagen_path, cv2.IMREAD_UNCHANGED)
        if self.imagen is None:
            raise FileNotFoundError(f"No se pudo cargar: {imagen_path}")

        self.blend_mode = blend_mode
        self.escala = escala
        self.offset_y = offset_y

        if self.imagen.shape[2] == 4:
            self.cabello_rgb = self.imagen[:, :, :3]
            self.cabello_alpha = (self.imagen[:, :, 3] / 255.0).astype(np.float32)
        else:
            self.cabello_rgb = self.imagen
            self.cabello_alpha = np.ones((self.imagen.shape[0], self.imagen.shape[1]), dtype=np.float32)

    def _aplicar_blend_mode(self, base, overlay, alpha):
        base_norm = base.astype(np.float32) / 255.0
        overlay_norm = overlay.astype(np.float32) / 255.0

        if self.blend_mode == 'multiply':
            blended = base_norm * overlay_norm
        elif self.blend_mode == 'screen':
            blended = 1 - (1 - base_norm) * (1 - overlay_norm)
        elif self.blend_mode == 'overlay':
            mask = base_norm < 0.5
            blended = np.where(mask, 2 * base_norm * overlay_norm,
                               1 - 2 * (1 - base_norm) * (1 - overlay_norm))
        elif self.blend_mode == 'soft_light':
            blended = np.where(overlay_norm < 0.5,
                              base_norm - (1 - 2 * overlay_norm) * base_norm * (1 - base_norm),
                              base_norm + (2 * overlay_norm - 1) * (np.sqrt(base_norm) - base_norm))
        else:
            blended = overlay_norm

        alpha_3 = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        result = base_norm * (1 - alpha_3) + blended * alpha_3
        return np.clip(result * 255, 0, 255).astype(np.uint8)

    def get_name(self) -> str:
        return self.name

    def aplicar(self, frame, landmarks_px):
        h_frame, w_frame = frame.shape[:2]

        # Uso de landmarks: 10, 9, 127, 356 (como ejemplo)
        try:
            frente_superior = landmarks_px[10]
            frente_sup = landmarks_px[9]
            frente_inf = landmarks_px[10]
            sien_izq = landmarks_px[127]
            sien_der = landmarks_px[356]
        except Exception:
            return frame

        ancho_cara = np.linalg.norm(sien_der - sien_izq)
        nuevo_ancho = max(1, int(ancho_cara * self.escala))
        h_cabello, w_cabello = self.cabello_rgb.shape[:2]
        ratio = h_cabello / w_cabello
        nuevo_alto = max(1, int(nuevo_ancho * ratio))

        cabello_resized = cv2.resize(self.cabello_rgb, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR)
        alpha_resized = cv2.resize(self.cabello_alpha, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR)

        centro_x = int(frente_superior[0])
        centro_y = int(frente_superior[1]) + self.offset_y

        ancho_frente = np.linalg.norm(frente_sup - frente_inf)

        x1 = centro_x - nuevo_ancho // 2
        y1 = centro_y - int(ancho_frente * 2.8)

        x1_valid = max(0, x1)
        y1_valid = max(0, y1)
        x2_valid = min(w_frame, x1 + nuevo_ancho)
        y2_valid = min(h_frame, y1 + nuevo_alto)

        cabello_x1 = x1_valid - x1
        cabello_y1 = y1_valid - y1
        cabello_x2 = cabello_x1 + (x2_valid - x1_valid)
        cabello_y2 = cabello_y1 + (y2_valid - y1_valid)

        if x1_valid >= x2_valid or y1_valid >= y2_valid:
            return frame

        cabello_crop = cabello_resized[cabello_y1:cabello_y2, cabello_x1:cabello_x2]
        alpha_crop = alpha_resized[cabello_y1:cabello_y2, cabello_x1:cabello_x2]

        roi = frame[y1_valid:y2_valid, x1_valid:x2_valid]
        frame[y1_valid:y2_valid, x1_valid:x2_valid] = self._aplicar_blend_mode(roi, cabello_crop, alpha_crop)
        return frame

# ---------------------- Funciones de carga/gestión de filtros ----------------------

# Listas globales que el worker utilizará
filtros_cabello = []
filtros_mascaras = []

def cargar_filtros(categorias):
    """Carga filtros de las categorías pasadas."""
    for filtroName in categorias:
        configs = FILTROS.get(filtroName, [])
        filtros_por_categoria[filtroName] = []  # Inicializa lista de nombres por categoría

        for config in configs:
            try:
                if config["type"] == "mask":
                    f = FiltroFacial(
                        filtroName,
                        config["mask_path"],
                        config["csv_path"],
                        config.get("blend_mode", "normal")
                    )
                    filtros_mascaras.append(f)
                    filtros_por_categoria[filtroName].append(f.get_name())
                    print(f"✓ Filtro máscara cargado: {f.get_name()}")

                elif config["type"] == "cabello":
                    f = FiltroCabello(
                        filtroName,
                        config["imagen"],
                        config.get("blend_mode", "normal"),
                        config.get("escala", 1.5),
                        config.get("offset_y", 0)
                    )
                    filtros_cabello.append(f)
                    filtros_por_categoria[filtroName].append(f.get_name())
                    print(f"✓ Filtro cabello cargado: {f.get_name()}")

            except Exception as e:
                print(f"✗ No se pudo cargar {filtroName}: {e}")

def quitar_filtro(filtroName: str) -> bool:
    """Quita la primera instancia que coincida por nombre en las listas globales."""
    for i in range(len(filtros_mascaras)-1, -1, -1):
        if filtros_mascaras[i].get_name() == filtroName:
            filtros_mascaras.pop(i)
            print(f"✓ Filtro máscara quitado: {filtroName}")
            return True
    for i in range(len(filtros_cabello)-1, -1, -1):
        if filtros_cabello[i].get_name() == filtroName:
            filtros_cabello.pop(i)
            print(f"✓ Filtro cabello quitado: {filtroName}")
            return True
    print(f"✗ No se encontró filtro para quitar: {filtroName}")
    return False

def aplicar_filtros(frame, landmarks_px):
    """Aplica filtros en orden: máscaras -> cabello."""
    for filtro in filtros_mascaras:
        try:
            frame = filtro.aplicar(frame, landmarks_px)
        except Exception:
            pass
    for filtro in filtros_cabello:
        try:
            frame = filtro.aplicar(frame, landmarks_px)
        except Exception:
            pass
    return frame

# ---------------------- Worker PyQt5 ----------------------

mp_face_mesh = mp.solutions.face_mesh

class FaceDetectionWorker(QObject):
    frame_ready = pyqtSignal(QImage)
    finished = pyqtSignal()
    filter_changed = pyqtSignal(str, str)

    def __init__(self, cam_index=0, width=640, height=480):
        super().__init__()
        self.running = False
        self.cam_index = cam_index
        self.width = width
        self.height = height

        # para manejo simple de categorías y selección por categoría
        self.loaded_categories = []
        self.filter_index_by_category = {}

    # API para UI
    def set_filter(self, category=None, index=0):
        """
        Carga la categoría si no está cargada.
        """
        if not category:
            return
        if category in self.loaded_categories:
            # ya cargada
            filtro_name = filtros_por_categoria.get(category, [""])[0]
            self.filter_changed.emit(category, filtro_name)
            return

        # cargar la categoría
        cargar_filtros([category])
        self.loaded_categories.append(category)
        # index inicial
        self.filter_index_by_category[category] = index
        
        filtro_name = filtros_por_categoria.get(category, [""])[0]
        self.filter_changed.emit(category, filtro_name)

    def remove_filter_category(self, category):
        """Quita todos los filtros pertenecientes a category"""
        # quitar tantas coincidencias encuentre
        removed = False
        while quitar_filtro(category):
            removed = True
        filtro_name = "Ninguno"
        if removed:
            try:
                self.loaded_categories.remove(category)
            except ValueError:
                pass
            if category in self.filter_index_by_category:
                del self.filter_index_by_category[category]
            self.filter_changed.emit(category, filtro_name)

    def change_filter_next(self, category):
        filtro_name = filtros_por_categoria.get(category, [""])[0]
        self.filter_changed.emit(category, filtro_name)

    def change_filter_prev(self, category):
        filtro_name = filtros_por_categoria.get(category, [""])[0]
        self.filter_changed.emit(category, filtro_name)

    # Bucle de detección
    def start_detection(self):
        self.running = True

        cap = cv2.VideoCapture(self.cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara")
            self.finished.emit()
            return

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        ) as face_mesh:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                # Procesar con MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                resultados = face_mesh.process(rgb)
                rgb.flags.writeable = True

                if resultados.multi_face_landmarks:
                    lm = resultados.multi_face_landmarks[0]
                    try:
                        landmarks_px = np.array([[p.x * w, p.y * h] for p in lm.landmark]).astype(np.int32)
                    except Exception:
                        landmarks_px = None
                    if landmarks_px is not None and len(landmarks_px) > 0:
                        try:
                            frame_out = aplicar_filtros(frame.copy(), landmarks_px)
                        except Exception:
                            frame_out = frame.copy()
                    else:
                        frame_out = frame.copy()
                else:
                    frame_out = frame.copy()
                
                rgb_image = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
                h2, w2, ch = rgb_image.shape
                bytes_per_line = ch * w2
                qt_image = QImage(rgb_image.data, w2, h2, bytes_per_line, QImage.Format_RGB888)
                self.frame_ready.emit(qt_image)

            cap.release()
            self.finished.emit()

    def stop_detection(self):
        self.running = False

# ---------------------- Módulo de prueba ----------------------
if __name__ == "__main__":
    # Prueba rápida en consola (sin Qt) para verificar que se cargan filtros y la cámara
    print("Prueba rápida de processWorker.py")
    try:
        cargar_filtros(list(FILTROS.keys()))
        print("Filtros cargados:", [f.get_name() for f in filtros_mascaras + filtros_cabello])
    except Exception as e:
        print("Error cargando filtros:", e)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara de prueba.")
    else:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            print("Presiona 'q' para salir")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                resultados = face_mesh.process(rgb)
                rgb.flags.writeable = True
                if resultados.multi_face_landmarks:
                    lm = resultados.multi_face_landmarks[0]
                    landmarks_px = np.array([[p.x * w, p.y * h] for p in lm.landmark]).astype(np.int32)
                    frame = aplicar_filtros(frame, landmarks_px)
                cv2.imshow("Prueba IntegracionPrueba", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
        
__all__ = [
    "FILTROS", "FaceDetectionWorker", "cargar_filtros",
    "quitar_filtro", "aplicar_filtros", "filtros_por_categoria"
]