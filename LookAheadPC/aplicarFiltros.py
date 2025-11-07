import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

# --- Configuración de múltiples filtros ---
FILTROS = {
    "cabello": [
        {
            "type": "cabello",
            "imagen": "Filtros/cabello_corto.png",
            "blend_mode": "multiply",
            "escala": 1.4,  # 1.8x el ancho de la cara
            "offset_y": -20  # 20 píxeles hacia arriba
        }
    ],
    "sombras": [
        {
            "type": "mask",
            "mask_path": "Filtros/Filtro2.png",
            "csv_path": "Filtros/eyes.csv",
            "blend_mode": "multiply"
        }
    ],
    "delineador": [
        {
            "type": "mask",
            "mask_path": "Filtros/Filtro1.png",
            "csv_path": "Filtros/eyes.csv",
            "blend_mode": "normal"
        }
    ],
    "labial": [
        {
            "type": "mask",
            "mask_path": "Filtros/Filtro3.png",
            "csv_path": "Filtros/lips.csv",
            "blend_mode": "soft_light"
        }
    ],
    "barba":[
        {
            "type": "mask",
            "mask_path": "Filtros/Filtro4.png",
            "csv_path": "Filtros/jaw.csv",
            "blend_mode": "multiply"
        }
    ],
    "rubor":[
        {
            "type": "mask",
            "mask_path": "Filtros/Filtro4.png",
            "csv_path": "Filtros/jaw.csv",
            "blend_mode": "multiply"
        }
    ]
}

mp_face_mesh = mp.solutions.face_mesh

class FiltroFacial:
    def __init__(self, name, mask_path, csv_path, blend_mode='normal'):
        self.name = name
        # Cargar imagen del filtro
        self.mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if self.mask is None:
            raise FileNotFoundError(f"No se pudo cargar: {mask_path}")
        
        # Cargar puntos definidos en el csv
        self.puntos = pd.read_csv(csv_path, header=None, names=["id", "x", "y"])
        self.puntos['id'] = self.puntos['id'].astype(int)
        
        # Preparar coordenadas
        self.src_pts = np.float32(self.puntos[['x', 'y']].values)
        
        # Obtener triangulación con delaunay
        self.triangulos = self._triangular_con_Delaunay()
        
        # Modo de fusión para la imagen: 'normal', 'multiply', 'screen', 'overlay'
        self.blend_mode = blend_mode
        
        # Pre-procesar máscara
        if self.mask.shape[2] == 4:
            self.mask_alpha = self.mask[:, :, 3] / 255.0
            self.mask_rgb = self.mask[:, :, :3]
        else:
            self.mask_alpha = np.ones((self.mask.shape[0], self.mask.shape[1]), dtype=np.float32)
            self.mask_rgb = self.mask
    
    def _aplicar_blend_mode(self, base, overlay, alpha):
        # Normalizar a rango [0, 1]
        base_norm = base.astype(np.float32) / 255.0
        overlay_norm = overlay.astype(np.float32) / 255.0
        
        if self.blend_mode == 'multiply':
            # Multiply: multiplica los colores (oscurece)
            blended = base_norm * overlay_norm
            
        elif self.blend_mode == 'screen':
            # Screen: lo contrario de multiply (aclara)
            blended = 1 - (1 - base_norm) * (1 - overlay_norm)
            
        elif self.blend_mode == 'overlay':
            # Overlay: combina multiply y screen
            mask = base_norm < 0.5
            blended = np.where(mask, 
                              2 * base_norm * overlay_norm,
                              1 - 2 * (1 - base_norm) * (1 - overlay_norm))
            
        elif self.blend_mode == 'soft_light':
            # Soft Light: versión suave de overlay
            blended = np.where(overlay_norm < 0.5,
                              base_norm - (1 - 2 * overlay_norm) * base_norm * (1 - base_norm),
                              base_norm + (2 * overlay_norm - 1) * (np.sqrt(base_norm) - base_norm))
            
        else:  # 'normal'
            # Normal: mezcla directa
            blended = overlay_norm
        
        # Aplicar alfa(transparencia) y devolver al rango [0, 255](rango que manejan los colores)
        alpha_3 = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        result = base_norm * (1 - alpha_3) + blended * alpha_3
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    
    def _triangular_con_Delaunay(self):
        # Algoritmo para juntar los puntos para hacer un plano con triangulos
        # (Esto es para que se realice mejor el "filtro" en los labios)
        return Delaunay(self.src_pts).simplices
    
    def _warp_triangle(self, dst, t_src, t_dst):
        r1 = cv2.boundingRect(np.float32([t_src]))
        r2 = cv2.boundingRect(np.float32([t_dst]))
        
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        
        # Verificar que las coordenadas no sean negativas o 0
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return
        
        # Ajustar coordenadas al frame destino
        y_end = min(y2 + h2, dst.shape[0])
        x_end = min(x2 + w2, dst.shape[1])
        
        if y2 >= dst.shape[0] or x2 >= dst.shape[1] or y_end <= y2 or x_end <= x2:
            return
        
        h_valid = y_end - y2
        w_valid = x_end - x2
        
        # Ajustar triángulos a la mesh (área de trabajo)
        t1_rect = np.float32([(t_src[i][0] - x1, t_src[i][1] - y1) for i in range(3)])
        t2_rect = np.float32([(t_dst[i][0] - x2, t_dst[i][1] - y2) for i in range(3)])
        
        # Crear máscara del triángulo con antialiasing
        mask_tri = np.zeros((h2, w2), dtype=np.float32)
        cv2.fillConvexPoly(mask_tri, np.int32(t2_rect), 1.0, lineType=cv2.LINE_AA)
        
        # Recorte y transformación
        img1_rect = self.mask_rgb[y1:y1+h1, x1:x1+w1]
        alpha1_rect = self.mask_alpha[y1:y1+h1, x1:x1+w1]
        
        if img1_rect.size == 0:
            return
        
        # Transformación afín con interpolación de alta calidad
        M = cv2.getAffineTransform(t1_rect, t2_rect)
        warped = cv2.warpAffine(img1_rect, M, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        warped_alpha = cv2.warpAffine(alpha1_rect, M, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # Aplicar máscara del triángulo con suavizado
        warped_alpha = warped_alpha * mask_tri
        
        # Aplica un GaussianBlur a la imagen puesta para que los bordes de una imagen se vean más suavizados
        kernel_size = 3 
        warped_alpha = cv2.GaussianBlur(warped_alpha, (kernel_size, kernel_size), 0)
        
        # Recortar a región válida
        warped = warped[:h_valid, :w_valid]
        warped_alpha = warped_alpha[:h_valid, :w_valid]
        
        # Asegurar que alfa esté en rango [0, 1]
        warped_alpha = np.clip(warped_alpha, 0, 1)
        
        # Aplicar modo de fusión en lugar de blending normal
        roi = dst[y2:y2+h_valid, x2:x2+w_valid]
        dst[y2:y2+h_valid, x2:x2+w_valid] = self._aplicar_blend_mode(roi, warped, warped_alpha)
    
    def get_name(self) -> str:
        return self.name
    
    def aplicar(self, frame, landmarks_px):
        # Obtener puntos destino
        dst_pts = landmarks_px[self.puntos['id'].values]
        
        # Envolver cada triángulo de la máscara/filtros con los landmarks de la mesh de mediapipe
        for tri in self.triangulos:
            t_src = self.src_pts[tri]
            t_dst = dst_pts[tri]
            
            self._warp_triangle(frame, t_src, t_dst)
        
        return frame

class FiltroCabello:
    def __init__(self, name, imagen_path, blend_mode='normal', escala=1.0, offset_y=0):
        self.name = name
        self.imagen = cv2.imread(imagen_path, cv2.IMREAD_UNCHANGED)
        if self.imagen is None:
            raise FileNotFoundError(f"No se pudo cargar: {imagen_path}")
        
        self.blend_mode = blend_mode
        self.escala = escala
        self.offset_y = offset_y
        
        # Separar canales
        if self.imagen.shape[2] == 4:
            self.cabello_rgb = self.imagen[:, :, :3]
            self.cabello_alpha = self.imagen[:, :, 3] / 255.0
        else:
            self.cabello_rgb = self.imagen
            self.cabello_alpha = np.ones((self.imagen.shape[0], self.imagen.shape[1]), dtype=np.float32)
    
    def _aplicar_blend_mode(self, base, overlay, alpha):
        # Normalizar a rango [0, 1]
        base_norm = base.astype(np.float32) / 255.0
        overlay_norm = overlay.astype(np.float32) / 255.0
        
        if self.blend_mode == 'multiply':
            # Multiply: multiplica los colores (oscurece)
            blended = base_norm * overlay_norm
            
        elif self.blend_mode == 'screen':
            # Screen: lo contrario de multiply (aclara)
            blended = 1 - (1 - base_norm) * (1 - overlay_norm)
            
        elif self.blend_mode == 'overlay':
            # Overlay: combina multiply y screen
            mask = base_norm < 0.5
            blended = np.where(mask, 
                              2 * base_norm * overlay_norm,
                              1 - 2 * (1 - base_norm) * (1 - overlay_norm))
            
        elif self.blend_mode == 'soft_light':
            # Soft Light: versión suave de overlay
            blended = np.where(overlay_norm < 0.5,
                              base_norm - (1 - 2 * overlay_norm) * base_norm * (1 - base_norm),
                              base_norm + (2 * overlay_norm - 1) * (np.sqrt(base_norm) - base_norm))
            
        else:  # 'normal'
            # Normal: mezcla directa
            blended = overlay_norm
        
        # Aplicar alfa(transparencia) y devolver al rango [0, 255](rango que manejan los colores)
        alpha_3 = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        result = base_norm * (1 - alpha_3) + blended * alpha_3
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)
  
    def get_name(self) -> str:
        return self.name
  
    def aplicar(self, frame, landmarks_px):
        h_frame, w_frame = frame.shape[:2]
        
        # Calcular puntos de referencia de la cara
        # Punto superior de la frente (landmark 10)
        frente_superior = landmarks_px[10]
        
        # Ancho de la cara (distancia entre sienes)
        # Landmarks 127 (sien izquierda) y 356 (sien derecha)
        sien_izq = landmarks_px[127]
        sien_der = landmarks_px[356]
        ancho_cara = np.linalg.norm(sien_der - sien_izq)
        
        # Calcular tamaño del cabello
        nuevo_ancho = int(ancho_cara * self.escala)
        h_cabello, w_cabello = self.cabello_rgb.shape[:2]
        ratio = h_cabello / w_cabello
        nuevo_alto = int(nuevo_ancho * ratio)
        
        # Redimensionar cabello
        cabello_resized = cv2.resize(self.cabello_rgb, (nuevo_ancho, nuevo_alto), 
                                     interpolation=cv2.INTER_LINEAR)
        alpha_resized = cv2.resize(self.cabello_alpha, (nuevo_ancho, nuevo_alto), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Calcular posición central
        centro_x = int(frente_superior[0])
        centro_y = int(frente_superior[1]) + self.offset_y
        
        frente_sup = landmarks_px[9]
        frente_inf = landmarks_px[10]
        ancho_frente = np.linalg.norm(frente_sup - frente_inf)
        
        
        # Posición superior izquierda del cabello
        x1 = centro_x - nuevo_ancho // 2
        y1 = centro_y - int(ancho_frente*2.8) # El cabello va hacia arriba desde la frente
        
        # Calcular límites válidos
        x1_valid = max(0, x1)
        y1_valid = max(0, y1)
        x2_valid = min(w_frame, x1 + nuevo_ancho)
        y2_valid = min(h_frame, y1 + nuevo_alto)
        
        # Calcular recortes correspondientes en el cabello
        cabello_x1 = x1_valid - x1
        cabello_y1 = y1_valid - y1
        cabello_x2 = cabello_x1 + (x2_valid - x1_valid)
        cabello_y2 = cabello_y1 + (y2_valid - y1_valid)
        
        # Verificar que hay área válida
        if x1_valid >= x2_valid or y1_valid >= y2_valid:
            return frame
        
        # Extraer región del cabello
        cabello_crop = cabello_resized[cabello_y1:cabello_y2, cabello_x1:cabello_x2]
        alpha_crop = alpha_resized[cabello_y1:cabello_y2, cabello_x1:cabello_x2]
        
        # Aplicar en el frame
        roi = frame[y1_valid:y2_valid, x1_valid:x2_valid]
        frame[y1_valid:y2_valid, x1_valid:x2_valid] = self._aplicar_blend_mode(
            roi, cabello_crop, alpha_crop
        )
        
        return frame
    

# --- Cargar todos los filtros ---
filtros_cabello = []
filtros_mascaras = []

def cargar_filtros(filtros):
    print("Cargando filtros ...") 
    for filtroName in filtros:
        for config in FILTROS.get(filtroName, []):
            if config["type"] == "mask":
                try:
                    filtro = FiltroFacial(
                        filtroName,
                        config["mask_path"],
                        config["csv_path"],
                        config.get("blend_mode", "normal")
                    )
                    filtros_mascaras.append(filtro)
                    print(f"✓ Filtro cargado: {config['mask_path']}")
                except Exception as e:
                    print(f"✗ Error: {e}")
            elif config["type"] == "cabello":
                try:
                    filtro = FiltroCabello(
                        filtroName,
                        config["imagen"],
                        config.get("blend_mode", "normal"),
                        config.get("escala", 1.5),
                        config.get("offset_y", 0)
                    )
                    filtros_cabello.append(filtro)
                    print(f"✓ Cabello cargado: {config['imagen']}")
                except Exception as e:
                    print(f"✗ Error: {e}")

def quitar_filtro(filtroName: str) -> bool:
    for config in FILTROS.get(filtroName, []):
        if config["type"] == "mask":
            for i in range(0, len(filtros_mascaras)):
                print(filtros_mascaras[i].get_name())
                if filtroName == filtros_mascaras[i].get_name():
                    filtros_mascaras.pop(i)
                    print(f"✓ Filtro quitado: {filtroName}")
                    return True
        if config["type"] == "cabello":
            for i in range(0, len(filtros_cabello)):
                print(filtros_cabello[i].get_name())
                if filtroName == filtros_cabello[i].get_name():
                    filtros_cabello.pop(i)
                    print(f"✓ Filtro quitado: {filtroName}")
                    return True
    print(f"✗ Filtro no quitado: {filtroName}")
    return False
                    
cargar_filtros(["barba", "sombras", "labios", "delineador", "rubor", "cabello"])

def aplicar_filtros(frame, landmarks_px):
    """Aplica todos los filtros en el orden correcto"""
    # 1. Facial 
    for filtro in filtros_mascaras:
        frame = filtro.aplicar(frame, landmarks_px)
    
    # 2. Cabello
    for filtro in filtros_cabello:
        frame = filtro.aplicar(frame, landmarks_px)
    
    return frame

# --- Iniciar cámara ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False
) as face_mesh:

    print("\nPresiona 'q' para salir")
    
    while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Procesar con MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            resultados = face_mesh.process(rgb)
            rgb.flags.writeable = True
            
            if resultados.multi_face_landmarks:
                face_landmarks = resultados.multi_face_landmarks[0]
                
                # Obtener landmarks
                landmarks_px = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
                
                # Aplicar cada filtro en secuencia
                frame = aplicar_filtros(frame, landmarks_px)
            
            cv2.imshow("Filtros con Cabello", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if cv2.waitKey(1) & 0xFF == ord('r'):
                quitar_filtro("barba")
            
# Cerrar el uso de cámara
cap.release()
cv2.destroyAllWindows()