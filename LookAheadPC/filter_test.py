import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

# --- Configuración de múltiples filtros ---
FILTROS = [
    {
        "mask_path": "Filtros/Filtro2.png",
        "csv_path": "Filtros/eyes.csv",
        "blend_mode": "multiply"  # Opciones: 'normal', 'multiply', 'screen', 'overlay', 'soft_light'
    },
    {
        "mask_path": "Filtros/Filtro1.png",
        "csv_path": "Filtros/eyes.csv",
        "blend_mode": "normal"  # Ideal para sombras y maquillaje
    },
    {
        "mask_path": "Filtros/Filtro3.png",
        "csv_path": "Filtros/lips.csv",
        "blend_mode": "soft_light"  # Ideal para iluminación sutil
    }
    # Agrega más filtros aquí...
]

mp_face_mesh = mp.solutions.face_mesh

class Filtro:
    """Clase para manejar cada filtro individualmente"""
    def __init__(self, mask_path, csv_path, blend_mode='normal'):
        # Cargar máscara
        self.mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if self.mask is None:
            raise FileNotFoundError(f"No se pudo cargar: {mask_path}")
        
        # Cargar puntos
        self.puntos = pd.read_csv(csv_path, header=None, names=["id", "x", "y"])
        self.puntos['id'] = self.puntos['id'].astype(int)
        
        # Preparar datos
        self.src_pts = np.float32(self.puntos[['x', 'y']].values)
        
        # Obtener triangulación con restricción de labios
        self.triangulos = self._triangular_con_restricciones()
        
        # Modo de fusión: 'normal', 'multiply', 'screen', 'overlay'
        self.blend_mode = blend_mode
        
        # Pre-procesar máscara
        if self.mask.shape[2] == 4:
            self.mask_alpha = self.mask[:, :, 3] / 255.0
            self.mask_rgb = self.mask[:, :, :3]
        else:
            self.mask_alpha = np.ones((self.mask.shape[0], self.mask.shape[1]), dtype=np.float32)
            self.mask_rgb = self.mask
    
    def _aplicar_blend_mode(self, base, overlay, alpha):
        """Aplica diferentes modos de fusión entre base y overlay"""
        # Normalizar a rango [0, 1]
        base_norm = base.astype(np.float32) / 255.0
        overlay_norm = overlay.astype(np.float32) / 255.0
        
        if self.blend_mode == 'multiply':
            # Multiply: multiplica los colores (oscurece)
            blended = base_norm * overlay_norm
            
        elif self.blend_mode == 'screen':
            # Screen: inverso de multiply (aclara)
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
        
        # Aplicar alfa y devolver al rango [0, 255]
        alpha_3 = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        result = base_norm * (1 - alpha_3) + blended * alpha_3
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    
    def _triangular_con_restricciones(self):
        """Triangulación que evita cruces en labios"""
        # Definir TODAS las regiones de labios según MediaPipe Face Mesh
        # Contorno exterior labio superior
        labio_superior_ext = set([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291])
        # Contorno interior labio superior
        labio_superior_int = set([78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308])
        # Contorno exterior labio inferior
        labio_inferior_ext = set([146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61])
        # Contorno interior labio inferior
        labio_inferior_int = set([95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78])
        
        # Unir todos los puntos de cada labio
        todos_labio_superior = labio_superior_ext | labio_superior_int
        todos_labio_inferior = labio_inferior_ext | labio_inferior_int
        
        # Puntos compartidos (comisuras) - estos pueden estar en ambos labios
        puntos_compartidos = {78, 308, 61, 291}
        
        # Obtener IDs de puntos usados
        ids_usados = set(self.puntos['id'].values)
        
        # Ver si hay puntos de labios en este filtro
        puntos_lab_sup = (todos_labio_superior - puntos_compartidos) & ids_usados
        puntos_lab_inf = (todos_labio_inferior - puntos_compartidos) & ids_usados
        
        # Si no hay puntos de labios, usar Delaunay normal
        if not puntos_lab_sup and not puntos_lab_inf:
            return Delaunay(self.src_pts).simplices
        
        # Crear máscara para evitar triangulación entre labios
        triangulacion = Delaunay(self.src_pts)
        triangulos_filtrados = []
        
        for tri in triangulacion.simplices:
            # Obtener IDs originales de los 3 vértices del triángulo
            ids_triangulo = set([self.puntos.iloc[i]['id'] for i in tri])
            
            # Contar cuántos puntos tiene de cada labio (sin contar compartidos)
            puntos_solo_superior = ids_triangulo & (todos_labio_superior - puntos_compartidos)
            puntos_solo_inferior = ids_triangulo & (todos_labio_inferior - puntos_compartidos)
            
            # Rechazar el triángulo si tiene puntos EXCLUSIVOS de ambos labios
            # Esto permite triángulos con puntos compartidos (comisuras)
            if puntos_solo_superior and puntos_solo_inferior:
                continue  # Saltar este triángulo problemático
            
            triangulos_filtrados.append(tri)
        
        return np.array(triangulos_filtrados) if triangulos_filtrados else triangulacion.simplices
    
    def aplicar(self, frame, landmarks_px):
        """Aplica este filtro al frame"""
        h, w = frame.shape[:2]
        
        # Obtener puntos destino
        dst_pts = landmarks_px[self.puntos['id'].values]
        
        # Aplicar cada triángulo
        for tri in self.triangulos:
            t_src = self.src_pts[tri]
            t_dst = dst_pts[tri]
            
            self._warp_triangle(frame, t_src, t_dst)
        
        return frame
    
    def _warp_triangle(self, dst, t_src, t_dst):
        """Warpea un triángulo individual con suavizado de bordes"""
        r1 = cv2.boundingRect(np.float32([t_src]))
        r2 = cv2.boundingRect(np.float32([t_dst]))
        
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        
        # Verificar límites
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return
        
        # Ajustar coordenadas al frame destino
        y_end = min(y2 + h2, dst.shape[0])
        x_end = min(x2 + w2, dst.shape[1])
        
        if y2 >= dst.shape[0] or x2 >= dst.shape[1] or y_end <= y2 or x_end <= x2:
            return
        
        h_valid = y_end - y2
        w_valid = x_end - x2
        
        # Ajustar triángulos a bounding boxes
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
        
        # Suavizar los bordes del alfa para eliminar aristas visibles
        kernel_size = 3  # Aumenta a 5 o 7 para más suavizado
        warped_alpha = cv2.GaussianBlur(warped_alpha, (kernel_size, kernel_size), 0)
        
        # Recortar a región válida
        warped = warped[:h_valid, :w_valid]
        warped_alpha = warped_alpha[:h_valid, :w_valid]
        
        # Asegurar que alfa esté en rango [0, 1]
        warped_alpha = np.clip(warped_alpha, 0, 1)
        
        # Aplicar modo de fusión en lugar de blending normal
        roi = dst[y2:y2+h_valid, x2:x2+w_valid]
        dst[y2:y2+h_valid, x2:x2+w_valid] = self._aplicar_blend_mode(roi, warped, warped_alpha)

# --- Cargar todos los filtros ---
filtros_cargados = []
for config in FILTROS:
    try:
        blend_mode = config.get("blend_mode", "normal")
        filtro = Filtro(config["mask_path"], config["csv_path"], blend_mode)
        filtros_cargados.append(filtro)
        print(f"✓ Filtro cargado: {config['mask_path']} (modo: {blend_mode})")
    except Exception as e:
        print(f"✗ Error cargando {config['mask_path']}: {e}")

if not filtros_cargados:
    print("No se cargaron filtros. Verifica las rutas.")
    exit()

def procesar_imagen(img_paths):
    for img_path in img_paths:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=True
        )
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        # Procesar con MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        resultados = face_mesh.process(rgb)
        rgb.flags.writeable = True
        
        if resultados.multi_face_landmarks:
            face_landmarks = resultados.multi_face_landmarks[0]
            
            # Obtener landmarks una sola vez
            landmarks_px = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
            
            # Aplicar cada filtro en secuencia
            for filtro in filtros_cargados:
                img = filtro.aplicar(img, landmarks_px)

        cv2.imwrite(f"tmp/filtro_{img_path[:-4]}.png", img)
    
procesar_imagen(["a.jpg","b.jpg"])