import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget
)
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QPixmap
import processWorker as face  # tu módulo con FaceDetectionWorker
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QSpacerItem, QSizePolicy
from processWorker import FILTROS
from processWorker import filtros_por_categoria
import os

class CameraView(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")

    def update_frame(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        # Escalar la imagen manteniendo la proporción, cubriendo todo el QLabel
        scaled = pixmap.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation
        )
        # Centrar y recortar los bordes que sobresalgan
        x_offset = max((scaled.width() - self.width()) // 2, 0)
        y_offset = max((scaled.height() - self.height()) // 2, 0)
        cropped = scaled.copy(x_offset, y_offset, self.width(), self.height())
        self.setPixmap(cropped)

# --- Template de vista ---
class BaseView(QWidget):
    def __init__(self, parent=None, title="", back_callback=None, filtro_name=None):
        super().__init__(parent)
        self.filtro_name = filtro_name

        # Cámara como fondo absoluto
        self.camera_label = CameraView(self)
        self.camera_label.setGeometry(0, 0, parent.width() if parent else 400, parent.height() if parent else 900)

        # Layout principal de la UI encima de la cámara
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)

        # --- Top bar ---
        top_bar = QHBoxLayout()

        # Botón atrás
        if back_callback:
            back_btn = QPushButton("←")
            back_btn.setFixedSize(40, 40)
            back_btn.clicked.connect(back_callback)
            top_bar.addWidget(back_btn)
        else:
            top_bar.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))

        # Spacer izquierdo para separar botón del título
        top_bar.addSpacerItem(QSpacerItem(10, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))

        # Título centrado en todo el espacio restante
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("""
            font-size: 16px;
            color: white;
            font-weight: bold;
            background-color: transparent;
        """)
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_bar.addWidget(title_lbl)

        # Spacer derecho para equilibrar el botón de atrás
        top_bar.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))
        self.layout.addLayout(top_bar)

        # --- Spacer que empuja todo hacia abajo ---
        self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # --- Layout inferior (botones) ---
        self.bottom_layout = QHBoxLayout()
        self.bottom_layout.setAlignment(Qt.AlignCenter)
        self.layout.addLayout(self.bottom_layout)

    def resizeEvent(self, event):
        # Hacer que la cámara siempre ocupe toda la vista
        self.camera_label.setGeometry(0, 0, self.width(), self.height())

    def update_camera_frame(self, qt_image):
        self.camera_label.update_frame(qt_image)

# --- Ventana principal ---
class MainWindow(QWidget):
    def go_home(self):
        self.stack.setCurrentWidget(self.view_home)

    def go_makeup(self):
        self.stack.setCurrentWidget(self.view_makeup)

    def go_hair(self):
        self.stack.setCurrentWidget(self.view_hair)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LookAhead - Filtros en tiempo real")
        self.setGeometry(300, 50, 400, 900)
        self.setStyleSheet("background-color: #111; color: white;")

        self.stack = QStackedWidget(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.stack)

        # --- Worker y thread (una sola vez) ---
        self.thread = QThread()
        self.worker = face.FaceDetectionWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start_detection)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.filter_changed.connect(self.on_filter_changed)
        self.thread.start()

        # --- Vistas con filtro asociado ---
        self.view_delineador = self.create_subview("Delineador", "delineador", self.go_makeup)
        self.view_labios = self.create_subview("Labios", "labios", self.go_makeup)
        self.view_rubor = self.create_subview("Rubor", "rubor", self.go_makeup)
        self.view_sombras = self.create_subview("Sombras", "sombras", self.go_makeup)
        self.view_corte = self.create_subview("Corte", "corte", self.go_hair)
        self.view_tinte = self.create_subview("Tinte", "tinte", self.go_hair)

        self.view_home = self.create_home_view()
        self.view_makeup = self.create_makeup_view()
        self.view_hair = self.create_hair_view()

        for v in [
            self.view_home, self.view_makeup, self.view_hair,
            self.view_delineador, self.view_sombras, self.view_rubor, self.view_labios,
            self.view_corte, self.view_tinte
        ]:
            self.stack.addWidget(v)

        self.stack.setCurrentWidget(self.view_home)

        # al cambiar vista, cambiar filtro
        self.stack.currentChanged.connect(self.on_view_changed)
        self.worker.set_filter(None)

    def on_frame_ready(self, qt_image):
        current = self.stack.currentWidget()
        if isinstance(current, BaseView):
            current.update_camera_frame(qt_image)

    def on_view_changed(self, index):
        view = self.stack.widget(index)
        filtro = getattr(view, "filtro_name", None)
        if filtro:
            filtro_index = self.worker.filter_index_by_category.get(filtro, -1)
            if filtro_index >= 0:
                # aplicar la categoría (si no está cargada, set_filter la cargará)
                self.worker.set_filter(filtro)
                print(f"[UI] set_filter -> {filtro} (idx {filtro_index})")
            else:
                # quitar la categoría para que no quede aplicada
                if hasattr(self.worker, "remove_filter_category"):
                    self.worker.remove_filter_category(filtro)
                else:
                    # fallback: marcar índice y confiar que aplicar_filtros ignore índices -1
                    self.worker.filter_index_by_category[filtro] = -1
                print(f"[UI] quitar filtro -> {filtro}")
        else:
            # sin filtro seleccionado en la vista actual: quitar cualquier categoría activa allí
            print("[UI] set_filter -> None")

    def on_filter_changed(self, categoria, nombre):
        print(f"[UI] filter_changed -> {categoria} ({nombre})")

        # Obtener la vista actual
        current_view = self.stack.currentWidget()

        # Verifica que la vista actual tenga el mismo filtro
        if hasattr(current_view, "filtro_name") and current_view.filtro_name == categoria:
            # Busca el QLabel principal del filtro en esa vista
            for i in range(current_view.layout.count()):
                widget = current_view.layout.itemAt(i).widget()
                if isinstance(widget, QLabel):
                    widget.setText(nombre if nombre else "Ninguno")
                    break

    # --- Creación de pantallas ---
    def create_home_view(self):
        view = BaseView(title="Inicio", filtro_name=None)
        btn_makeup = MainWindow.create_icon_button("icons/maquillaje.png", size=120, tooltip="Maquillaje")
        btn_hair = MainWindow.create_icon_button("icons/cabello.png", size=120, tooltip="Cabello")

        for b in (btn_makeup, btn_hair):
            b.setFixedWidth(120)
            b.setStyleSheet("background-color: #444; color: white; border-radius: 10px; font-size: 14px;")
        btn_makeup.clicked.connect(lambda: self.stack.setCurrentWidget(self.view_makeup))
        btn_hair.clicked.connect(lambda: self.stack.setCurrentWidget(self.view_hair))
        view.bottom_layout.addWidget(btn_makeup)
        view.bottom_layout.addWidget(btn_hair)
        return view
    
    # --- Helper para crear botones circulares con iconos ---
    @staticmethod
    def create_icon_button(icon_path, size=80, tooltip=None):
        btn = QPushButton()
        btn.setIcon(QIcon(icon_path))
        btn.setIconSize(QSize(size, size))
        btn.setFixedSize(size, size)
        btn.setStyleSheet(f"""
            QPushButton {{
                border: none;
                border-radius: {size//2}px;
                background-color: rgba(100,100,100,0.2);  /* gris muy transparente */
            }}
            QPushButton:pressed {{
                background-color: rgba(100,100,100,0.4);  /* un poco más oscuro al presionar */
            }}
        """)
        if tooltip:
            btn.setToolTip(tooltip)
        return btn

# --- Crear la vista de maquillaje ---
    def create_makeup_view(self):
        view = BaseView(title="Maquillaje", back_callback=self.go_home, filtro_name=None)
        
        # Lista de botones: (icono, vista destino, tooltip)
        buttons = [
            ("icons/delineador.png", self.view_delineador, "Delineador"),
            ("icons/sombras.png", self.view_sombras, "Sombras"),
            ("icons/rubor.png", self.view_rubor, "Rubor"),
            ("icons/labios.png", self.view_labios, "Labios")
        ]
        
        for icon_path, target, tooltip in buttons:
            btn = MainWindow.create_icon_button(icon_path, size=80, tooltip=tooltip)
            btn.clicked.connect(lambda _, v=target: self.stack.setCurrentWidget(v))
            view.bottom_layout.addWidget(btn)
        
        return view
    
    # --- Crear la vista de cabello ---
    def create_hair_view(self):
        view = BaseView(title="Cabello", back_callback=self.go_home, filtro_name=None)
        
        buttons = [
            ("icons/corte.png", self.view_corte, "Corte"),
            ("icons/tinte.png", self.view_tinte, "Tinte")
        ]
        
        for icon_path, target, tooltip in buttons:
            btn = MainWindow.create_icon_button(icon_path, size=80, tooltip=tooltip)
            btn.clicked.connect(lambda _, v=target: self.stack.setCurrentWidget(v))
            view.bottom_layout.addWidget(btn)
        
        return view

    def create_subview(self, title, filtro_name, back_callback):
        filters_list = FILTROS.get(filtro_name, [])
        nombres = ["Ninguno"] + filtros_por_categoria.get(filtro_name, [])
        items = ["Ninguno"]

        for f in filters_list:
            ruta = f.get("path") or f.get("imagen") or f.get("mask_path")
            if ruta:
                nombre = os.path.splitext(os.path.basename(ruta))[0]
                items.append(nombre)

        view = BaseView(title=title, back_callback=back_callback, filtro_name=filtro_name)
        index = 0

        # Texto principal (nombre del filtro actual)
        main_label = QLabel(nombres[index] if index < len(nombres) else "Ninguno")
        main_label.setAlignment(Qt.AlignCenter)
        main_label.setStyleSheet("""
            font-size: 22px;
            color: white;
            font-weight: bold;
            background-color: transparent;
        """)
        view.layout.addWidget(main_label)

        # Layout del carrusel (miniaturas)
        carousel_layout = QHBoxLayout()
        carousel_layout.setSpacing(15)
        view.layout.addLayout(carousel_layout)

        def update_carousel(emit_change=False):
            nonlocal index
            # Limpiar miniaturas previas
            for i in reversed(range(carousel_layout.count())):
                widget = carousel_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

            total = len(items)
            visible_indices = [(index - 1) % total, index, (index + 1) % total]

            for idx in visible_indices:
                btn = QPushButton()
                btn.setFixedSize(80, 80)
                btn.setCursor(Qt.PointingHandCursor)
                btn.setToolTip(nombres[idx] if idx < len(nombres) else items[idx])

                # Transparencia y selección
                if idx == index:
                    btn.setStyleSheet("""
                        QPushButton {
                            border: 2px solid #2d89ef;
                            border-radius: 10px;
                            background-color: rgba(255, 255, 255, 60);
                        }
                        QPushButton:hover {
                            background-color: rgba(255, 255, 255, 90);
                        }
                    """)
                else:
                    btn.setStyleSheet("""
                        QPushButton {
                            border: 2px solid #555;
                            border-radius: 10px;
                            background-color: rgba(255, 255, 255, 30);
                        }
                        QPushButton:hover {
                            background-color: rgba(255, 255, 255, 60);
                        }
                    """)

                # Miniatura
                if items[idx] != "Ninguno":
                    ruta_img = None
                    for f in FILTROS.get(filtro_name, []):
                        ruta_img = f.get("imagen") or f.get("mask_path")
                        if ruta_img and os.path.splitext(os.path.basename(ruta_img))[0] == items[idx]:
                            break
                    if ruta_img and os.path.exists(ruta_img):
                        pix = QPixmap(ruta_img)
                        btn.setIcon(QIcon(pix))
                        btn.setIconSize(QSize(70, 70))
                    else:
                        btn.setText(items[idx])
                else:
                    btn.setText("Ninguno")

                # Hacer clicable
                def make_clickable(i):
                    def on_click(event):
                        nonlocal index
                        index = i
                        update_carousel(emit_change=True)

                        if index == 0:
                            try:
                                if hasattr(self.worker, "remove_filter_category"):
                                    self.worker.remove_filter_category(filtro_name)
                                elif hasattr(self.worker, "quitar_filtro"):
                                    self.worker.quitar_filtro(filtro_name)
                                self.worker.filter_index_by_category[filtro_name] = -1
                                print(f"[UI] filtro quitado -> {filtro_name}")
                            except Exception as e:
                                print("[ERROR] al quitar filtro:", e)
                        else:
                            try:
                                self.worker.set_filter(filtro_name)
                                self.worker.filter_index_by_category[filtro_name] = index - 1
                                print(f"[UI] filtro aplicado -> {filtro_name} idx {index-1}")
                            except Exception as e:
                                print("[ERROR] al aplicar filtro:", e)

                        try:
                            self.worker.filter_changed.emit(filtro_name)
                        except Exception:
                            pass

                    return on_click

                btn.mousePressEvent = make_clickable(idx)
                carousel_layout.addWidget(btn)

            # Actualizar el texto superior (nombre del filtro)
            if index < len(nombres):
                main_label.setText(nombres[index])
            else:
                main_label.setText(items[index])

        # Inicializar
        update_carousel(emit_change=False)
        return view

    def closeEvent(self, event):
        print("[INFO] Cerrando cámara y hilo...")
        try:
            self.worker.stop_detection()
            self.thread.quit()
            self.thread.wait(3000)
        except Exception as e:
            print("[ERROR] Cerrando hilo:", e)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
