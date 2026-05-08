import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas
import cv2
import numpy as np
from PIL import Image, ImageTk

# Intentar importar inpaint_biharmonic de scikit-image (mejor calidad)
try:
    from skimage.restoration import inpaint_biharmonic
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("scikit-image no instalado. Usando cv2.inpaint. Para mejor calidad ejecuta: pip install scikit-image")

class ClearLensPID(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ClearLens v2.0 - Procesamiento Digital de Imágenes")
        self.geometry("1200x850")
        
        self.imagen_original = None
        self.mascara_manual = None
        self.pintando = False
        self.brush_size = 20
        self.photo = None

        # Parámetros de inpainting (para cv2.inpaint, si se usa)
        self.inpaint_radius = 7
        self.inpaint_method = cv2.INPAINT_NS   # Navier-Stokes

        # Elegir método preferido (True = usar skimage, False = usar cv2)
        self.use_skimage = SKIMAGE_AVAILABLE   # True si está disponible

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", rowspan=2)
        
        ctk.CTkLabel(self.sidebar, text="ClearLens PID", font=("Arial", 24, "bold")).pack(pady=20)

        # ----------------------------------------------
        # Módulo 1: Limpieza manual
        # ----------------------------------------------
        self.btn_limpieza = ctk.CTkButton(self.sidebar, text="1. Limpieza de Escena", 
                                         fg_color="#3498db", command=self.modulo_limpieza)
        self.btn_limpieza.pack(pady=10, padx=20, fill="x")
        ctk.CTkLabel(self.sidebar, text="(Pinta sobre el objeto a eliminar)", font=("Arial", 10)).pack()

        # Slider para grosor del pincel
        ctk.CTkLabel(self.sidebar, text="Grosor del pincel (píxeles):", font=("Arial", 12)).pack(pady=(10,0))
        self.slider_brush = ctk.CTkSlider(self.sidebar, from_=5, to=50, number_of_steps=45, command=self.cambiar_brush)
        self.slider_brush.set(20)
        self.slider_brush.pack(pady=5, padx=20, fill="x")
        self.label_brush = ctk.CTkLabel(self.sidebar, text="20 px", font=("Arial", 10))
        self.label_brush.pack()

        # Si no se puede usar skimage, mostramos controles de radio y método
        if not SKIMAGE_AVAILABLE:
            ctk.CTkLabel(self.sidebar, text="Radio de inpainting (solo cv2):", font=("Arial", 12)).pack(pady=(10,0))
            self.slider_radius = ctk.CTkSlider(self.sidebar, from_=1, to=15, number_of_steps=14, command=self.cambiar_radius)
            self.slider_radius.set(7)
            self.slider_radius.pack(pady=5, padx=20, fill="x")
            self.label_radius = ctk.CTkLabel(self.sidebar, text="7 px", font=("Arial", 10))
            self.label_radius.pack()

            self.inpaint_method_name = "Navier-Stokes"
            self.btn_method = ctk.CTkButton(self.sidebar, text=f"Método: {self.inpaint_method_name}", 
                                           command=self.cambiar_metodo, fg_color="#555")
            self.btn_method.pack(pady=5, padx=20, fill="x")
        else:
            # Indicar que se usa método avanzado
            ctk.CTkLabel(self.sidebar, text="Método: Inpainting Biharmónico\n(skimage)", font=("Arial", 10), text_color="green").pack(pady=5)

        # Botón automático
        self.btn_auto_cables = ctk.CTkButton(self.sidebar, text="Eliminar cables/poste (auto)", 
                                            fg_color="#e67e22", command=self.eliminar_cables_auto)
        self.btn_auto_cables.pack(pady=10, padx=20, fill="x")
        ctk.CTkLabel(self.sidebar, text="(Detecta objetos alargados)", font=("Arial", 10)).pack()

        # Módulo 2 y 3
        self.btn_macro = ctk.CTkButton(self.sidebar, text="2. Enfoque Macro", 
                                      fg_color="#27ae60", command=self.modulo_macro)
        self.btn_macro.pack(pady=10, padx=20, fill="x")
        ctk.CTkLabel(self.sidebar, text="(Desenfoque de fondo por bordes)", font=("Arial", 10)).pack()

        self.btn_docs = ctk.CTkButton(self.sidebar, text="3. Restaurar Documento", 
                                     fg_color="#f39c12", command=self.modulo_documentos)
        self.btn_docs.pack(pady=10, padx=20, fill="x")
        ctk.CTkLabel(self.sidebar, text="(Corrige perspectiva y limpia manchas)", font=("Arial", 10)).pack()

        self.btn_importar = ctk.CTkButton(self.sidebar, text="Importar Imagen", command=self.importar)
        self.btn_importar.pack(pady=20, padx=20, fill="x")

        self.btn_salir = ctk.CTkButton(self.sidebar, text="Salir", fg_color="gray", command=self.destroy)
        self.btn_salir.pack(pady=10, padx=20, fill="x")

        # Canvas para mostrar la imagen
        self.canvas = Canvas(self, bg="gray15", highlightthickness=0)
        self.canvas.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.canvas.bind("<Button-1>", self.iniciar_pintado)
        self.canvas.bind("<B1-Motion>", self.pintar)
        self.canvas.bind("<ButtonRelease-1>", self.terminar_pintado)
        self.canvas.configure(cursor="cross")

        self.label_info = ctk.CTkLabel(self, text="", font=("Arial", 12))
        self.label_info.grid(row=1, column=1, pady=5)

        # Variables de escalado para el canvas
        self.escala_img = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0

    # ------------------- Auxiliares -------------------
    def cambiar_brush(self, value):
        self.brush_size = int(value)
        self.label_brush.configure(text=f"{self.brush_size} px")

    def cambiar_radius(self, value):
        self.inpaint_radius = int(value)
        self.label_radius.configure(text=f"{self.inpaint_radius} px")

    def cambiar_metodo(self):
        if self.inpaint_method == cv2.INPAINT_TELEA:
            self.inpaint_method = cv2.INPAINT_NS
            self.inpaint_method_name = "Navier-Stokes"
        else:
            self.inpaint_method = cv2.INPAINT_TELEA
            self.inpaint_method_name = "Telea"
        self.btn_method.configure(text=f"Método: {self.inpaint_method_name}")

    def importar(self):
        ruta = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp")])
        if ruta:
            self.imagen_original = cv2.imread(ruta)
            if self.imagen_original is None:
                messagebox.showerror("Error", "No se pudo leer la imagen")
                return
            self.mostrar(self.imagen_original)
            self.mascara_manual = np.zeros(self.imagen_original.shape[:2], dtype=np.uint8)
            self.label_info.configure(text="Imagen cargada. Pinta sobre el objeto a eliminar.")

    def mostrar(self, img_cv):
        if img_cv is None:
            return
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]
        max_w, max_h = 850, 650
        escala = min(max_w / w_orig, max_h / h_orig)
        nuevo_ancho = int(w_orig * escala)
        nuevo_alto = int(h_orig * escala)
        self.escala_img = escala

        img_resized = cv2.resize(img_rgb, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(img_resized))

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1:
            canvas_w = 850
            canvas_h = 650
        self.canvas_offset_x = (canvas_w - nuevo_ancho) // 2
        self.canvas_offset_y = (canvas_h - nuevo_alto) // 2

        self.canvas.delete("all")
        self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y, anchor="nw", image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    # ------------------- Pintado manual -------------------
    def iniciar_pintado(self, event):
        if self.imagen_original is None:
            return
        if self.mascara_manual is None:
            self.mascara_manual = np.zeros(self.imagen_original.shape[:2], dtype=np.uint8)
        self.pintando = True
        self.pintar(event)

    def pintar(self, event):
        if not self.pintando or self.imagen_original is None:
            return
        x_canvas = event.x
        y_canvas = event.y
        x_rel = x_canvas - self.canvas_offset_x
        y_rel = y_canvas - self.canvas_offset_y

        img_resized_w = int(self.imagen_original.shape[1] * self.escala_img)
        img_resized_h = int(self.imagen_original.shape[0] * self.escala_img)
        if x_rel < 0 or y_rel < 0 or x_rel >= img_resized_w or y_rel >= img_resized_h:
            return

        x_orig = int(x_rel / self.escala_img)
        y_orig = int(y_rel / self.escala_img)

        if 0 <= x_orig < self.imagen_original.shape[1] and 0 <= y_orig < self.imagen_original.shape[0]:
            cv2.circle(self.mascara_manual, (x_orig, y_orig), self.brush_size, 255, -1)
            # Feedback visual: overlay rojo
            temp = self.imagen_original.copy()
            overlay = np.zeros_like(temp)
            overlay[:, :, 2] = self.mascara_manual
            temp = cv2.addWeighted(temp, 0.7, overlay, 0.3, 0)
            self.mostrar(temp)

    def terminar_pintado(self, event):
        self.pintando = False
        if self.mascara_manual is not None and np.any(self.mascara_manual):
            self.label_info.configure(text="Máscara creada. Presiona 'Limpieza de Escena'.")
        else:
            self.label_info.configure(text="No se pintó nada. Vuelve a intentarlo.")

    # ------------------- Limpieza manual con método mejorado -------------------
    def modulo_limpieza(self):
        if self.imagen_original is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return
        if self.mascara_manual is None or not np.any(self.mascara_manual):
            messagebox.showwarning("Aviso", "Pinta sobre los objetos a eliminar primero.")
            return

        # Aplicar inpainting con el método elegido
        if self.use_skimage and SKIMAGE_AVAILABLE:
            # Convertir imagen a float [0,1] y máscara a booleano
            img_float = self.imagen_original.astype(np.float32) / 255.0
            mask_bool = self.mascara_manual > 0
            # Biharmonic inpainting (excelente para texturas)
            result_float = inpaint_biharmonic(img_float, mask_bool, multichannel=True)
            resultado = (result_float * 255).astype(np.uint8)
            metodo_usado = "skimage (biharmónico)"
        else:
            # Suavizado de máscara para transiciones
            mascara_suave = self.mascara_manual.astype(np.float32) / 255.0
            mascara_suave = cv2.GaussianBlur(mascara_suave, (5,5), 0)
            mascara_suave = (mascara_suave * 255).astype(np.uint8)
            resultado = cv2.inpaint(self.imagen_original, mascara_suave,
                                    inpaintRadius=self.inpaint_radius,
                                    flags=self.inpaint_method)
            metodo_usado = f"cv2.inpaint ({self.inpaint_method_name})"

        self.imagen_original = resultado
        self.mascara_manual = np.zeros(self.imagen_original.shape[:2], dtype=np.uint8)
        self.mostrar(resultado)
        self.label_info.configure(text=f"Limpieza aplicada con {metodo_usado}.")
        messagebox.showinfo("Éxito", f"Limpieza completada.\nMétodo: {metodo_usado}")

    # ------------------- Módulos automático, macro y documentos -------------------
    # (Mantén las funciones que ya tenías; aquí las incluyo completas para que funcione)
    def eliminar_cables_auto(self):
        if self.imagen_original is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        h, w = self.imagen_original.shape[:2]
        total_pixeles = h * w

        gray = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        gray_blur = cv2.GaussianBlur(gray_eq, (5,5), 0)
        edges = cv2.Canny(gray_blur, 30, 100)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 30))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        edges_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)
        edges_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)
        edges_connected = cv2.bitwise_or(edges_v, edges_h)

        contornos, _ = cv2.findContours(edges_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mascara = np.zeros_like(gray)
        max_area_contorno = total_pixeles * 0.15

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            x, y, wc, hc = cv2.boundingRect(cnt)
            aspecto = max(wc, hc) / (min(wc, hc) + 1)
            longitud = max(wc, hc)
            if 300 < area < max_area_contorno and aspecto > 2.5 and longitud > 70:
                x_margin = max(0, x - 10)
                y_margin = max(0, y - 10)
                w_margin = min(w - x_margin, wc + 20)
                h_margin = min(h - y_margin, hc + 20)
                cv2.rectangle(mascara, (x_margin, y_margin), (x_margin + w_margin, y_margin + h_margin), 255, cv2.FILLED)

        lines = cv2.HoughLinesP(edges_connected, rho=1, theta=np.pi/180, threshold=80,
                                minLineLength=50, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(mascara, (x1, y1), (x2, y2), 255, thickness=6)

        _, thresh = cv2.threshold(gray_blur, 100, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        cont_oscuros, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cont_oscuros:
            area = cv2.contourArea(cnt)
            x, y, wc, hc = cv2.boundingRect(cnt)
            aspecto = max(wc, hc) / (min(wc, hc) + 1)
            if area > 500 and aspecto > 2.5 and wc > 20 and hc > 100:
                cv2.drawContours(mascara, [cnt], -1, 255, cv2.FILLED)

        porcentaje_mascara = (cv2.countNonZero(mascara) / total_pixeles) * 100
        if porcentaje_mascara > 25:
            messagebox.showwarning("Advertencia", f"Máscara cubre {porcentaje_mascara:.1f}% -> cancelado.")
            return
        if cv2.countNonZero(mascara) == 0:
            messagebox.showinfo("Info", "No se detectaron postes o cables.")
            return

        kernel = np.ones((5,5), np.uint8)
        mascara = cv2.dilate(mascara, kernel, iterations=3)

        # Para la reconstrucción automática también usamos el método avanzado si está disponible
        if self.use_skimage and SKIMAGE_AVAILABLE:
            img_float = self.imagen_original.astype(np.float32) / 255.0
            mask_bool = mascara > 0
            result_float = inpaint_biharmonic(img_float, mask_bool, multichannel=True)
            resultado = (result_float * 255).astype(np.uint8)
        else:
            mascara_suave = cv2.GaussianBlur(mascara.astype(np.float32), (5,5), 0)
            mascara_suave = (mascara_suave * 255).astype(np.uint8)
            resultado = cv2.inpaint(self.imagen_original, mascara_suave, 7, cv2.INPAINT_NS)

        self.imagen_original = resultado
        self.mascara_manual = np.zeros_like(mascara)
        self.mostrar(resultado)
        self.label_info.configure(text=f"Postes/cables eliminados (máscara {porcentaje_mascara:.1f}%)")
        messagebox.showinfo("Éxito", "Limpieza automática completada.")

    def modulo_macro(self):
        if self.imagen_original is None:
            messagebox.showwarning("Aviso", "Carga una imagen primero.")
            return
        img = self.imagen_original.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bordes = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5,5), np.uint8)
        bordes_cerrados = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel, iterations=2)
        contornos, _ = cv2.findContours(bordes_cerrados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mascara_sujeto = np.zeros_like(gray)
        if contornos:
            contorno_principal = max(contornos, key=cv2.contourArea)
            cv2.drawContours(mascara_sujeto, [contorno_principal], -1, 255, cv2.FILLED)
        else:
            h, w = gray.shape
            cv2.ellipse(mascara_sujeto, (w//2, h//2), (w//4, h//4), 0, 0, 360, 255, -1)
        fondo_borroso = cv2.GaussianBlur(img, (55,55), 0)
        sujeto = cv2.bitwise_and(img, img, mask=mascara_sujeto)
        fondo = cv2.bitwise_and(fondo_borroso, fondo_borroso, mask=cv2.bitwise_not(mascara_sujeto))
        resultado = cv2.add(sujeto, fondo)
        self.imagen_original = resultado
        self.mostrar(resultado)
        self.label_info.configure(text="Enfoque macro aplicado.")
        messagebox.showinfo("Éxito", "Enfoque macro aplicado.")

    def modulo_documentos(self):
        if self.imagen_original is None:
            messagebox.showwarning("Aviso", "Carga una imagen primero.")
            return
        img = self.imagen_original.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Corrección de perspectiva (simple)
        bordes = cv2.Canny(gray, 50, 150)
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contornos:
            c = max(contornos, key=cv2.contourArea)
            peri = cv2.arcLength(c, True)
            aprox = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(aprox) == 4:
                pts = aprox.reshape(4, 2)
                rect = np.zeros((4,2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                (tl, tr, br, bl) = rect
                anchoA = np.linalg.norm(br - bl)
                anchoB = np.linalg.norm(tr - tl)
                ancho = max(int(anchoA), int(anchoB))
                altoA = np.linalg.norm(tr - br)
                altoB = np.linalg.norm(tl - bl)
                alto = max(int(altoA), int(altoB))
                dst = np.array([[0,0], [ancho-1,0], [ancho-1,alto-1], [0,alto-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                img = cv2.warpPerspective(img, M, (ancho, alto))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # CLAHE + binarización
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_eq = clahe.apply(gray)
        _, thresh = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        thresh_limpio = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        resultado_binario = cv2.bitwise_not(thresh_limpio)
        resultado = cv2.cvtColor(resultado_binario, cv2.COLOR_GRAY2BGR)
        self.imagen_original = resultado
        self.mostrar(resultado)
        self.label_info.configure(text="Documento restaurado.")
        messagebox.showinfo("Éxito", "Restauración completada.")

if __name__ == "__main__":
    app = ClearLensPID()
    app.mainloop()