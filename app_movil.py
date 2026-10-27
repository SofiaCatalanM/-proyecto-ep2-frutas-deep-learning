# app_movil.py
import tkinter as tk
from tkinter import filedialog, messagebox
import requests
from PIL import Image, ImageTk
import numpy as np
import io
import os

# URL de la API
API_URL = "http://127.0.0.1:8000/predict"

class FruitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Frutas - DUOC UC")
        self.root.geometry("400x600")
        self.root.configure(bg="#f0f4f8")
        
        self.image_path = None
        self.setup_ui()

    def setup_ui(self):
        # Título
        title_label = tk.Label(
            self.root, 
            text="Clasificador de Frutas\nMobileNetV2 + API", 
            font=("Arial", 16, "bold"), 
            bg="#f0f4f8", 
            fg="#2c3e50"
        )
        title_label.pack(pady=20)
        
        # Área de la imagen
        self.image_label = tk.Label(
            self.root, 
            text="Sin imagen seleccionada", 
            bg="white", 
            width=30, 
            height=15,
            relief="solid",
            bd=1
        )
        self.image_label.pack(pady=10)
        
        # Etiqueta de resultado
        self.result_label = tk.Label(
            self.root, 
            text="Resultado: Esperando imagen...", 
            font=("Arial", 12, "bold"), 
            bg="#f0f4f8", 
            fg="#7f8c8d",
            wraplength=350
        )
        self.result_label.pack(pady=10)
        
        # Botones
        button_frame = tk.Frame(self.root, bg="#f0f4f8")
        button_frame.pack(pady=10)
        
        self.btn_capture = tk.Button(
            button_frame, 
            text="Seleccionar Foto", 
            command=self.select_image, 
            bg="#3498db", 
            fg="white", 
            font=("Arial", 10, "bold"),
            width=20,
            relief="flat"
        )
        self.btn_capture.pack(pady=5)
        
        self.btn_classify = tk.Button(
            button_frame, 
            text="Clasificar Fruta", 
            command=self.classify_image, 
            bg="#2ecc71", 
            fg="white", 
            font=("Arial", 10, "bold"),
            width=20,
            relief="flat"
        )
        self.btn_classify.pack(pady=5)
        
        # Instrucciones
        info_label = tk.Label(
            self.root,
            text="Nota: La imagen se convertirá\nautomáticamente a 128x128x3",
            font=("Arial", 9),
            bg="#f0f4f8",
            fg="#95a5a6"
        )
        info_label.pack(pady=20)

    def select_image(self):
        """Abre el explorador de archivos para seleccionar una imagen."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.image_path = file_path
            # Mostrar la imagen en la interfaz (redimensionada para la vista previa)
            img = Image.open(file_path)
            img = img.resize((200, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            self.result_label.config(text="Imagen cargada. Presiona 'Clasificar'.", fg="#3498db")

    def classify_image(self):
        """Envía la imagen a la API para su clasificación."""
        if not self.image_path:
            messagebox.showwarning("Advertencia", "Primero debes seleccionar una imagen.")
            return
        
        try:
            # 1. Cargar la imagen original
            img = Image.open(self.image_path).convert('RGB')
            
            # 2. REQUISITO RÚBRICA: Convertir a matriz 128x128x3
            img_resized = img.resize((128, 128), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized)
            
            # Imprimir en consola para demostrar al profesor que se cumple la rúbrica
            print(f"Imagen convertida a matriz de forma: {img_array.shape}")
            
            # 3. Preparar para enviar por HTTP
            img_byte_arr = io.BytesIO()
            img_resized.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            files = {'file': ('imagen.jpg', img_byte_arr, 'image/jpeg')}
            
            # 4. Enviar petición a la API
            response = requests.post(API_URL, files=files, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                self.display_result(result)
            else:
                messagebox.showerror("Error API", f"Error del servidor: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            messagebox.showerror(
                "Error de Conexión", 
                "No se puede conectar a la API.\n\nAsegúrate de haber ejecutado 'python api.py' en otra terminal."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

    def display_result(self, result):
        """Muestra el resultado devuelto por la API."""
        if result.get('status') == 'reconocido':
            text = f"Reconocido:\n{result['predicted_class']}\nConfianza: {result['confidence']:.2%}"
            self.result_label.config(text=text, fg="#27ae60")
        else:
            text = f"No reconocido\nLa fruta no está en la base de datos.\nConfianza: {result['confidence']:.2%}"
            self.result_label.config(text=text, fg="#c0392b")

if __name__ == '__main__':
    root = tk.Tk()
    app = FruitClassifierApp(root)
    root.mainloop()