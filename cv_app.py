import os
os.environ['TCL_LIBRARY'] = r'C:\Users\ADMIN\AppData\Local\Programs\Python\Python311\tcl\tcl8.6'
os.environ['TK_LIBRARY']  = r'C:\Users\ADMIN\AppData\Local\Programs\Python\Python311\tcl\tk8.6'

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf
import os

def load_models():
    models = {}

    if os.path.exists('ann_mnist.h5'):
        models['ANN'] = tf.keras.models.load_model('ann_mnist.h5')
        print("ANN model loaded.")
    else:
        print("WARNING: ann_mnist.h5 not found. Run train_models.py first.")

    if os.path.exists('cnn_mnist.h5'):
        models['CNN'] = tf.keras.models.load_model('cnn_mnist.h5')
        print("CNN model loaded.")
    else:
        print("WARNING: cnn_mnist.h5 not found. Run train_models.py first.")

    return models

def preprocess_canvas(pil_image, model_type='ANN'):
    gray = pil_image.convert('L')

    resized = gray.resize((28, 28), Image.LANCZOS)
    inverted = ImageOps.invert(resized)
    arr = np.array(inverted).astype('float32') / 255.0

    if model_type == 'ANN':
        return arr.reshape(1, 784)
    else:
        return arr.reshape(1, 28, 28, 1)

class DigitDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSC 126 - Digit Recognition (ANN & CNN)")
        self.root.configure(bg='#1e1e2e')
        self.root.resizable(False, False)

        self.models = load_models()
        self.active_model = tk.StringVar(value='CNN')

        self.canvas_size = 280
        self.drawing = False
        self.last_x = None
        self.last_y = None

        self.pil_image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'white')
        self.pil_draw  = ImageDraw.Draw(self.pil_image)

        self._build_ui()

    # UI BUILDER

    def _build_ui(self):
        tk.Label(
            self.root,
            text="Handwritten Digit Recognizer",
            bg='#1e1e2e', fg='#89b4fa',
            font=('Segoe UI', 14, 'bold')
        ).pack(pady=(16, 4))

        tk.Label(
            self.root,
            text="Draw a digit (0-9) on the canvas below",
            bg='#1e1e2e', fg='#a6adc8',
            font=('Segoe UI', 9)
        ).pack()

        model_frame = tk.Frame(self.root, bg='#1e1e2e')
        model_frame.pack(pady=8)

        tk.Label(model_frame, text="Model:", bg='#1e1e2e',
                 fg='#cdd6f4', font=('Segoe UI', 10)).pack(side='left')

        for m in ('ANN', 'CNN'):
            tk.Radiobutton(
                model_frame, text=m,
                variable=self.active_model, value=m,
                bg='#1e1e2e', fg='#cdd6f4',
                selectcolor='#313244',
                activebackground='#1e1e2e',
                font=('Segoe UI', 10)
            ).pack(side='left', padx=10)

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='white',
            cursor='crosshair',
            highlightthickness=2,
            highlightbackground='#89b4fa'
        )
        self.canvas.pack(padx=16, pady=8)

        self.canvas.bind('<Button-1>',        self._start_draw)
        self.canvas.bind('<B1-Motion>',       self._draw)
        self.canvas.bind('<ButtonRelease-1>', self._stop_draw)

        btn_frame = tk.Frame(self.root, bg='#1e1e2e')
        btn_frame.pack(pady=8)

        tk.Button(
            btn_frame, text="Predict",
            bg='#89b4fa', fg='#1e1e2e',
            font=('Segoe UI', 11, 'bold'),
            relief='flat', cursor='hand2',
            activebackground='#b4d0fa',
            width=12,
            command=self._predict
        ).pack(side='left', padx=6, ipady=5)

        tk.Button(
            btn_frame, text="Clear",
            bg='#f38ba8', fg='#1e1e2e',
            font=('Segoe UI', 11, 'bold'),
            relief='flat', cursor='hand2',
            activebackground='#f5a0b5',
            width=12,
            command=self._clear
        ).pack(side='left', padx=6, ipady=5)

        result_frame = tk.Frame(self.root, bg='#313244',
                                padx=16, pady=12)
        result_frame.pack(fill='x', padx=16, pady=(4, 16))

        tk.Label(result_frame, text="Predicted Digit:",
                 bg='#313244', fg='#a6adc8',
                 font=('Segoe UI', 10)).pack()

        self.result_label = tk.Label(
            result_frame, text="-",
            bg='#313244', fg='#a6e3a1',
            font=('Segoe UI', 48, 'bold')
        )
        self.result_label.pack()

        self.confidence_label = tk.Label(
            result_frame, text="",
            bg='#313244', fg='#cdd6f4',
            font=('Segoe UI', 10)
        )
        self.confidence_label.pack()

        self.model_used_label = tk.Label(
            result_frame, text="",
            bg='#313244', fg='#a6adc8',
            font=('Segoe UI', 9)
        )
        self.model_used_label.pack()

        self.bar_frame = tk.Frame(result_frame, bg='#313244')
        self.bar_frame.pack(pady=(8, 0))
        self.bar_labels = []
        for i in range(10):
            col_frame = tk.Frame(self.bar_frame, bg='#313244')
            col_frame.pack(side='left', padx=2)
            bar = tk.Canvas(col_frame, width=20, height=60,
                            bg='#45475a', highlightthickness=0)
            bar.pack()
            lbl = tk.Label(col_frame, text=str(i),
                           bg='#313244', fg='#cdd6f4',
                           font=('Segoe UI', 8))
            lbl.pack()
            self.bar_labels.append(bar)

    # DRAWING HANDLERS

    def _start_draw(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def _draw(self, event):
        if not self.drawing:
            return
        x, y = event.x, event.y
        r = 10 
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                fill='black', width=r * 2,
                capstyle=tk.ROUND, smooth=True
            )
            self.pil_draw.line(
                [self.last_x, self.last_y, x, y],
                fill='black', width=r * 2
            )
        self.last_x = x
        self.last_y = y

    def _stop_draw(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None

    # CLEAR

    def _clear(self):
        self.canvas.delete('all')
        self.pil_image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'white')
        self.pil_draw  = ImageDraw.Draw(self.pil_image)
        self.result_label.config(text='-')
        self.confidence_label.config(text='')
        self.model_used_label.config(text='')
        for bar in self.bar_labels:
            bar.delete('all')

    # PREDICT

    def _predict(self):
        model_name = self.active_model.get()

        if model_name not in self.models:
            messagebox.showerror("Error",
                f"{model_name} model not loaded.\nPlease run train_models.py first.")
            return

        model = self.models[model_name]

        processed = preprocess_canvas(self.pil_image, model_type=model_name)

        predictions = model.predict(processed, verbose=0)[0]
        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        self.result_label.config(text=str(predicted_digit))
        self.confidence_label.config(
            text=f"Confidence: {confidence:.1f}%"
        )
        self.model_used_label.config(
            text=f"Model used: {model_name}"
        )

        for i, bar in enumerate(self.bar_labels):
            bar.delete('all')
            prob = predictions[i]
            bar_height = int(prob * 55)
            color = '#89b4fa' if i == predicted_digit else '#585b70'
            bar.create_rectangle(
                0, 60 - bar_height, 20, 60,
                fill=color, outline=''
            )


if __name__ == '__main__':
    root = tk.Tk()
    app = DigitDrawingApp(root)
    root.mainloop()