# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import tkinter as tk
# from tkinter import filedialog, Label, Button
# from PIL import Image, ImageTk
# import numpy as np
# import cv2
# from tensorflow.keras.models import model_from_json

# # Suppress TensorFlow logs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# # Function to load the model
# def FacialExpressionModel(json_file, weights_file):
#     try:
#         with open(json_file, "r") as file:
#             loaded_model_json = file.read()
#             model = model_from_json(loaded_model_json)
#         model.load_weights(weights_file)
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         return model
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         exit()


# # Base directory paths
# base_path = os.path.dirname(os.path.abspath(__file__))
# haar_path = os.path.join(base_path, "haarcascade_frontalface_default.xml")
# json_path = os.path.join(base_path, "model_a1.json")
# weights_path = os.path.join(base_path, "model_weights.keras")

# # Load Haarcascade file
# try:
#     facec = cv2.CascadeClassifier(haar_path)
#     if facec.empty():
#         raise FileNotFoundError("Haarcascade file not found!")
# except FileNotFoundError as e:
#     print(e)
#     exit()

# # Load Emotion Detection model
# model = FacialExpressionModel(json_path, weights_path)

# # Emotion categories
# EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# # Initialize GUI
# top = tk.Tk()
# top.geometry('800x600')
# top.title('Emotion Detector')
# top.configure(background='#CDCDCD')

# label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
# sign_image = Label(top)


# # Emotion detection function
# def Detect(file_path):
#     global Label_packed

#     image = cv2.imread(file_path)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = facec.detectMultiScale(gray_image, 1.3, 5)
#     try:
#         if len(faces) == 0:
#             raise ValueError("No faces detected")
#         for (x, y, w, h) in faces:
#             fc = gray_image[y:y + h, x:x + w]
#             roi = cv2.resize(fc, (48, 48))
#             pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
#         print("Predicted Emotion is " + pred)
#         label1.configure(foreground="#011638", text=pred)
#     except Exception as e:
#         print(f"Error during detection: {e}")
#         label1.configure(foreground="#011638", text="Unable to detect")


# # Show the Detect button
# def show_Detect_button(file_path):
#     detect_b = Button(top, text="Detect Emotion", command=lambda: Detect(file_path), padx=10, pady=5)
#     detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
#     detect_b.place(relx=0.79, rely=0.46)


# # Upload image function
# def upload_image():
#     try:
#         file_path = filedialog.askopenfilename()
#         if not file_path:
#             return
#         uploaded = Image.open(file_path)
#         uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
#         im = ImageTk.PhotoImage(uploaded)

#         sign_image.configure(image=im)
#         sign_image.image = im
#         label1.configure(text='')
#         show_Detect_button(file_path)
#     except Exception as e:
#         print(f"Error uploading image: {e}")


# # Create Upload button
# upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
# upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
# upload.pack(side='bottom', pady=50)

# # Configure GUI elements
# sign_image.pack(side='bottom', expand=True)
# label1.pack(side='bottom', expand=True)
# heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
# heading.configure(background='#CDCDCD', foreground="#364156")
# heading.pack()

# # Start GUI
# top.mainloop()


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json

# Suppress TensorFlow and OpenCV logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cv2.setLogLevel(0)

class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.load_dependencies()
        self.create_widgets()

    def setup_window(self):
        """Configure the main window"""
        self.root.title('Emotion Detector')
        self.root.geometry('900x700')
        self.root.configure(background='#1F2937')
        self.root.resizable(True, True)

    def load_dependencies(self):
        """Load model and cascade classifier"""
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        try:
            # Load Haarcascade for face detection
            haar_path = os.path.join(base_path, "haarcascade_frontalface_default.xml")
            self.facec = cv2.CascadeClassifier(haar_path)
            if self.facec.empty():
                raise FileNotFoundError("Haarcascade file not found!")

            # Load Emotion Detection model
            json_path = os.path.join(base_path, "model_a2.json")
            weights_path = os.path.join(base_path, "modelWeight.weights.h5")
            self.model = self.load_model(json_path, weights_path)
            
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))
            self.root.quit()

    def load_model(self, json_file, weights_file):
        """Load the emotion detection model"""
        try:
            with open(json_file, "r") as file:
                loaded_model_json = file.read()
                model = model_from_json(loaded_model_json)
            model.load_weights(weights_file)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Could not load model: {e}")
            return None

    def create_widgets(self):
        """Create and layout GUI widgets"""
        # Title
        self.title_label = tk.Label(
            self.root, 
            text='Emotion Detector', 
            font=('Arial', 25, 'bold'), 
            background='#1F2937',
            foreground="#FFFFFF"
        )
        self.title_label.pack(pady=20)

        # Image Display Frame
        self.image_frame = tk.Frame(self.root, background='#1F2937')
        self.image_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        self.image_label = tk.Label(
            self.image_frame, 
            background='#FFFFFF', 
            relief=tk.SUNKEN, 
            borderwidth=2
        )
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Result Frame
        self.result_frame = tk.Frame(self.root, background='#1F2937')
        self.result_frame.pack(expand=False, fill=tk.X, padx=20, pady=10)

        self.result_label = tk.Label(
            self.result_frame, 
            text='', 
            font=('Arial', 16, 'bold'), 
            background='#1F2937',
            foreground="#FFFFFF"
        )
        self.result_label.pack(pady=10)

        # Button Frame
        self.button_frame = tk.Frame(self.root, background='#1F2937')
        self.button_frame.pack(expand=False, fill=tk.X, padx=20, pady=20)

        self.upload_button = ttk.Button(
            self.button_frame, 
            text="Upload Image", 
            command=self.upload_image,
            style="Accent.TButton"
        )
        self.upload_button.pack(expand=True, padx=20, pady=15)

    def upload_image(self):
        """Handle image upload and emotion detection"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                    ("All files", "*.*")
                ]
            )
            if not file_path:
                return

            # Load and display image
            uploaded = Image.open(file_path)
            uploaded.thumbnail((500, 500))
            img_tk = ImageTk.PhotoImage(uploaded)
            
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk

            # Detect emotion
            emotion = self.detect_emotion(file_path)
            self.result_label.config(text=f"Predicted Emotion: {emotion}")

        except Exception as e:
            messagebox.showerror("Upload Error", str(e))

    def detect_emotion(self, file_path):
        """Detect emotion in the uploaded image"""
        EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.facec.detectMultiScale(gray_image, 1.3, 5)

        if len(faces) == 0:
            return "No face detected"

        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = EMOTIONS_LIST[np.argmax(self.model.predict(roi[np.newaxis, :, :, np.newaxis]))]
            return pred

def main():
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()