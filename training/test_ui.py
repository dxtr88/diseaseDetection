import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from load import PlantDiseasePredictor
from tkinter import messagebox, filedialog
import sys
from tkinterdnd2 import DND_FILES, TkinterDnD
import os



class PlantDiseaseUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Predictor")
        
        # Initialize the predictor
        try:
            self.predictor = PlantDiseasePredictor()
            print("Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            sys.exit(1)

        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Mode selection
        self.mode_frame = ttk.Frame(self.main_frame)
        self.mode_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        self.mode_label = ttk.Label(self.mode_frame, text="Select Mode:")
        self.mode_label.grid(row=0, column=0, padx=5)
        
        self.mode_var = tk.StringVar(value="webcam")
        self.mode_webcam = ttk.Radiobutton(
            self.mode_frame, 
            text="Webcam", 
            variable=self.mode_var, 
            value="webcam",
            command=self.switch_mode
        )
        self.mode_webcam.grid(row=0, column=1, padx=5)
        
        self.mode_dragdrop = ttk.Radiobutton(
            self.mode_frame, 
            text="Drag & Drop", 
            variable=self.mode_var, 
            value="dragdrop",
            command=self.switch_mode
        )
        self.mode_dragdrop.grid(row=0, column=2, padx=5)

        # Create frames for both modes
        self.webcam_frame = ttk.Frame(self.main_frame)
        self.dragdrop_frame = ttk.Frame(self.main_frame)

        # Initialize webcam components
        self.setup_webcam_mode()

        # Initialize drag & drop components
        self.setup_dragdrop_mode()

        # Results area (shared between modes)
        self.result_text = tk.Text(self.main_frame, height=10, width=50)
        self.result_text.grid(row=3, column=0, columnspan=2, pady=10)

        # Start in webcam mode
        self.switch_mode()

    def setup_webcam_mode(self):
        """Setup components for webcam mode"""
        # Video display area
        self.video_label = ttk.Label(self.webcam_frame)
        self.video_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Control buttons
        self.button_frame = ttk.Frame(self.webcam_frame)
        self.button_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.analyze_button = ttk.Button(
            self.button_frame, 
            text="Analyze Current Frame",
            command=self.analyze_frame
        )
        self.analyze_button.grid(row=0, column=0, padx=5)

        self.is_analyzing_live = False
        self.live_button = ttk.Button(
            self.button_frame,
            text="Start Live Analysis",
            command=self.toggle_live_analysis
        )
        self.live_button.grid(row=0, column=1, padx=5)

        self.cap = None

    def setup_dragdrop_mode(self):
        """Setup components for drag & drop mode"""
        # Create and configure drop zone
        self.drop_zone = ttk.Label(
            self.dragdrop_frame,
            text="Drag and drop image here\nor click to select",
            borderwidth=2,
            relief="solid",
            padding=100
        )
        self.drop_zone.grid(row=0, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Image display area
        self.image_label = ttk.Label(self.dragdrop_frame)
        self.image_label.grid(row=1, column=0, pady=10)

        # Bind events
        self.drop_zone.bind('<Button-1>', self.handle_click)
        self.drop_zone.drop_target_register(DND_FILES)
        self.drop_zone.dnd_bind('<<Drop>>', self.handle_drop)

    def switch_mode(self):
        """Switch between webcam and drag & drop modes"""
        mode = self.mode_var.get()
        
        # Hide both frames first
        self.webcam_frame.grid_remove()
        self.dragdrop_frame.grid_remove()

        if mode == "webcam":
            # Initialize webcam if needed
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Failed to open webcam")
                    self.mode_var.set("dragdrop")
                    self.dragdrop_frame.grid(row=1, column=0, columnspan=2)
                    return
            self.webcam_frame.grid(row=1, column=0, columnspan=2)
            self.update_frame()
        else:
            # Release webcam if it's open
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
                self.cap = None
            self.dragdrop_frame.grid(row=1, column=0, columnspan=2)

    def update_frame(self):
        """Update the video frame"""
        if self.cap is not None and self.cap.isOpened() and self.mode_var.get() == "webcam":
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.is_analyzing_live:
                    self.process_frame(frame_rgb)

                height, width = frame.shape[:2]
                max_size = 500
                if height > max_size or width > max_size:
                    scale = max_size / max(height, width)
                    frame_rgb = cv2.resize(frame_rgb, None, fx=scale, fy=scale)

                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=image)
                
                self.video_label.configure(image=photo)
                self.video_label.image = photo

            self.root.after(10, self.update_frame)

    def handle_drop(self, event):
        """Handle dropped files"""
        try:
            file_path = event.data
            if file_path.startswith('{'):
                file_path = file_path[1:-1]
            self.process_image(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process dropped file: {str(e)}")

    def handle_click(self, event):
        """Handle click on drop zone"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
            )
            if file_path:
                self.process_image(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {str(e)}")

    def process_image(self, image_path):
        """Process the selected image and display results"""
        try:
            self.result_text.delete(1.0, tk.END)
            
            image = Image.open(image_path)
            display_size = (300, 300)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

            predicted_class, confidence, _ = self.predictor.predict_image(
                image_path, show_image=False
            )

            self.result_text.insert(tk.END, f"=== Prediction Results ===\n\n")
            self.result_text.insert(tk.END, f"Image: {os.path.basename(image_path)}\n")
            self.result_text.insert(tk.END, f"Predicted Disease: {predicted_class}\n")
            self.result_text.insert(tk.END, f"Confidence: {confidence:.4f}\n")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def process_frame(self, frame):
        """Process a single frame and update results"""
        try:
            image = Image.fromarray(frame)
            temp_path = "temp_frame.jpg"
            image.save(temp_path)
            
            predicted_class, confidence, _ = self.predictor.predict_image(
                temp_path, show_image=False
            )

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"=== Live Analysis Results ===\n\n")
            self.result_text.insert(tk.END, f"Predicted Disease: {predicted_class}\n")
            self.result_text.insert(tk.END, f"Confidence: {confidence:.4f}\n")

            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error in analysis: {str(e)}\n")

    def analyze_frame(self):
        """Analyze current frame on button press"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.process_frame(frame_rgb)

    def toggle_live_analysis(self):
        """Toggle live analysis on/off"""
        self.is_analyzing_live = not self.is_analyzing_live
        if self.is_analyzing_live:
            self.live_button.configure(text="Stop Live Analysis")
        else:
            self.live_button.configure(text="Start Live Analysis")

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

def main():
    try:
        root = TkinterDnD.Tk()
    except:
        messagebox.showerror("Error", "Please install tkinterdnd2: pip install tkinterdnd2")
        return

    app = PlantDiseaseUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 