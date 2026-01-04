import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

# Set appearance and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class SalaryPredictorGUI:
    def __init__(self):
        # Model parameters from C++ calculation
        self.slope = 9449.96
        self.intercept = 24848.20
        self.professional_bonus = False
        
        # Load MobileNetV2 model
        self.image_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=True
        )
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("AI Salary Predictor")
        self.root.geometry("800x520")
        self.root.resizable(False, False)
        
        self.setup_ui()
    
    def setup_ui(self):
        # Create main frame for left side controls
        left_frame = ctk.CTkFrame(self.root, width=400)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        left_frame.pack_propagate(False)
        
        # Create right frame for chart
        right_frame = ctk.CTkFrame(self.root, width=380)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Profile photo display
        self.photo_label = ctk.CTkLabel(
            left_frame,
            text="No Photo",
            width=120,
            height=120,
            fg_color="gray30",
            corner_radius=10
        )
        self.photo_label.pack(pady=10)
        
        # Upload photo button
        upload_button = ctk.CTkButton(
            left_frame,
            text="Upload Profile Photo",
            command=self.upload_photo,
            font=ctk.CTkFont(size=14),
            width=180,
            height=30
        )
        upload_button.pack(pady=5)
        
        # Title label
        title_label = ctk.CTkLabel(
            left_frame,
            text="AI Salary Predictor",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(pady=15)
        
        # Input field for years of experience
        input_label = ctk.CTkLabel(
            left_frame,
            text="Years of Experience:",
            font=ctk.CTkFont(size=16)
        )
        input_label.pack(pady=(10, 5))
        
        self.experience_entry = ctk.CTkEntry(
            left_frame,
            placeholder_text="Enter years...",
            font=ctk.CTkFont(size=14),
            width=200,
            height=35
        )
        self.experience_entry.pack(pady=5)
        
        # Predict button
        predict_button = ctk.CTkButton(
            left_frame,
            text="Predict",
            command=self.predict_salary,
            font=ctk.CTkFont(size=18, weight="bold"),
            width=200,
            height=45
        )
        predict_button.pack(pady=20)
        
        # Result label
        self.result_label = ctk.CTkLabel(
            left_frame,
            text="Predicted Salary: $0.00",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.result_label.pack(pady=10)
        self.final_y = self.result_label.winfo_reqheight() + 400
        
        # Progress bar for AI analysis
        self.progress_bar = ctk.CTkProgressBar(
            left_frame,
            width=200,
            height=20
        )
        
        # Progress label
        self.progress_label = ctk.CTkLabel(
            left_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        
        # Image classification result
        self.classification_label = ctk.CTkLabel(
            left_frame,
            text="Image: Not analyzed",
            font=ctk.CTkFont(size=14)
        )
        self.classification_label.pack(pady=5)
        
        # Create matplotlib chart in right frame
        self.create_chart(right_frame)
    
    def predict_salary(self):
        try:
            # Get input value
            experience = float(self.experience_entry.get())
            
            # Calculate prediction using linear regression formula
            predicted_salary = self.slope * experience + self.intercept
            
            # Apply professional bonus if image was analyzed
            if self.professional_bonus:
                predicted_salary *= 1.05
            
            # Update chart with prediction point
            self.update_chart_prediction(experience, predicted_salary)
            
            # Store result and start progress animation
            self.final_result = f"Predicted Salary: ${predicted_salary:,.2f}"
            self.start_ai_analysis()
            
        except ValueError:
            # Handle invalid input
            self.final_result = "Invalid Input"
            self.start_ai_analysis()
    
    def classify_image(self, image_path):
        try:
            # Load and preprocess image for MobileNetV2
            image = tf.keras.preprocessing.image.load_img(
                image_path, target_size=(224, 224)
            )
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
            
            # Make prediction
            predictions = self.image_model.predict(image_array, verbose=0)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
            
            # Get top prediction
            class_name = decoded[0][1].replace('_', ' ').title()
            confidence = decoded[0][2] * 100
            
            return f"{class_name} ({confidence:.1f}%)"
            
        except Exception as e:
            return "Classification failed"
    
    def create_chart(self, parent_frame):
        # Load data from CSV
        data = pd.read_csv('data.csv')
        
        # Create matplotlib figure with dark theme
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        fig.patch.set_facecolor('#212121')
        ax.set_facecolor('#2b2b2b')
        
        # Scatter plot of data points
        ax.scatter(data['YearsExperience'], data['Salary'], 
                  color='lightblue', alpha=0.7, s=30)
        
        # Prepare regression line data for animation
        self.x_line = np.linspace(data['YearsExperience'].min(), 
                                 data['YearsExperience'].max(), 50)
        self.y_line = self.slope * self.x_line + self.intercept
        
        # Styling
        ax.set_xlabel('Years of Experience', color='white')
        ax.set_ylabel('Salary ($)', color='white')
        ax.set_title('Salary vs Experience', color='white', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Embed chart in tkinter
        self.canvas = FigureCanvasTkAgg(fig, parent_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        self.canvas.draw()
        
        # Store references for updating
        self.fig = fig
        self.ax = ax
        self.prediction_point = None
        
        # Start animated line drawing
        self.animate_line_drawing(0)
    
    def animate_line_drawing(self, segment):
        if segment <= len(self.x_line) - 1:
            # Draw line segment by segment
            if segment > 0:
                self.ax.plot(self.x_line[:segment+1], self.y_line[:segment+1], 
                           color='#1f77b4', linewidth=2)
                self.canvas.draw()
            
            # Schedule next segment (1000ms / 50 segments = 20ms per segment)
            self.root.after(20, lambda: self.animate_line_drawing(segment + 1))
        else:
            # Animation complete, add legend
            self.ax.plot([], [], color='#1f77b4', linewidth=2, 
                        label=f'y = {self.slope:.0f}x + {self.intercept:.0f}')
            self.ax.legend()
            self.canvas.draw()
    
    def update_chart_prediction(self, x, y):
        # Remove previous prediction point if it exists
        if self.prediction_point:
            self.prediction_point.remove()
        
        # Add new prediction point as red star
        self.prediction_point = self.ax.scatter(x, y, color='red', s=100, 
                                               marker='*', label='Your Prediction', zorder=5)
        
        # Update legend and redraw
        self.ax.legend()
        self.canvas.draw()
        
        # Start pulsing animation
        self.pulse_prediction_point(0)
    
    def pulse_prediction_point(self, pulse_count):
        if pulse_count < 6:  # 3 pulses (fade out + fade in = 2 steps per pulse)
            # Alternate between visible and hidden
            alpha = 0.3 if pulse_count % 2 == 0 else 1.0
            if self.prediction_point:
                self.prediction_point.set_alpha(alpha)
                self.canvas.draw()
            
            # Schedule next pulse step
            self.root.after(200, lambda: self.pulse_prediction_point(pulse_count + 1))
        else:
            # Ensure final visibility
            if self.prediction_point:
                self.prediction_point.set_alpha(1.0)
                self.canvas.draw()
    
    def start_ai_analysis(self):
        # Show progress bar and label
        self.progress_bar.pack(pady=5)
        self.progress_label.pack(pady=2)
        self.progress_label.configure(text="AI is analyzing pixels...")
        
        # Reset progress bar
        self.progress_bar.set(0)
        
        # Start progress animation
        self.animate_progress(0)
    
    def animate_progress(self, progress):
        if progress <= 1.0:
            self.progress_bar.set(progress)
            # Schedule next frame (1 second / 50 frames = 20ms per frame)
            self.root.after(20, lambda: self.animate_progress(progress + 0.02))
        else:
            # Hide progress elements
            self.progress_bar.pack_forget()
            self.progress_label.pack_forget()
            
            # Show final result with animation
            self.result_label.configure(text=self.final_result)
            self.animate_slide_up()
    
    def animate_slide_up(self):
        # Start position (off-screen at bottom)
        start_y = 600
        end_y = self.final_y
        
        # Animation parameters
        duration = 500  # 0.5 seconds in milliseconds
        frames = 30
        frame_delay = duration // frames
        
        # Position the label at start position
        self.result_label.place(x=200, y=start_y, anchor="center")
        
        # Start animation
        self.animate_frame(start_y, end_y, frames, frame_delay, 0)
    
    def animate_frame(self, start_y, end_y, total_frames, frame_delay, current_frame):
        if current_frame <= total_frames:
            # Calculate current position using easing
            progress = current_frame / total_frames
            current_y = start_y + (end_y - start_y) * progress
            
            # Update position
            self.result_label.place(x=200, y=current_y, anchor="center")
            
            # Schedule next frame
            self.root.after(frame_delay, lambda: self.animate_frame(
                start_y, end_y, total_frames, frame_delay, current_frame + 1
            ))
        else:
            # Animation complete, switch back to pack layout
            self.result_label.place_forget()
            self.result_label.pack(pady=10)
    
    def upload_photo(self):
        file_path = filedialog.askopenfilename(
            title="Select Profile Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                # Open and resize image for display
                image = Image.open(file_path)
                image = image.resize((120, 120), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image)
                
                # Update label
                self.photo_label.configure(image=photo, text="")
                self.photo_label.image = photo  # Keep reference
                
                # Classify the image
                classification = self.classify_image(file_path)
                self.classification_label.configure(text=f"Image: {classification}")
                
                # Show popup and set professional bonus
                label_name = classification.split('(')[0].strip()
                messagebox.showinfo(
                    "AI Analysis Complete",
                    f"AI Analysis Complete! I see {label_name}. This professional look adds a 5% bonus to the predicted salary!"
                )
                self.professional_bonus = True
                
            except Exception as e:
                self.photo_label.configure(text="Error Loading Image")
                self.classification_label.configure(text="Image: Classification failed")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = SalaryPredictorGUI()
    app.run()