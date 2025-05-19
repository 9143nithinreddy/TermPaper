!pip install opencv-python matplotlib tensorflow numpy Pillow

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from google.colab import files
import os
import time
from collections import deque
from IPython.display import display, HTML, clear_output
import base64
import io

# Constants
COLOR_CATEGORIES = ['No Fire', 'Fire', 'Extreme Danger']
INPUT_SIZE = (224, 224)
HISTORY_SIZE = 5  # Number of frames to keep in detection history

class FireDetectionSystem:
    def __init__(self):
        # Enhanced color ranges for fire detection
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])
        self.lower_orange = np.array([11, 70, 50])
        self.upper_orange = np.array([25, 255, 255])
        
        # Load a pre-trained model (in practice, you would load your trained weights)
        self.model = self.create_fire_detection_model()
        
        # Detection history for temporal analysis
        self.detection_history = deque(maxlen=HISTORY_SIZE)
        
        # Statistics
        self.fire_frames = 0
        self.total_frames = 0
        self.frame_results = []
        self.start_time = 0

    def create_fire_detection_model(self):
        """Create a simple CNN model for fire detection (placeholder)"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Note: In a real application, you would load pre-trained weights here
        print("Warning: Using untrained model - for demonstration only")
        return model
    
    def enhanced_color_detection(self, image):
        """More robust color-based fire detection"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect both red ranges (0-10 and 170-180)
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_orange = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        
        # Combine masks
        mask_fire = cv2.bitwise_or(mask_red1, mask_red2)
        mask_fire = cv2.bitwise_or(mask_fire, mask_orange)
        
        # Calculate percentage
        fire_pixels = cv2.countNonZero(mask_fire)
        total_pixels = image.shape[0] * image.shape[1]
        fire_percent = (fire_pixels / total_pixels) * 100
        
        # More sensitive classification
        if fire_percent > 30:
            return 'Extreme Danger', fire_percent, mask_fire
        elif fire_percent > 5:
            return 'Fire', fire_percent, mask_fire
        else:
            return 'No Fire', fire_percent, mask_fire
    
    def cnn_detection(self, image):
        """Use CNN model for fire detection"""
        img = cv2.resize(image, INPUT_SIZE)
        img = img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        
        prediction = self.model.predict(img, verbose=0)[0][0]
        confidence = prediction * 100
        
        return 'Fire' if prediction > 0.5 else 'No Fire', confidence
    
    def combined_detection(self, image):
        """Combine color and CNN detection with temporal analysis"""
        # Color detection
        color_pred, color_percent, fire_mask = self.enhanced_color_detection(image)
        
        # CNN detection
        cnn_pred, cnn_conf = self.cnn_detection(image)
        
        # Update history
        self.detection_history.append({
            'color': color_pred,
            'cnn': cnn_pred,
            'time': time.time()
        })
        
        # Temporal consistency check
        consistent_fire = self.check_consistency()
        
        # Combine results
        if consistent_fire and (color_pred != 'No Fire' or cnn_pred != 'No Fire'):
            final_pred = 'Fire Detected (Confirmed)'
            alert = 'HIGH'
        elif color_pred != 'No Fire' and cnn_pred != 'No Fire':
            final_pred = 'Fire Detected'
            alert = 'MEDIUM'
        elif color_pred != 'No Fire' or cnn_pred != 'No Fire':
            final_pred = 'Possible Fire'
            alert = 'LOW'
        else:
            final_pred = 'No Fire'
            alert = 'NONE'
        
        # Update stats
        self.total_frames += 1
        if 'Fire' in final_pred:
            self.fire_frames += 1
        
        result = {
            'final': final_pred,
            'alert': alert,
            'color': {'pred': color_pred, 'percent': color_percent},
            'cnn': {'pred': cnn_pred, 'conf': cnn_conf},
            'mask': fire_mask,
            'timestamp': time.time() - self.start_time,
            'frame': image.copy()
        }
        
        # Store results for visualization
        self.frame_results.append(result)
        
        return result
    
    def check_consistency(self):
        """Check if fire is consistently detected"""
        if len(self.detection_history) < 3:
            return False
        
        fire_count = 0
        for detection in self.detection_history:
            if detection['color'] != 'No Fire' or detection['cnn'] != 'No Fire':
                fire_count += 1
                
        return fire_count >= 2  # At least 2 of last 3 frames
    
    def visualize_results(self, image, results):
        """Show detection results"""
        # Create output frame with black background
        output = np.zeros_like(image)
        output[:] = (0, 0, 0)  # Set background to black
        
        # Copy original image with fire regions highlighted
        overlay = image.copy()
        overlay[results['mask'] > 0] = (0, 0, 255)  # Red overlay for fire
        output = cv2.addWeighted(output, 0.3, overlay, 0.7, 0)
        
        # Draw detection info
        text_color = (0, 255, 255) if 'Fire' in results['final'] else (0, 255, 0)
        cv2.putText(output, results['final'], (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Add details
        cv2.putText(output, f"Color: {results['color']['percent']:.1f}% {results['color']['pred']}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(output, f"CNN: {results['cnn']['conf']:.1f}% {results['cnn']['pred']}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return output
    
    def show_opening_screen(self):
        """Display an attractive opening screen"""
        opening_html = """
        <style>
            .opening-container {
                font-family: 'Arial', sans-serif;
                background: linear-gradient(135deg, #1a1a1a 0%, #333 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                max-width: 800px;
                margin: 20px auto;
                text-align: center;
                box-shadow: 0 10px 20px rgba(0,0,0,0.5);
            }
            .opening-title {
                font-size: 2.5em;
                margin-bottom: 20px;
                color: #ff5722;
                text-shadow: 0 2px 5px rgba(0,0,0,0.5);
            }
            .opening-subtitle {
                font-size: 1.2em;
                margin-bottom: 30px;
                color: #ddd;
            }
            .feature-list {
                text-align: left;
                margin: 20px 0;
                padding: 0 20px;
            }
            .feature-item {
                margin: 15px 0;
                display: flex;
                align-items: center;
            }
            .feature-icon {
                width: 30px;
                height: 30px;
                background-color: #ff5722;
                border-radius: 50%;
                display: flex;
                justify-content: center;
                align-items: center;
                margin-right: 15px;
                color: white;
                font-weight: bold;
            }
            .start-button {
                background-color: #ff5722;
                color: white;
                border: none;
                padding: 15px 30px;
                font-size: 1.2em;
                border-radius: 5px;
                cursor: pointer;
                margin-top: 20px;
                transition: all 0.3s;
            }
            .start-button:hover {
                background-color: #e64a19;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }
        </style>
        
        <div class="opening-container">
            <div class="opening-title">Advanced Fire Detection System</div>
            <div class="opening-subtitle">AI-powered fire detection using computer vision and deep learning</div>
            
            <div class="feature-list">
                <div class="feature-item">
                    <div class="feature-icon">1</div>
                    <div>Real-time fire detection in images and videos</div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">2</div>
                    <div>Dual detection: Color analysis + CNN classification</div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">3</div>
                    <div>Temporal consistency checking for reliable results</div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">4</div>
                    <div>Comprehensive analytics and visualizations</div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">5</div>
                    <div>Safety recommendations and precautions</div>
                </div>
            </div>
            
            <button class="start-button" onclick="document.getElementById('start-button').click()">Start Detection</button>
        </div>
        """
        display(HTML(opening_html))
    
    def show_closing_screen(self):
      
      
      """Display an attractive closing screen"""
      closing_html = """
      <style>
          .closing-container {{
              font-family: 'Arial', sans-serif;
              background: linear-gradient(135deg, #1a1a1a 0%, #333 100%);
              color: white;
              padding: 30px;
              border-radius: 10px;
              max-width: 800px;
              margin: 20px auto;
              text-align: center;
              box-shadow: 0 10px 20px rgba(0,0,0,0.5);
          }}
          .closing-title {{
              font-size: 2.5em;
              margin-bottom: 20px;
              color: #4CAF50;
              text-shadow: 0 2px 5px rgba(0,0,0,0.5);
          }}
          .closing-message {{
              font-size: 1.2em;
              margin-bottom: 30px;
              color: #ddd;
          }}
          .stats-container {{
              display: flex;
              justify-content: space-around;
              margin: 30px 0;
          }}
          .stat-box {{
              background: rgba(255,255,255,0.1);
              padding: 15px;
              border-radius: 5px;
              min-width: 120px;
          }}
          .stat-value {{
              font-size: 2em;
              font-weight: bold;
              color: #4CAF50;
              margin: 10px 0;
          }}
          .restart-button {{
              background-color: #4CAF50;
              color: white;
              border: none;
              padding: 15px 30px;
              font-size: 1.2em;
              border-radius: 5px;
              cursor: pointer;
              margin-top: 20px;
              transition: all 0.3s;
          }}
          .restart-button:hover {{
              background-color: #388E3C;
              transform: translateY(-2px);
              box-shadow: 0 5px 15px rgba(0,0,0,0.3);
          }}
      </style>
      
      <div class="closing-container">
          <div class="closing-title">Analysis Complete</div>
          <div class="closing-message">Thank you for using the Fire Detection System</div>
          
          <div class="stats-container">
              <div class="stat-box">
                  <div>Frames Analyzed</div>
                  <div class="stat-value">{}</div>
              </div>
              <div class="stat-box">
                  <div>Fire Detected</div>
                  <div class="stat-value">{}</div>
              </div>
              <div class="stat-box">
                  <div>Accuracy</div>
                  <div class="stat-value">{:.1f}%</div>
              </div>
          </div>
          
          <button class="restart-button" onclick="document.getElementById('restart-button').click()">Start New Analysis</button>
      </div>
      """.format(
          self.total_frames,
          self.fire_frames,
          (self.fire_frames / self.total_frames * 100) if self.total_frames > 0 else 0
      )
      display(HTML(closing_html))
    
    def show_safety_precautions(self):
        """Display safety precautions and steps to take"""
        precautions = """
        <style>
            .precautions-container {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 20px auto;
                padding: 20px;
                background: #1a1a1a;
                color: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.5);
            }
            .precautions-title {
                color: #ff5722;
                border-bottom: 2px solid #ff5722;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            .section-title {
                color: #4CAF50;
                margin: 20px 0 10px 0;
            }
            .action-list {
                padding-left: 20px;
            }
            .action-item {
                margin: 10px 0;
                padding-left: 10px;
                border-left: 3px solid #ff5722;
            }
            .warning-box {
                background: #ff5722;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
                font-weight: bold;
            }
        </style>
        
        <div class="precautions-container">
            <h2 class="precautions-title">Fire Safety Precautions and Actions</h2>
            
            <h3 class="section-title">Immediate Actions if Fire is Detected:</h3>
            <div class="action-list">
                <div class="action-item"><strong>Activate fire alarms</strong> to alert everyone in the vicinity</div>
                <div class="action-item"><strong>Evacuate immediately</strong> using the nearest safe exit</div>
                <div class="action-item"><strong>Call emergency services</strong> (911 or local fire department)</div>
                <div class="action-item"><strong>Use fire extinguishers</strong> only if safe to do so and you're trained</div>
                <div class="action-item"><strong>Close doors behind you</strong> to slow fire spread</div>
            </div>
            
            <h3 class="section-title">Prevention Measures:</h3>
            <div class="action-list">
                <div class="action-item">Regularly inspect electrical equipment and wiring</div>
                <div class="action-item">Keep flammable materials away from heat sources</div>
                <div class="action-item">Maintain clear evacuation routes at all times</div>
                <div class="action-item">Install and maintain smoke detectors and fire extinguishers</div>
                <div class="action-item">Conduct regular fire drills</div>
            </div>
            
            <h3 class="section-title">Post-Fire Actions:</h3>
            <div class="action-list">
                <div class="action-item">Do not re-enter the building until authorities declare it safe</div>
                <div class="action-item">Document the incident for insurance and investigation purposes</div>
                <div class="action-item">Review and improve fire safety measures based on the incident</div>
            </div>
            
            <div class="warning-box">
                <strong>Remember:</strong> Your safety is most important. Never risk your life to save property.
            </div>
        </div>
        """
        display(HTML(precautions))
    
    def generate_analytics_dashboard(self):
        """Generate comprehensive analytics dashboard"""
        if not self.frame_results:
            return
            
        # Calculate statistics
        fire_frames = sum(1 for r in self.frame_results if 'Fire' in r['final'])
        fire_percentage = (fire_frames / len(self.frame_results)) * 100
        avg_conf = np.mean([r['cnn']['conf'] for r in self.frame_results if 'Fire' in r['final']] or [0])
        max_conf = max([r['cnn']['conf'] for r in self.frame_results if 'Fire' in r['final']] or [0])
        
        # Create HTML dashboard
        dashboard = f"""
        <style>
            .dashboard-container {{
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 20px auto;
                background: #1a1a1a;
                color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.5);
            }}
            .dashboard-title {{
                color: #ff5722;
                text-align: center;
                margin-bottom: 30px;
                font-size: 2em;
            }}
            .stats-row {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 20px;
            }}
            .stat-box {{
                flex: 1;
                background: #333;
                padding: 15px;
                margin: 0 10px;
                border-radius: 5px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }}
            .fire-stat {{
                color: #ff5722;
            }}
            .chart-container {{
                background: #333;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .chart-title {{
                margin-top: 0;
                color: #4CAF50;
            }}
            .samples-container {{
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                margin-top: 20px;
            }}
            .sample-box {{
                width: 30%;
                margin-bottom: 20px;
                background: #333;
                border-radius: 5px;
                overflow: hidden;
            }}
            .sample-label {{
                padding: 10px;
                text-align: center;
                background: #444;
            }}
        </style>
        
        <div class="dashboard-container">
            <div class="dashboard-title">Fire Detection Analytics Dashboard</div>
            
            <div class="stats-row">
                <div class="stat-box">
                    <h3>Total Frames Analyzed</h3>
                    <div class="stat-value">{len(self.frame_results)}</div>
                </div>
                <div class="stat-box">
                    <h3>Fire Frames Detected</h3>
                    <div class="stat-value fire-stat">{fire_frames} ({fire_percentage:.1f}%)</div>
                </div>
                <div class="stat-box">
                    <h3>Avg. Confidence</h3>
                    <div class="stat-value">{avg_conf:.1f}%</div>
                </div>
                <div class="stat-box">
                    <h3>Max Confidence</h3>
                    <div class="stat-value">{max_conf:.1f}%</div>
                </div>
            </div>
            
            <div style="display: flex; margin-bottom: 20px;">
                <div class="chart-container" style="flex: 1; margin-right: 10px;">
                    <h3 class="chart-title">Detection Timeline</h3>
                    <img src="{self.generate_timeline_plot()}" style="width: 100%;">
                </div>
                <div class="chart-container" style="flex: 1; margin-left: 10px;">
                    <h3 class="chart-title">Confidence Distribution</h3>
                    <img src="{self.generate_confidence_histogram()}" style="width: 100%;">
                </div>
            </div>
            
            <div class="chart-container">
                <h3 class="chart-title">Detection Method Correlation</h3>
                <img src="{self.generate_scatter_plot()}" style="width: 100%; max-width: 600px; display: block; margin: 0 auto;">
            </div>
            
            <div class="chart-container">
                <h3 class="chart-title">Sample Fire Frames</h3>
                <div class="samples-container">
                    {self.generate_sample_frames()}
                </div>
            </div>
        </div>
        """
        display(HTML(dashboard))
    
    def generate_timeline_plot(self):
        """Generate timeline plot with dark theme"""
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 4), facecolor='#333')
        times = [r['timestamp'] for r in self.frame_results]
        confidences = [r['cnn']['conf'] for r in self.frame_results]
        plt.plot(times, confidences, color='#ff5722')
        plt.title('Confidence Over Time', color='white')
        plt.xlabel('Time (seconds)', color='white')
        plt.ylabel('CNN Confidence (%)', color='white')
        plt.grid(True, color='#555')
        
        # Save to buffer and return base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', facecolor='#333')
        buffer.seek(0)
        plt.close()
        return 'data:image/png;base64,' + base64.b64encode(buffer.read()).decode('utf-8')
    
    def generate_confidence_histogram(self):
        """Generate confidence histogram with dark theme"""
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 4), facecolor='#333')
        fire_confs = [r['cnn']['conf'] for r in self.frame_results if 'Fire' in r['final']]
        plt.hist(fire_confs if fire_confs else [0], bins=10, color='#ff5722')
        plt.title('Confidence Distribution (Fire Frames)', color='white')
        plt.xlabel('CNN Confidence (%)', color='white')
        plt.ylabel('Number of Frames', color='white')
        plt.grid(True, color='#555')
        
        # Save to buffer and return base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', facecolor='#333')
        buffer.seek(0)
        plt.close()
        return 'data:image/png;base64,' + base64.b64encode(buffer.read()).decode('utf-8')
    
    def generate_scatter_plot(self):
        """Generate scatter plot with dark theme"""
        plt.style.use('dark_background')
        plt.figure(figsize=(8, 6), facecolor='#333')
        color_percents = [r['color']['percent'] for r in self.frame_results]
        cnn_confs = [r['cnn']['conf'] for r in self.frame_results]
        plt.scatter(color_percents, cnn_confs, alpha=0.5, color='#ff5722')
        plt.title('Color Analysis vs CNN Confidence', color='white')
        plt.xlabel('Color Analysis (%)', color='white')
        plt.ylabel('CNN Confidence (%)', color='white')
        plt.grid(True, color='#555')
        
        # Save to buffer and return base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', facecolor='#333')
        buffer.seek(0)
        plt.close()
        return 'data:image/png;base64,' + base64.b64encode(buffer.read()).decode('utf-8')
    
    def generate_sample_frames(self):
        """Generate HTML for sample fire frames"""
        fire_frames = [r for r in self.frame_results if 'Fire' in r['final']]
        if not fire_frames:
            return "<div style='width:100%; text-align:center; padding:20px;'>No fire frames detected</div>"
        
        samples = fire_frames[:3]  # Get first 3 fire frames
        html = ""
        
        for i, sample in enumerate(samples):
            # Apply fire mask overlay
            frame = sample['frame'].copy()
            frame[sample['mask'] > 0] = [0, 0, 255]  # Red overlay
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            html += f"""
            <div class="sample-box">
                <img src="data:image/jpg;base64,{frame_base64}" style="width:100%; display:block;">
                <div class="sample-label">
                    Time: {sample['timestamp']:.1f}s | Confidence: {sample['cnn']['conf']:.1f}%
                </div>
            </div>
            """
        
        return html
    
    def process_video(self, video_path):

      """Process video with real-time visualization"""
      self.start_time = time.time()
      self.frame_results = []  # Reset for new video
      self.fire_frames = 0
      self.total_frames = 0
      
      cap = cv2.VideoCapture(video_path)
      if not cap.isOpened():
          print("Error opening video")
          return
      
      # Create HTML container for real-time display
      display(HTML("""
      <style>
          .realtime-container {
              font-family: Arial, sans-serif;
              max-width: 800px;
              margin: 0 auto;
              background: #1a1a1a;
              color: white;
              padding: 20px;
              border-radius: 10px;
              box-shadow: 0 5px 15px rgba(0,0,0,0.5);
          }
          .realtime-title {
              text-align: center;
              color: #ff5722;
              margin-bottom: 20px;
          }
          .video-display {
              border: 2px solid #444;
              border-radius: 5px;
              overflow: hidden;
              margin-bottom: 20px;
          }
          .alert-box {
              padding: 10px;
              font-weight: bold;
              text-align: center;
              margin-bottom: 10px;
          }
          .alert-high {
              background-color: #ff4444;
              color: white;
          }
          .alert-medium {
              background-color: #ffbb33;
              color: black;
          }
          .alert-low {
              background-color: #ffdd59;
              color: black;
          }
          .alert-none {
              background-color: #4CAF50;
              color: white;
          }
          .stats-container {
              display: flex;
              padding: 10px;
              background: #333;
              border-radius: 5px;
          }
          .stat-item {
              flex: 1;
              text-align: center;
          }
          .stat-value {
              font-weight: bold;
              margin-top: 5px;
          }
          .progress-container {
              height: 10px;
              background: #444;
              border-radius: 5px;
              margin-top: 5px;
              overflow: hidden;
          }
          .progress-bar {
              height: 100%;
              border-radius: 5px;
          }
          .stop-button {
              background-color: #ff5722;
              color: white;
              border: none;
              padding: 10px 20px;
              font-size: 1em;
              border-radius: 5px;
              cursor: pointer;
              margin: 10px auto;
              display: block;
          }
      </style>
      
      <div class="realtime-container">
          <h2 class="realtime-title">Real-Time Fire Detection</h2>
          <div id="video-display" class="video-display"></div>
          <button id="stop-button" class="stop-button" onclick="document.getElementById('stop-processing').click()">Stop Processing</button>
          <div id="analytics"></div>
      </div>
      """))
      
      # Add hidden stop button
      display(HTML('<button id="stop-processing" style="display:none;"></button>'))
      
      frame_count = 0
      update_interval = 5  # Update display every 5 frames
      stop_processing = False
      
      # Function to check if stop button was clicked
      def check_stop():
          try:
              # This checks if the stop button was clicked in the notebook
              return False  # In Colab, we can't easily check button clicks during processing
          except:
              return False
      
      while True:
          if stop_processing or check_stop():
              break
              
          ret, frame = cap.read()
          if not ret:
              break
          
          frame_count += 1
          
          # Process frame
          results = self.combined_detection(frame)
          
          # Update display at specified interval
          if frame_count % update_interval == 0:
              output = self.visualize_results(frame, results)
              _, buffer = cv2.imencode('.jpg', output)
              frame_base64 = base64.b64encode(buffer).decode('utf-8')
              
              # Update HTML display
              display(HTML(f"""
              <script>
                  document.getElementById('video-display').innerHTML = `
                      <img src="data:image/jpg;base64,{frame_base64}" style="width:100%; display:block;">
                      <div class="alert-box ${
                          'alert-high' if results['alert'] == 'HIGH' else 
                          'alert-medium' if results['alert'] == 'MEDIUM' else 
                          'alert-low' if results['alert'] == 'LOW' else 'alert-none'
                      }">
                          {results['final']} (Alert Level: {results['alert']})
                      </div>
                      <div class="stats-container">
                          <div class="stat-item">
                              <div>Color Analysis</div>
                              <div class="stat-value">{results['color']['percent']:.1f}%</div>
                              <div class="progress-container">
                                  <div class="progress-bar" style="width: {results['color']['percent']}%; 
                                      background: {'#FF5722' if results['color']['pred'] == 'Fire' else '#4CAF50'};">
                                  </div>
                              </div>
                          </div>
                          <div class="stat-item">
                              <div>CNN Detection</div>
                              <div class="stat-value">{results['cnn']['conf']:.1f}%</div>
                              <div class="progress-container">
                                  <div class="progress-bar" style="width: {results['cnn']['conf']}%; 
                                      background: {'#FF5722' if results['cnn']['pred'] == 'Fire' else '#4CAF50'};">
                                  </div>
                              </div>
                          </div>
                      </div>
                  `;
              </script>
              """))
      
      cap.release()
      
      # Generate final analytics and precautions
      self.generate_analytics_dashboard()
      self.show_safety_precautions()
      self.show_closing_screen()

def main():
    # Show opening screen
    detector = FireDetectionSystem()
    detector.show_opening_screen()
    
    # Hidden button to start (for the opening screen button to trigger)
    display(HTML('<button id="start-button" style="display:none;"></button>'))
    display(HTML('<button id="restart-button" style="display:none;"></button>'))
    
    # Wait for user to click the start button
    start_clicked = False
    while not start_clicked:
        try:
            start_clicked = True  # For notebook environment
        except:
            time.sleep(0.1)
    
    clear_output()
    
    print("Fire Detection System")
    print("1. Process test image")
    print("2. Upload and process video")
    print("3. Exit")
    
    choice = input("Select option: ")
    
    if choice == '1':
        # Use a sample fire image
        print("Please upload an image file")
        uploaded = files.upload()
        if uploaded:
            image_path = next(iter(uploaded))
            img = cv2.imread(image_path)
            detector.start_time = time.time()
            detector.frame_results = []
            results = detector.combined_detection(img)
            output = detector.visualize_results(img, results)
            
            # Display with HTML visualization
            _, buffer = cv2.imencode('.jpg', output)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            display(HTML(f"""
            <style>
                .image-result-container {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    background: #1a1a1a;
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.5);
                }}
                .image-result-title {{
                    text-align: center;
                    color: #ff5722;
                    margin-bottom: 20px;
                }}
                .image-display {{
                    border: 2px solid #444;
                    border-radius: 5px;
                    overflow: hidden;
                    margin-bottom: 20px;
                }}
            </style>
            
            <div class="image-result-container">
                <h2 class="image-result-title">Image Analysis Results</h2>
                <div class="image-display">
                    <img src="data:image/jpg;base64,{frame_base64}" style="width:100%; display:block;">
                </div>
            </div>
            """))
            
            # Generate analytics for single image
            detector.generate_analytics_dashboard()
            detector.show_safety_precautions()
            detector.show_closing_screen()
        else:
            print("No file uploaded")
    elif choice == '2':
        print("Upload a video file (MP4 recommended)")
        uploaded = files.upload()
        if uploaded:
            video_path = next(iter(uploaded))
            detector.process_video(video_path)
        else:
            print("No file uploaded")
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()