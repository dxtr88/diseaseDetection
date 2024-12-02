import torch
from torchvision import transforms
from PIL import Image
import os
from training import create_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import datetime
from collections import Counter

class PlantDiseasePredictor:
    def __init__(self, model_path='best_plant_disease_model.pth'):
        """
        Initialize the predictor with a trained model
        Args:
            model_path (str): Path to the trained model checkpoint
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Hardcoded class names
        self.classes = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy',
            'Wheat___Healthy',
            'Wheat___septoria',
            'Wheat___stripe_rust'
        ]
        
        self.num_classes = len(self.classes)
        
        # Create and load the model
        self.model = create_model(self.num_classes)
        self.load_model(model_path)
        self.model.eval()
        
        # Recommended transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        """Load the trained model checkpoint"""
        if not os.path.exists(model_path):
            raise ValueError(f"Model checkpoint not found at: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        print(f"Loaded model checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")

    def predict_image(self, image_path, show_image=True, confidence_threshold=0.7):
        """
        Predict the disease class for a single image
        Args:
            image_path (str): Path to the image file
            show_image (bool): Whether to display the image with prediction
            confidence_threshold (float): Minimum confidence threshold for predictions
        Returns:
            tuple: (predicted_class_name, confidence_score, true_label)
        """
        # Load and preprocess the image
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Basic image quality checks
            if image.size[0] < 100 or image.size[1] < 100:
                print("Warning: Image resolution is very low, this might affect prediction quality")
            
            # Apply transforms
            input_tensor = self.transform(image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
            # Get top 3 predictions and confidences
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            # Get prediction and confidence
            confidence, predicted = torch.max(probabilities, 0)
            predicted_class = self.classes[predicted.item()]
            confidence_score = confidence.item()
            
            # Get true label from parent directory name if available
            parent_dir = os.path.basename(os.path.dirname(image_path))
            
            # Display results
            print("\n=== Prediction Results ===")
            print(f"Image: {os.path.basename(image_path)}")
            
            if confidence_score < confidence_threshold:
                print(f"Warning: Low confidence prediction ({confidence_score:.4f})")
                print("Top 3 predictions:")
                for i in range(3):
                    print(f"{self.classes[top_indices[i]]}: {top_probs[i]:.4f}")
            else:
                print(f"Prediction: {predicted_class}")
                print(f"Confidence: {confidence_score:.4f}")
            
            return predicted_class, confidence_score, parent_dir if parent_dir in self.classes else None
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None, 0.0, None

    def predict_directory(self, test_dir, true_labels=None, limit=None, show_metrics=True):
        """
        Predict disease classes for all images in a directory and calculate accuracy metrics
        Args:
            test_dir (str): Path to directory containing test images
            true_labels (list): List of true labels for the test images (optional)
            limit (int): Maximum number of images to process (optional)
            show_metrics (bool): Whether to display detailed metrics
        Returns:
            float: Accuracy score
        """
        if not os.path.exists(test_dir):
            raise ValueError(f"Test directory not found: {test_dir}")
        
        # Get all subdirectories (each represents a class)
        class_dirs = [d for d in os.listdir(test_dir) 
                     if os.path.isdir(os.path.join(test_dir, d))]
        
        if not class_dirs:
            raise ValueError(f"No class directories found in {test_dir}")
        
        all_predictions = []
        all_true_labels = []
        confidences = []
        total_images = 0
        
        # Process images from each class directory
        for class_dir in class_dirs:
            class_path = os.path.join(test_dir, class_dir)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if limit:
                image_files = image_files[:limit]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                try:
                    predicted_class, confidence, true_class = self.predict_image(img_path, show_image=False)
                    if true_class is not None:  # Only add if we have a valid true class
                        all_predictions.append(predicted_class)
                        all_true_labels.append(true_class)
                        confidences.append(confidence)
                        total_images += 1
                        
                        if total_images % 10 == 0:  # Progress update every 10 images
                            print(f"Processed {total_images} images...")
                            print(f"Current accuracy: {accuracy_score(all_true_labels, all_predictions):.4f}")
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue  # Skip this image and continue with the next
        
        # Calculate and display metrics
        if all_predictions:
            # Create results directory if it doesn't exist
            results_dir = "validation_results"
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save numerical results
            accuracy = accuracy_score(all_true_labels, all_predictions)
            results_text = f"\nValidation Results:\n"
            results_text += f"Timestamp: {timestamp}\n"
            results_text += f"Total images processed successfully: {total_images}\n"
            results_text += f"Overall Accuracy: {accuracy:.4f}\n"
            
            if show_metrics:
                # Classification Report
                report = classification_report(all_true_labels, all_predictions)
                results_text += f"\nDetailed Classification Report:\n{report}\n"
                
                # Create abbreviated labels
                def abbreviate_label(label):
                    try:
                        parts = label.split('___')
                        if len(parts) == 2:
                            plant, disease = parts
                            # Take first letter of plant and first letter of each disease word
                            abbr = plant[0] + '_' + ''.join(word[0] for word in disease.split('_'))
                            return abbr
                        else:
                            # If no ___ in label, just take first letters of each word
                            return ''.join(word[0] for word in label.split('_'))
                    except Exception:
                        # Fallback: just take first 3 chars if everything else fails
                        return label[:3] if label else 'UNK'
                
                # Get unique labels in a consistent order
                unique_labels = sorted(set(all_true_labels).union(set(all_predictions)))
                abbr_labels = [abbreviate_label(label) for label in unique_labels]
                
                # Check for duplicate abbreviations and make them unique if necessary
                abbr_count = Counter(abbr_labels)
                for i, (label, abbr) in enumerate(zip(unique_labels, abbr_labels)):
                    if abbr_count[abbr] > 1:
                        # If duplicate exists, add a number to make it unique
                        abbr_labels[i] = f"{abbr}{i+1}"
                
                # Save label mapping
                results_text += "\nLabel Abbreviations:\n"
                for full, abbr in zip(unique_labels, abbr_labels):
                    results_text += f"{abbr}: {full}\n"
                
                # Save all results to text file
                with open(os.path.join(results_dir, f"validation_results_{timestamp}.txt"), 'w') as f:
                    f.write(results_text)
                
                # 1. Confusion Matrix
                plt.figure(figsize=(12, 8))
                cm = confusion_matrix(all_true_labels, all_predictions)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=abbr_labels,
                           yticklabels=abbr_labels)
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.xticks(rotation=45)
                plt.yticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'confusion_matrix_{timestamp}.png'))
                if show_metrics: plt.show()
                plt.close()
                
                # 2. Per-class Accuracy Bar Plot
                plt.figure(figsize=(15, 6))
                class_accuracies = {}
                for true, pred in zip(all_true_labels, all_predictions):
                    if true not in class_accuracies:
                        class_accuracies[true] = {'correct': 0, 'total': 0}
                    class_accuracies[true]['total'] += 1
                    if true == pred:
                        class_accuracies[true]['correct'] += 1
                
                accuracies = [class_accuracies[label]['correct'] / class_accuracies[label]['total'] 
                            for label in unique_labels]
                
                plt.bar(abbr_labels, accuracies)
                plt.title('Per-class Accuracy')
                plt.xlabel('Class')
                plt.ylabel('Accuracy')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'per_class_accuracy_{timestamp}.png'))
                if show_metrics: plt.show()
                plt.close()
                
                # 3. Prediction Distribution
                plt.figure(figsize=(15, 6))
                pred_counts = Counter(all_predictions)
                plt.bar(abbr_labels, [pred_counts[label] for label in unique_labels])
                plt.title('Prediction Distribution')
                plt.xlabel('Class')
                plt.ylabel('Number of Predictions')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'prediction_distribution_{timestamp}.png'))
                if show_metrics: plt.show()
                plt.close()
                
                # 4. Confidence Distribution
                plt.figure(figsize=(10, 6))
                plt.hist(confidences, bins=50)
                plt.title('Confidence Score Distribution')
                plt.xlabel('Confidence Score')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'confidence_distribution_{timestamp}.png'))
                if show_metrics: plt.show()
                plt.close()
            
            print(f"\nResults saved in {results_dir} directory")
            return accuracy
        else:
            print("No predictions were made successfully.")
            return 0.0

def main():
    # Initialize predictor
    predictor = PlantDiseasePredictor()
    
    # Test on validation set
    val_dir = "data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"  # Update this path to your validation directory
    
    if not os.path.exists(val_dir):
        print(f"\nValidation directory not found: {val_dir}")
        print("Please check the path to your validation data.")
        return
    
    print(f"\nRunning validation on {val_dir}...")
    try:
        accuracy = predictor.predict_directory(val_dir, show_metrics=True)
        print(f"\nValidation completed successfully!")
        print(f"Final Accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"Error during validation: {str(e)}")

if __name__ == "__main__":
    main() 
    