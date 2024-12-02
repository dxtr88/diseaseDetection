import torch
import matplotlib.pyplot as plt
import seaborn as sns
from load import PlantDiseasePredictor
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from PIL import Image
import pandas as pd
from torchvision import transforms
from collections import defaultdict

class ModelAnalyzer:
    def __init__(self, model_path='best_plant_disease_model.pth'):
        """Initialize the analyzer with a trained model"""
        self.predictor = PlantDiseasePredictor(model_path)
        self.device = self.predictor.device
        self.model = self.predictor.model
        self.classes = self.predictor.classes
        
        # Create results directory
        self.results_dir = "model_analysis_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def analyze_model_architecture(self):
        """Analyze and save model architecture details"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        architecture_info = (
            f"Model Architecture Analysis\n"
            f"===========================\n"
            f"Model Type: MobileNetV2\n"
            f"Total Parameters: {total_params:,}\n"
            f"Trainable Parameters: {trainable_params:,}\n"
            f"Number of Classes: {len(self.classes)}\n"
            f"Device: {self.device}\n"
        )
        
        # Save architecture info
        with open(os.path.join(self.results_dir, "model_architecture.txt"), "w") as f:
            f.write(architecture_info)
        
        print(architecture_info)
    
    def analyze_class_distribution(self, data_dir):
        """Analyze and visualize class distribution"""
        class_counts = defaultdict(int)
        
        # Count images per class
        for class_name in self.classes:
            class_path = os.path.join(data_dir, class_name)
            if os.path.exists(class_path):
                class_counts[class_name] = len([f for f in os.listdir(class_path) 
                                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Create distribution plot
        plt.figure(figsize=(15, 8))
        plt.bar(range(len(class_counts)), list(class_counts.values()))
        plt.xticks(range(len(class_counts)), list(class_counts.keys()), rotation=45, ha='right')
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'class_distribution.png'))
        plt.close()
        
        # Save distribution data
        df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
        df.to_csv(os.path.join(self.results_dir, 'class_distribution.csv'), index=False)
        
    def analyze_model_performance(self, test_dir):
        """Analyze model performance on test set"""
        all_predictions = []
        all_confidences = []
        all_true_labels = []
        
        # Process test images
        for class_name in os.listdir(test_dir):
            class_path = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            for img_name in os.listdir(class_path):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(class_path, img_name)
                pred_class, confidence, _ = self.predictor.predict_image(img_path, show_image=False)
                
                all_predictions.append(pred_class)
                all_confidences.append(confidence)
                all_true_labels.append(class_name)
        
        # Create confusion matrix
        cm = confusion_matrix(all_true_labels, all_predictions)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes,
                   yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(all_confidences, bins=50)
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confidence_distribution.png'))
        plt.close()
        
        # Per-class accuracy
        class_accuracy = defaultdict(list)
        for true, pred in zip(all_true_labels, all_predictions):
            class_accuracy[true].append(true == pred)
            
        accuracies = {cls: np.mean(vals) for cls, vals in class_accuracy.items()}
        
        plt.figure(figsize=(15, 6))
        plt.bar(accuracies.keys(), accuracies.values())
        plt.title('Per-class Accuracy')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'per_class_accuracy.png'))
        plt.close()
        
    def analyze_model_robustness(self, sample_image_path):
        """Analyze model robustness to various image transformations"""
        transforms_list = {
            'original': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            'brightness': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.5),
                transforms.ToTensor(),
            ]),
            'contrast': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(contrast=0.5),
                transforms.ToTensor(),
            ]),
            'blur': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.GaussianBlur(kernel_size=5),
                transforms.ToTensor(),
            ]),
            'noise': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        }
        
        results = {}
        image = Image.open(sample_image_path).convert('RGB')
        
        plt.figure(figsize=(15, 5))
        for idx, (name, transform) in enumerate(transforms_list.items()):
            # Apply transformation
            transformed_image = transform(image)
            if name == 'noise':
                # Add random noise
                noise = torch.randn_like(transformed_image) * 0.1
                transformed_image = torch.clamp(transformed_image + noise, 0, 1)
            
            # Get prediction
            transformed_image = transforms.Normalize(
                [0.485, 0.456, 0.406], 
                [0.229, 0.224, 0.225]
            )(transformed_image)
            
            with torch.no_grad():
                output = self.model(transformed_image.unsqueeze(0).to(self.device))
                prob = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted = torch.max(prob, 0)
                
            results[name] = {
                'predicted': self.classes[predicted.item()],
                'confidence': confidence.item()
            }
            
            # Plot
            plt.subplot(1, 5, idx+1)
            plt.imshow(transforms.ToPILImage()(transform(image)))
            plt.title(f'{name}\n{results[name]["confidence"]:.2f}')
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'robustness_analysis.png'))
        plt.close()
        
        # Save results
        with open(os.path.join(self.results_dir, "robustness_analysis.txt"), "w") as f:
            f.write("Robustness Analysis Results\n")
            f.write("==========================\n")
            for name, result in results.items():
                f.write(f"\n{name.capitalize()}:\n")
                f.write(f"Predicted: {result['predicted']}\n")
                f.write(f"Confidence: {result['confidence']:.4f}\n")

def main():
    analyzer = ModelAnalyzer()
    
    # Analyze model architecture
    print("Analyzing model architecture...")
    analyzer.analyze_model_architecture()
    
    # Analyze class distribution
    print("\nAnalyzing class distribution...")
    data_dir = "data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
    analyzer.analyze_class_distribution(data_dir)
    
    # Analyze model performance
    print("\nAnalyzing model performance...")
    test_dir = "data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"
    analyzer.analyze_model_performance(test_dir)
    
    # Analyze model robustness
    print("\nAnalyzing model robustness...")
    sample_image_path = "path/to/sample/image.jpg"  # Update this path
    if os.path.exists(sample_image_path):
        analyzer.analyze_model_robustness(sample_image_path)
    
    print(f"\nAnalysis complete! Results saved in {analyzer.results_dir}/")

if __name__ == "__main__":
    main() 