import torch
import torch.nn.functional as F
import numpy as np

class RobustTraining:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def fgsm_attack(self, images, labels, epsilon=0.007):
        """
        Fast Gradient Sign Method attack
        Args:
            images: Input images
            labels: True labels
            epsilon: Attack strength
        """
        # Create a copy of the input
        perturbed_images = images.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(perturbed_images)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Create perturbation
        perturbation = epsilon * perturbed_images.grad.data.sign()
        
        # Add perturbation to images
        perturbed_images = perturbed_images + perturbation
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images.detach()
    
    def mixup(self, images, labels, alpha=0.2):
        """
        Mixup augmentation
        Args:
            images: Input images
            labels: True labels
            alpha: Mixup strength parameter
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = images.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_images = lam * images + (1 - lam) * images[index, :]
        label_a, label_b = labels, labels[index]
        
        return mixed_images, label_a, label_b, lam
    
    def train_step(self, inputs, labels, optimizer, criterion, mixup_alpha=0.2, adv_epsilon=0.007, return_outputs=True):
        """
        Single training step with both mixup and adversarial training
        """
        self.model.train()
        
        # Regular training
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        
        # Adversarial training
        perturbed_images = self.fgsm_attack(inputs, labels, epsilon=adv_epsilon)
        outputs_adv = self.model(perturbed_images)
        loss_adv = criterion(outputs_adv, labels)
        
        # Mixup training
        mixed_images, labels_a, labels_b, lam = self.mixup(inputs, labels, alpha=mixup_alpha)
        outputs_mixed = self.model(mixed_images)
        loss_mixup = (lam * criterion(outputs_mixed, labels_a) + 
                     (1 - lam) * criterion(outputs_mixed, labels_b))
        
        # Combined loss
        total_loss = (loss + 0.5 * loss_adv + 0.5 * loss_mixup) / 2
        
        # Optimization step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if return_outputs:
            return total_loss.item(), outputs
        return total_loss.item() 