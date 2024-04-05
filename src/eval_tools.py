import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as  metrics
import matplotlib.pyplot as plt
import torch

class LossTracker():
    def __init__(self):
        self.losses = {'train': [], 'validation': [], 'validation_accuracy': []}

    def add_loss(self, loss_type, loss):
        if loss_type not in self.losses:
            raise ValueError("Invalid loss type. Use 'train' or 'validation'.")
        self.losses[loss_type].append(loss)

    def get_losses(self, loss_type):
        if loss_type not in self.losses:
            raise ValueError("Invalid loss type. Use 'train' or 'validation'.")
        return self.losses[loss_type]

    def reset(self):
        self.losses = {'train': [], 'validation': []}

    def plot_loss_curves(self):
        fig, ax = plt.subplots()
        plt.grid()
        ax.plot(self.losses['train'], label='train loss')
        ax.plot(self.losses['validation'], label='validation_loss')
        ax.legend()




class GroundTruthStatsClass:
  def __init__(self, model,dataset) -> None:
     self.model = model
     self.dataset = dataset
     self.y_pred = np.array([])
     self.y_true = np.array([])
     for (image, labels) in dataset:
        image = image.reshape(1,image.shape[0], image.shape[1], image.shape[2]) 
        y_pi= torch.flatten(self.model(image)).detach().numpy()
        y_ti = torch.flatten(labels).detach().numpy()
        self.y_pred = np.append(self.y_pred,y_pi) 
        self.y_true = np.append(self.y_true,y_ti) 
        #keep that ram low
        del y_pi
        del y_ti
  def get_pr_numbers(self):
    return metrics.precision_recall_curve(self.y_true, self.y_pred)
  def get_roc_numbers(self):
    return metrics.roc_curve(self.y_true, self.y_pred)