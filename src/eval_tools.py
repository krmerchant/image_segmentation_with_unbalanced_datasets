import matplotlib.pyplot as plt

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
