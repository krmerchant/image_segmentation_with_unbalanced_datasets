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

