import torch.nn as nn

class DICELoss(nn.Module):
    """TODO Put into function"""
    def __init__(self):
        super(DICELoss, self).__init__()
        self.eps = 1e-6

    def forward(self, y_pred, y_true):
        # Flatten the input tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        numerator = 2*(y_pred * y_true).sum()
        denominator = y_pred.sum() + y_true.sum()

        dice = numerator / (denominator + self.eps)
        return 1 - dice 
