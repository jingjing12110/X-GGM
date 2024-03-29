import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


# class SimpleClassifier(nn.Module):
#     def __init__(self, in_dim, hid_dim, out_dim, dropout):
#         super(SimpleClassifier, self).__init__()
#         layers = [
#             weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
#         ]
#         self.main = nn.Sequential(*layers)
#
#     def forward(self, x):
#         logits = self.main(x)
#         return logits

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
