from torch import nn
import math

def losses_schedule(epoch):
    return 1/2 - 1/(2*math.exp(epoch/40))

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, *args):
        logits = args[0]
        targets = args[1] 
        distances = args[2] 
        epoch = args[3] 
        weights = args[4] 

        alpha = losses_schedule(epoch)
        bce = (self.bce_loss(logits, targets)*weights).mean()
        dist_loss = distances.min(dim=1)[0].mean()  # Take the minimum distance to any center for each sample and average it
        #print(f"{(1-alpha)} * {bce} + {alpha} * {dist_loss}")
        return (1-alpha) * bce + alpha * dist_loss
    
class RealDistLoss(nn.Module):
    def __init__(self):
        super(RealDistLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, *args):
        logits = args[0]
        targets = args[1] 
        distances = args[2] 
        epoch = args[3] 
        weights = args[4] 

        alpha = losses_schedule(epoch)
        bce = (self.bce_loss(logits, targets)*weights).mean()
        
        distances = distances * weights.unsqueeze(1) # weight the distances for class imbalance
        dist_real = distances[targets == 1][0].mean() # Take the minimum distance to the real class center for real samples and average it
        dist_fake = math.exp(- distances[targets == 0][0].mean()) # Take the maximum distance to the real class center for fake samples and average it
        dist_loss = dist_real + dist_fake
        #print(f"{(1-alpha)} * {bce} + {alpha} * {dist_loss}")
        return (1-alpha) * bce + alpha * dist_loss