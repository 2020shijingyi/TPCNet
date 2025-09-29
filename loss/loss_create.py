from loss.losses import *

def init_loss(opt):
    L1_weight = opt['L1_weight']
    D_weight = opt['D_weight']
    E_weight =  opt['E_weight']
    P_weight = opt['P_weight']

    L1_loss = L1Loss(loss_weight=L1_weight, reduction='mean')
    D_loss = SSIM(weight=D_weight)
    E_loss = EdgeLoss(loss_weight=E_weight)
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1}, perceptual_weight=P_weight,
                            criterion='mse')
    return L1_loss, P_loss, E_loss, D_loss