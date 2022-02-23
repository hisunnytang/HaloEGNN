import torch

def subtract_the_boundary(position, node_mask, box_size=205000):
    xx = remove_mean_with_mask(position, node_mask)
    batch_idx, pos_indx = torch.where(xx.std(1).log() > 8)
    if len(batch_idx) == 0: return
    position[batch_idx, :, pos_indx] -=   ((position[batch_idx, :, pos_indx] > box_size/2) & node_mask[batch_idx, :,0])*box_size

