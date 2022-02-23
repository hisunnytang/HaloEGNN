import torch
from flows.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
        assert_correctly_masked
from utils import subtract_the_boundary
from tqdm import tqdm
import time

def flow_forward(flow,
                 prior,
                 input_graph,
                 input_cond,
                 device='cuda',
                 x_norm=300,
                 ode_regularization=0.01):
    x, h, node_mask, edge_mask = input_graph
    context = input_cond[0]

    x = x.to(device)
    h = h.to(device)
    node_mask = node_mask.to(device)
    edge_mask = edge_mask.to(device)
    context   = context.to(device)

    subtract_the_boundary(x, node_mask)
    xx = remove_mean_with_mask(x, node_mask)

    # h = log_normalize_with_mask(h, node_mask)

    h = (h+1e-8).log()* node_mask
    # dont normalize here
    #context = log_norm_target(context)

    # center the position coordinate
    xx = remove_mean_with_mask(x/x_norm, node_mask)
    assert_mean_zero_with_mask(xx, node_mask)

    bs, n_nodes, n_dims = xx.size()
    # inflate context (condition) in to the shape of [batch, max_nodes, n_context]
    context_ = context.unsqueeze(1).repeat(1, n_nodes, 1)* node_mask
    assert_correctly_masked(context_, node_mask)

    xh = torch.cat([xx, h], dim=2)
    assert_correctly_masked(xh, node_mask)

    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)


    z, delta_logp, reg_term = flow(xh.float(),
                                   node_mask.float(),
                                   edge_mask.float(),
                                   context_.float())
    z_x, z_h = z[:, :, 0:n_dims].clone(), z[:, :, n_dims:].clone()

    assert_correctly_masked(z_x, node_mask)
    assert_correctly_masked(z_h, node_mask)

    N = node_mask.squeeze(2).sum(1).long()

    log_qh_x = 0
    log_pz = prior(z_x, z_h, node_mask)
    log_px = (log_pz + delta_logp - log_qh_x).mean()  # Average over batch.
    reg_term = reg_term.mean()  # Average over batch.
    nll = -log_px

    mean_abs_z = torch.mean(torch.abs(z)).item()
    loss = nll + ode_regularization * reg_term

    return nll, loss, z_x, z_h

def train_step(flow, prior, optim, dl_train, condition_normalizer, device='cuda', ode_regularization=1e-2):
    pbar = tqdm(dl_train)
    count = 0
    total_loss = 0
    for input_graph, input_cond in pbar:
        start = time.time()
        optim.zero_grad()
        input_cond = [ torch.from_numpy(condition_normalizer.transform(input_cond[0])) ]
        nll, loss, _, _ = flow_forward.flow_forward(flow, prior, input_graph, input_cond, device=device, ode_regularization=ode_regularization)
        loss.backward()
        optim.step()
        total_loss += loss.item()
        count += 1
        pbar.set_postfix({"total_loss": total_loss/count} )
        pbar.refresh()
    return total_loss / len(dl_train)


@torch.no_grad()
def val_step(flow, prior, dl_val, condition_normalizer, device='cuda', ode_regularization=1e-2):
    pbar = tqdm(dl_val)
    count = 0
    total_loss = 0
    for input_graph, input_cond in pbar:
        start = time.time()
        input_cond = [ torch.from_numpy(condition_normalizer.transform(input_cond[0])) ]
        nll, loss, _, _ = flow_forward.flow_forward(flow, prior, input_graph, input_cond, device='cuda', ode_regularization=0.01)
        total_loss += loss.item()
        count += 1
        pbar.set_postfix({"total_loss": total_loss/count} )
        pbar.refresh()
    return total_loss / len(dl_val)


