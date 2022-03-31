import torch
from .flows.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
        assert_correctly_masked
from .utils import subtract_the_boundary
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

import torch.distributed as dist
def train_step(flow, prior, optim, dl_train, condition_normalizer, device='cuda', ode_regularization=1e-2, transform_input=None):
    #rank = torch.distributed.get_rank()
    #disable = False if rank == 0 else True
    world_size = torch.cuda.device_count()
    disable = False
    pbar = tqdm(dl_train,disable=disable, mininterval=120)
    count = 0
    total_loss = torch.zeros(1, device=device)
    total_nll  = torch.zeros(1, device=device)
    if transform_input is None:
        transform_input = lambda x: x
    for input_graph, input_cond in pbar:
        start = time.time()
        optim.zero_grad()
        input_cond = [ torch.from_numpy(condition_normalizer.transform(input_cond[0])) ]
        input_graph = transform_input(input_graph)
        nll, loss, _, _ = flow_forward(flow, prior, input_graph, input_cond, device=device, ode_regularization=ode_regularization)

        # force syncrhonization every step???
        #if count == 0:
        #world_size = torch.distributed.get_world_size()
        if world_size > 1 and count == 0:
            actnorm = flow.module.transformations[0]
            dist.broadcast(actnorm.h_t.data, src=0, group=dist.new_group(list(range(world_size))))
            dist.broadcast(actnorm.x_log_s.data, src=0, group=dist.new_group(list(range(world_size))))
            dist.broadcast(actnorm.h_log_s.data, src=0, group=dist.new_group(list(range(world_size))))
            dist.barrier()

        loss.backward()
        optim.step()
        total_loss += loss.item()
        total_nll  += nll.item()
        count += 1
        pbar.set_postfix({"train_loss": total_loss.item()/count, "nll": total_nll.item()/count} )
    return total_loss / len(dl_train)


@torch.no_grad()
def val_step(flow, prior, dl_val, condition_normalizer, device='cuda', ode_regularization=1e-2, transform_input=None):
    pbar = tqdm(dl_val,mininterval=120)
    count = 0
    total_loss = torch.zeros(1, device=device)
    total_nll  = torch.zeros(1, device=device)
    if transform_input is None:
        transform_input = lambda x: x
    for input_graph, input_cond in pbar:
        start = time.time()
        input_cond = [ torch.from_numpy(condition_normalizer.transform(input_cond[0])) ]
        input_graph = transform_input(input_graph)
        nll, loss, _, _ = flow_forward(flow, prior, input_graph, input_cond, device=device, ode_regularization=ode_regularization)
        total_loss += loss.item()
        total_nll  += nll.item()
        count += 1
        pbar.set_postfix({"val_loss": total_loss.item()/count, "nll": total_nll.item()/count} )
    return total_loss / len(dl_val)


