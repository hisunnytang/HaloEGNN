import torch
from utils import subtract_the_boundary, remove_mean_with_mask, log_norm_context, assert_mean_zero_with_mask


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
    context = log_norm_target(context)

    # center the position coordinate
    xx = remove_mean_with_mask(x/x_norm, node_mask)
    assert_mean_zero_with_mask(xx, node_mask)

    bs, n_nodes, n_dims = xx.size()
    # inflate context (condition) in to the shape of [batch, max_nodes, n_context]
    context_ = context.unsqueeze(-1).repeat(1, n_nodes, 1)* node_mask
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

    log_pz = prior(z_x, z_h, node_mask)
    log_px = (log_pz + delta_logp - log_qh_x).mean()  # Average over batch.
    reg_term = reg_term.mean()  # Average over batch.
    nll = -log_px

    mean_abs_z = torch.mean(torch.abs(z)).item()
    loss = nll + 0.01 * reg_term

    return nll, loss, z_x, z_h

def train_step(flow, prior, optim, dl_train):
    total_loss = 0.0
    for input_graph, input_cond in tqdm(dl_train):
        start = time.time()
        optim.zero_grad()
        nll, loss, _, _ = flow_forward(flow, prior, input_graph, input_cond, device=device, ode_regularizaton=ode_regularization)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss / len(dl_train)

@torch.no_grad()
def val_step(flow, prior, dl_val):
    total_loss = 0.0
    for input_graph, input_cond in tqdm(dl_val):
        start = time.time()
        nll, loss, _, _ = flow_forward(flow, prior, input_graph, input_cond, device=device, ode_regularizaton=ode_regularization)
        total_loss += loss.item()
    return total_loss / len(dl_val)

