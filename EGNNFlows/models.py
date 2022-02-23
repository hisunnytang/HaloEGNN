import torch

from egnn.models import EGNN_dynamics_QM9
from flows.ffjord import FFJORD
from flows.dequantize import UniformDequantizer, \
    VariationalDequantizer, ArgmaxAndVariationalDequantizer
from flows import Flow
from flows.actnorm import ActNormPositionAndFeatures
from flows.distributions import PositionFeaturePrior


def get_model(
    in_node_nf = 1,
    dynamics_in_node_nf = 1,
    context_node_nf = 1,
    n_dims = 3,
    device='cuda',
    nf = 32,
    n_layers=3,
    attention=True,
    tanh=True,
    mode = "egnn_dynamics",
    ode_regularization = 1e-3
    trace = 'hutch',
    condition_time = True
):

    prior = PositionFeaturePrior(n_dim=n_dims, in_node_nf=in_node_nf)

    if condition_time:
        dynamics_in_node_nf = in_node_nf + 1

    net_dynamics = EGNN_dynamics_QM9(
                in_node_nf=dynamics_in_node_nf,
                context_node_nf=context_node_nf,
                n_dims=n_dims, device=device, hidden_nf=nf,
                act_fn=torch.nn.SiLU(),
                n_layers=n_layers,
                recurrent=True,
                attention=attention,
                condition_time=condition_time,
                tanh=tanh,
                mode=mode
    )

    flow_transforms = []

    actnorm = ActNormPositionAndFeatures(in_node_nf, n_dims=n_dims).to(device)
    flow_transforms.append(actnorm)

    ffjord = FFJORD(net_dynamics, trace_method='hutch',
                                    ode_regularization=ode_regularization).to(device)
    flow_transforms.append(ffjord)
    flow = Flow(transformations=flow_transforms).to(device)
    flow.set_trace(trace)
    return prior, flow
