import numpy as np
import torch
import glob
import pandas as pd
import ipywidgets as widgets
import seaborn as sns
import pickle
from HaloEGNNFlows.EGNNFlows.models import get_model
from HaloEGNNFlows.EGNNFlows.datasets import (
    ProgenitorDataset,
    find_closest_redshift_slice,
    prepare_input_data,
)

from HaloEGNNFlows.train import *

from HaloEGNNFlows.EGNNFlows.viz.utils import compute_metric_features
from HaloEGNNFlows.EGNNFlows.flows.utils import assert_correctly_masked
from HaloEGNNFlows.EGNNFlows.flow_forward import flow_forward


from HaloEGNNFlows.EGNNFlows.flows.utils import (
    assert_mean_zero_with_mask,
    remove_mean_with_mask,
    assert_correctly_masked,
)
from HaloEGNNFlows.EGNNFlows.utils import subtract_the_boundary
import re
from collections import OrderedDict
import emcee
import time


def main(ckpt_path, sample_idx, ckpt_params):
    # filelist = sorted(glob.glob("/jobfs/38701909.gadi-pbs/TG300_preprocessed_data/prog_sublink_*.npy"))
    # print(len(filelist))
    # data_columns = np.load("/jobfs/38701909.gadi-pbs/TNG300_preprocessed_data/subhalo_columns.npy")

    params = np.load(ckpt_params, allow_pickle=True)
    #print(params)
    #valid_files = [os.path.join("/home/196/kt9438/y89_scratch",f) for f in params["valid_files"]]
    valid_files = [os.path.join(".",f) for f in params["valid_files"]]
    condition_normalizer = params["condition_normalizer"]
    data_columns = params["full_columns_names"]
    max_progenitors = 20
    initial_slice = 0
    final_slice = 1

    # The "scalar" feature bounded to the progenitors
    feature_columns = ["SubhaloMass", "SubhaloMergeRedshift"]
    # The "positional" feature bounded to the progenitors
    position_columns = ["SubhaloPos_0", "SubhaloPos_1", "SubhaloPos_2"]

    # initialize the list condtional columns at redshift 0
    condition_columns = [
        "SubhaloBHMass",
        "SubhaloBHMdot",
        "SubhaloGasMetallicity",
        "SubhaloStarMetallicity",
        "SubhaloMass",
        "SubhaloSFR",
        "SubhaloVmax",
        "SubhaloVelDisp",
    ]

    n_dims = len(position_columns)
    in_node_nf = len(feature_columns)
    context_node_nf = len(condition_columns)

    """
    valid_files, condition_normalizer = prepare_filelist_and_transformer(
        filelist_npy=None,
        transformer_pkl=None,
        filelist=filelist,
        condition_columns=condition_columns,
        full_columns_names=data_columns,
        max_progenitors=20,
        initial_slice=0,
        final_slice=1,
        batch_size=512,
        num_workers=2,
        transform_type = "power"
    )
    """

    def preprocess_inputs(ds, idx):
        # select only one element!
        # from the dataset by index
        tmp1, tmp2 = ds[idx]
        input_graph = [t.unsqueeze(0) for t in tmp1]
        input_graph = transform_input(input_graph)
        input_cond = [
            torch.from_numpy(condition_normalizer.transform(t.unsqueeze(0)))
            for t in tmp2
        ]
        return input_graph, input_cond

    dataset = ProgenitorDataset(
        valid_files,
        condition_columns=condition_columns,
        position_columns=position_columns,
        feature_columns=feature_columns,
        max_progenitors=max_progenitors,
        initial_slice=initial_slice,
        final_slice=final_slice,
        data_columns=data_columns,
    )

    # tiny snippet to transform redshift to scale factor
    def transform_z_to_scale(z_idx):
        def transform_graph(input_graph):
            # transform only the features
            if z_idx is None:
                return input_graph
            x, h, node_mask, edge_mask = input_graph
            h[:, :, z_idx] = 1 / (1 + h[:, :, z_idx])
            h[:, :, z_idx] = h[:, :, z_idx].exp()
            return x, h, node_mask, edge_mask

        return transform_graph

    zidx = None
    if "SubhaloMergeRedshift" in feature_columns:
        zidx = feature_columns.index("SubhaloMergeRedshift")
    transform_input = transform_z_to_scale(zidx)

    prior_cuda, flow_cuda = prepare_ckpt_model(
        ckpt_path=ckpt_path, device="cuda", rtol=1e-4, trace_method="hutch"
    )

    dl_train, dl_val, dl_test = prepare_dataloaders(
        sorted(valid_files),
        condition_columns,
        data_columns,
        position_columns=position_columns,
        feature_columns=feature_columns,
        max_progenitors=20,
        initial_slice=0,
        final_slice=1,
        batch_size=256,
        num_workers=0,
        random_seed=42,
        train_test_split=[0.8, 0.1, 0.1],
        shuffle_train=True,
        distributed=False,
    )

    #val_step(flow_cuda, prior_cuda, dl_val,
    #         condition_normalizer, device="cuda", ode_regularization=1e-2,
    #         transform_input=transform_z_to_scale(zidx),)

    #idx = 9239
    idx = sample_idx
    input_graph, input_cond = preprocess_inputs(dataset,idx)
    print(input_graph, input_cond)
    npcond = input_cond[0][0].numpy()

    ndim, nwalkers = npcond.shape[0], 64
    p0 = np.random.randn(nwalkers, ndim)

    #cpndition_log_prob_vec(input_graph)
    a = condition_log_prob_vec(
        p0,
        input_graph,
        flow_cuda,
        prior_cuda,
        rtol=1e-4,
        trace_method='exact',
        device='cuda'
    )
    print(a)

    #cpndition_log_prob_vec(input_graph)
    b = condition_log_prob_vec(
        input_cond[0].numpy(),
        input_graph,
        flow_cuda,
        prior_cuda,
        rtol=1e-4,
        trace_method='exact',
        device='cuda'
    )
    print(b)
    ndim, nwalkers = npcond.shape[0], 64
    p0 = np.random.randn(nwalkers, ndim)

    device = 'cuda'
    rtol = 1e-4
    trace_method = 'exact'

    filename = f'{idx}_progress.h5'
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        condition_log_prob_vec,
        args =[
               input_graph,
               flow_cuda,
               prior_cuda,
               rtol,
               trace_method,
               device],
        backend=backend,
        vectorize=True
        )
    sampler.run_mcmc(p0, 5000,progress=True)
    print(sampler)

def prepare_ckpt_model(
    ckpt_path=None,
    device="cpu",
    bsz=1,
    rtol=1e-4,
    trace_method="exact",
    in_node_nf=2,
    context_node_nf=8,
):

    prior, flow = get_model(
        in_node_nf=in_node_nf,
        dynamics_in_node_nf=1,
        context_node_nf=context_node_nf,
        n_dims=3,
        device=device,
    )

    ckpt = torch.load(ckpt_path, map_location=torch.device(device))

    state_dict = ckpt["model_state_dict"]

    model_dict = OrderedDict()
    pattern = re.compile("module.")
    for k, v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, "", k)] = v
        else:
            model_dict[k] = v
    flow.load_state_dict(model_dict)

    return prior, flow


def repeat_batch(input_cond_np, input_graph, nrepeats=30):
    """repeat inputs, condition vector is numpy array"""
    tmp = torch.from_numpy(input_cond_np).unsqueeze(0)
    new_cond = [tmp.repeat(nrepeats, 1)]
    new_graph = [
        c.repeat(nrepeats, 1, 1) if (c.ndim == 3) else c.repeat(nrepeats, 1)
        for c in input_graph
    ]
    return new_graph, new_cond


@torch.no_grad()
def ensemble_log_prob(
    input_cond_np,
    input_graph,
    flow,
    prior,
    nsamples=30,
    device="cpu",
    rtol=1e-4,
    trace_method="exact",
):
    # we assume the graph is properly transformed already!
    flow.training = False
    flow.transformations[1]._rtol = rtol
    flow.transformations[1]._atol = rtol
    flow.transformations[1].odefunc.method = trace_method
    flow.transformations[1].odefunc.ode_regularization = 0.0

    new_graph, new_cond = repeat_batch(input_cond_np, input_graph, nrepeats=nsamples)
    logpx = flow_logpx(
        flow, prior, new_graph, new_cond, device=device, ode_regularization=0.01
    )
    return logpx.cpu().numpy()


@torch.no_grad()
def condition_log_prob(
    input_cond_np,
    input_graph,
    flow,
    prior,
    rtol=1e-4,
    trace_method="exact",
    device="cpu",
):
    # we assume the graph is properly transformed already!
    flow.training = False
    flow.transformations[1]._rtol = rtol
    flow.transformations[1]._atol = rtol
    flow.transformations[1].odefunc.method = trace_method
    flow.transformations[1].odefunc.ode_regularization = 0.0

    input_cond = [torch.from_numpy(input_cond_np).unsqueeze(0)]
    logpx = flow_logpx(
        flow, prior, input_graph, input_cond, device=device, ode_regularization=0.01
    )
    return logpx.cpu().numpy()


@torch.no_grad()
def condition_log_prob(
    input_cond_np,
    input_graph,
    flow,
    prior,
    rtol=1e-4,
    trace_method="exact",
    device="cpu",
):
    # we assume the graph is properly transformed already!
    flow.training = False
    flow.transformations[1]._rtol = rtol
    flow.transformations[1]._atol = rtol
    flow.transformations[1].odefunc.method = trace_method
    flow.transformations[1].odefunc.ode_regularization = 0.0

    input_cond = [torch.from_numpy(input_cond_np).unsqueeze(0)]
    logpx = flow_logpx(
        flow, prior, input_graph, input_cond, device=device, ode_regularization=0.01
    )
    return logpx.cpu().numpy()


def flow_logpx(
    flow,
    prior,
    input_graph,
    input_cond,
    device="cuda",
    x_norm=300,
    ode_regularization=0.01,
):
    x, h, node_mask, edge_mask = input_graph
    context = input_cond[0]

    x = x.to(device)
    h = h.to(device)
    node_mask = node_mask.to(device)
    edge_mask = edge_mask.to(device)
    context = context.to(device)

    subtract_the_boundary(x, node_mask)
    xx = remove_mean_with_mask(x, node_mask)

    # h = log_normalize_with_mask(h, node_mask)

    h = (h + 1e-8).log() * node_mask
    # dont normalize here
    # context = log_norm_target(context)

    # center the position coordinate
    xx = remove_mean_with_mask(x / x_norm, node_mask)
    assert_mean_zero_with_mask(xx, node_mask)

    bs, n_nodes, n_dims = xx.size()
    # inflate context (condition) in to the shape of [batch, max_nodes, n_context]
    context_ = context.unsqueeze(1).repeat(1, n_nodes, 1) * node_mask
    assert_correctly_masked(context_, node_mask)

    xh = torch.cat([xx, h], dim=2)
    assert_correctly_masked(xh, node_mask)

    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

    t1 = time.time()
    z, delta_logp, reg_term = flow(
        xh.float(), node_mask.float(), edge_mask.float(), context_.float()
    )
    z_x, z_h = z[:, :, 0:n_dims].clone(), z[:, :, n_dims:].clone()

    assert_correctly_masked(z_x, node_mask)
    assert_correctly_masked(z_h, node_mask)

    N = node_mask.squeeze(2).sum(1).long()

    log_qh_x = 0
    log_pz = prior(z_x, z_h, node_mask)
    log_px = log_pz + delta_logp - log_qh_x  # Average over batch.
    print(time.time() - t1, "time in flow") #, np.mean(log_px),np.std(log_px))
    return log_px


@torch.no_grad()
def condition_log_prob_vec(
    input_cond_np,
    input_graph,
    flow,
    prior,
    rtol=1e-4,
    trace_method='exact',
    device='cpu'
    ):
  nsamples = input_cond_np.shape[0]
  # we assume the graph is properly transformed already!
  flow.training = False
  flow.transformations[1]._rtol = rtol
  flow.transformations[1]._atol = rtol
  flow.transformations[1].odefunc.method = trace_method
  flow.transformations[1].odefunc.ode_regularization = 0.0

  input_cond = [ torch.from_numpy(input_cond_np) ]
  new_graph  = [c.repeat(nsamples,1,1)  if (c.ndim == 3) else c.repeat(nsamples,1) for c in input_graph]
  # print(input_cond)
  # print(new_graph)
  logpx = flow_logpx(
    flow,
    prior,
    new_graph,
    input_cond,
    device=device,
    ode_regularization=0.01
  )
  return logpx.cpu().numpy()

if __name__ == "__main__":
    ckpt_path = "/home/196/kt9438/HaloEGNN/train/log/38734815.gadi-pbs/egnn_26_val=22.829.pt"

    sample_idx =9826
    ckpt_params = "/home/196/kt9438/HaloEGNN/train/log/38734815.gadi-pbs/params.pkl"
    main(ckpt_path, sample_idx, ckpt_params)
