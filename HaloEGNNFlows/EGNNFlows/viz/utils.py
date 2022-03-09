from sklearn.preprocessing import PowerTransformer, StandardScaler
import torch
import glob
import pandas as pd
import ipywidgets as widgets
import seaborn as sns
import numpy as np
from ..datasets import ProgenitorDataset


def obtain_condition_transformer(dl, max_sample=500):
    count = 0
    targets = []
    for _, input_cond in dl:
        targets.append(input_cond[0])
        if count > max_sample:
            break
        count += 1
    targets = torch.cat(targets)
    condition_normalizer = PowerTransformer().fit(targets)
    return targets, condition_normalizer


def get_min_max(df):
    column = df.columns
    cols_min_max = []
    for c in column:
        min, max = df[c].min(), df[c].max()
        cols_min_max.append([c, min, max])
    return cols_min_max


def get_quantile_bins(df):
    column = df.columns
    cols_bins = []
    for c in column:
        q, bins = pd.qcut(df[c], 10, retbins=True, duplicates="drop")
        cols_bins.append([c, bins])
    return cols_bins


def create_selectionrange_slider_map(cols_bins):
    sliders_map = {}
    for c, _bins in cols_bins:
        sliders_map[c] = widgets.SelectionRangeSlider(options=_bins, index=[3, 5])
    return sliders_map


def create_slider_map(cols_min_max):
    sliders_map = {}
    for c, _min, _max in cols_min_max:
        # print(c, _min, _max, type(_min))
        sliders_map[c] = widgets.FloatRangeSlider(
            min=_min,
            max=_max,
            step=(_max - _min) / 100,
        )
    return sliders_map


def order_halo_by_mass(x, feat, mass_idx=0):
    # need to sort the ones that are not 0!

    mass = feat[:, :, mass_idx]
    idx_null = mass == 0
    mass[idx_null] = -1e9

    Msample_sorted, Msample_sorted_idx = torch.sort(mass, dim=1, descending=True)

    mass[idx_null] = 0.0

    x_sorted_by_mass = torch.stack(
        [x[idx] for x, idx in zip(Xsamp, Msample_sorted_idx)]
    )
    feat_sorted_by_mass = torch.stack(
        [x[idx] for x, idx in zip(feat, Msample_sorted_idx)]
    )
    return x_sorted_by_mass, feat_sorted_by_mass


def compute_metric_features(
    xposin,
    m,
    max_prog=3,
    box_size = 205000 / 300,
    column_names=None):

    # input shape of [batch, n_node, pos]
    # output shape of [batch, n_metrics]
    # 1. mass
    # 2. pairwise distance measure

    xpos = xposin.clone()
    all_dist = []
    for bsz in range(len(xpos)):
        idx1, idx2 = np.triu_indices(max_prog, 1)
        relvec = (xpos[bsz, idx1, :] - xpos[bsz, idx2, :]).abs()
        relvec[relvec > box_size / 2] = (relvec[relvec > box_size / 2] - box_size).abs()

        all_dist.append((relvec**2).sum(-1).numpy())
        # print((dist**2).sum(-1).shape)

    all_dist = np.array(all_dist)  # .reshape(len(xpos),-1)

    bsz = len(xpos)
    all_mass = m[:, :max_prog].transpose(2,1).reshape(len(xpos), -1).numpy()
    print(all_mass.shape, m.shape)

    dij = np.triu_indices(max_prog, 1)
    if column_names is None:
        feat_names = [f"feat_{i}" for i in range(all_mass.shape[1])] + [
            f"d{i}{j}" for i, j in zip(*dij)
        ]
    else:
        feat_names  = [f"{c}_{i}" for c in column_names for i in range(max_prog) ]
        feat_names += [f"d{i}{j}" for i, j in zip(*dij)
        ]
    # print(all_dist.shape, all_mass.shape)
    return np.hstack((all_mass, all_dist)), feat_names


def query_dataframe(df, max_samples=500, **kwargs):
    qstring = []
    for k, v in kwargs.items():
        qstring += [f" {k} > {v[0]:.2e} & {k} < {v[1]:.2e} "]
    df_selected = df.query("&".join(qstring))
    return df_selected


def get_dataloader_features(dl, max_progenitors=20, distance_norm=300, column_names=None):
    subset_all_feats = []
    for input_graph, input_cond in dl:
        data_mass = input_graph[1]
        data_xpos = input_graph[0]
        print(data_xpos[0])
        feats, col_names = compute_metric_features(
            data_xpos / distance_norm,
            data_mass,
            max_prog=max_progenitors,
            column_names=column_names,
        )
        subset_all_feats.append(feats)
    subset_feats_np = np.vstack(subset_all_feats)
    return subset_feats_np, col_names


def obtain_feature_df(
    filelist,
    condition_columns,
    feature_columns,
    initial_slice,
    final_slice,
    max_progenitors=20,
    max_samples=1000,
    distance_norm=300,
    conditional_df=None,
    full_columns_names=None,
    feat_max_progenitors=10,
):
    # return 4 dataframes
    # return:
    #   df_full_cond
    #   df_sub_cond
    #   df_full_feat
    #   df_sub_feat

    # first create the dataset dataloader for the full data
    position_columns=["SubhaloPos_0", "SubhaloPos_1", "SubhaloPos_2"]
    dataset = ProgenitorDataset(
        filelist,
        condition_columns=condition_columns,
        position_columns=position_columns,
        feature_columns=feature_columns,
        max_progenitors=max_progenitors,
        initial_slice=initial_slice,
        final_slice=final_slice,
        data_columns=full_columns_names,
    )
    print("sample position", dataset[0][0][0])
    print("sample feature", dataset[0][0][1])
    dl = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=2)

    # iterate it once to obtain the full, sorted "conditional" values
    sample_condition, condition_normalizer = obtain_condition_transformer(
        dl, max_sample=max_samples
    )

    df_full_cond = pd.DataFrame(sample_condition.numpy(), columns=condition_columns)

    all_feats_np, cname = get_dataloader_features(dl, feat_max_progenitors, distance_norm, feature_columns)
    print(all_feats_np.shape)
    print(len(cname))

    df1 = pd.DataFrame(all_feats_np, columns=cname)
    df1.replace([np.inf, -np.inf], np.nan, inplace=True)
    df1.dropna(inplace=True)

    # let say we have the full "condition dataframe" now
    # df_sub_cond = query_dataframe(df_full_cond, **query_bounds)

    # as soon as we have the selected subset
    # subset_dl = torch.utils.data.DataLoader(
    #  torch.utils.data.Subset(ds, df_sub_cond.index),
    #  shuffle=False
    #  )

    # subset_feats_np, cname = get_dataloader_features(
    #  subset_dl,
    #  max_progenitors,
    #  distance_norm
    # )

    # df2 = pd.DataFrame(subset_feats_np, columns=cname)

    # me being lazy; accounting ONLY for graphs with >= N progenitors
    # df2.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df2.dropna(inplace=True)

    return df_full_cond, df1


