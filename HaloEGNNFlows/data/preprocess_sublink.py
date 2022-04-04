import numpy as np
import h5py
import glob
from typing import Dict
from multiprocess import Pool
import os

# list of features that can be extracted
subhalo_feature_cols = [
    "SubhaloBHMass",
    "SubhaloBHMdot",
    "SubhaloBfldDisk",
    "SubhaloBfldHalo",
    #  'SubhaloCM',
    "SubhaloGasMetalFractions",
    "SubhaloGasMetalFractionsHalfRad",
    "SubhaloGasMetalFractionsMaxRad",
    "SubhaloGasMetalFractionsSfr",
    "SubhaloGasMetalFractionsSfrWeighted",
    "SubhaloGasMetallicity",
    "SubhaloGasMetallicityHalfRad",
    "SubhaloGasMetallicityMaxRad",
    "SubhaloGasMetallicitySfr",
    "SubhaloGasMetallicitySfrWeighted",
    #  'SubhaloGrNr',
    "SubhaloHalfmassRad",
    "SubhaloHalfmassRadType",
    #  'SubhaloID',
    #  'SubhaloIDMostbound',
    #  'SubhaloIDRaw',
    #  'SubhaloLen',
    #  'SubhaloLenType',
    "SubhaloMass",
    "SubhaloMassInHalfRad",
    "SubhaloMassInHalfRadType",
    "SubhaloMassInMaxRad",
    "SubhaloMassInMaxRadType",
    "SubhaloMassInRad",
    "SubhaloMassInRadType",
    "SubhaloMassType",
    #  'SubhaloParent',
    "SubhaloPos",
    "SubhaloSFR",
    "SubhaloSFRinHalfRad",
    "SubhaloSFRinMaxRad",
    "SubhaloSFRinRad",
    "SubhaloSpin",
    "SubhaloStarMetalFractions",
    "SubhaloStarMetalFractionsHalfRad",
    "SubhaloStarMetalFractionsMaxRad",
    "SubhaloStarMetallicity",
    "SubhaloStarMetallicityHalfRad",
    "SubhaloStarMetallicityMaxRad",
    "SubhaloStellarPhotometrics",
    "SubhaloStellarPhotometricsMassInRad",
    "SubhaloStellarPhotometricsRad",
    "SubhaloVel",
    "SubhaloVelDisp",
    "SubhaloVmax",
    "SubhaloVmaxRad",
    "SubhaloWindMass",
    #  'TreeID'
]

feat_names = [
    "SubhaloMassType",
    "SubhaloPos",
    "SubhaloVel",
    "SubhaloSpin",
    "SubhaloVMax",
    "SubhaloVelDisp",
    "SubhaloSFR",
    "SubhaloBHMass",
    "SubhaloBHMdot",
    # 'SubhaloBfldDisk', 'SubhaloBfldHalo',
    "SubhaloGasMetallicity",
    "SubhaloGasMetallicitySfr",
    "SubhaloHalfmassRad",
    "SubhaloMassType",
    "SubhaloStarMetallicity",
    "SubhaloStellarPhotometrics",
]


_redshifts = np.array(
    [
        2.00464916e01,
        1.49891729e01,
        1.19802132e01,
        1.09756432e01,
        9.99659061e00,
        9.38877106e00,
        9.00234032e00,
        8.44947624e00,
        8.01217270e00,
        7.59510708e00,
        7.23627615e00,
        7.00541687e00,
        6.49159765e00,
        6.01075745e00,
        5.84661388e00,
        5.52976561e00,
        5.22758102e00,
        4.99593353e00,
        4.66451788e00,
        4.42803383e00,
        4.17683506e00,
        4.00794506e00,
        3.70877433e00,
        3.49086142e00,
        3.28303313e00,
        3.00813103e00,
        2.89578509e00,
        2.73314261e00,
        2.57729030e00,
        2.44422579e00,
        2.31611085e00,
        2.20792556e00,
        2.10326958e00,
        2.00202823e00,
        1.90408957e00,
        1.82268929e00,
        1.74357057e00,
        1.66666961e00,
        1.60423458e00,
        1.53123903e00,
        1.49551213e00,
        1.41409826e00,
        1.35757661e00,
        1.30237842e00,
        1.24847257e00,
        1.20625806e00,
        1.15460277e00,
        1.11415052e00,
        1.07445788e00,
        1.03551042e00,
        9.97294247e-01,
        9.50531363e-01,
        9.23000813e-01,
        8.86896908e-01,
        8.51470888e-01,
        8.16709995e-01,
        7.91068256e-01,
        7.57441401e-01,
        7.32636154e-01,
        7.00106382e-01,
        6.76110387e-01,
        6.44641817e-01,
        6.21428728e-01,
        5.98543286e-01,
        5.75980842e-01,
        5.46392202e-01,
        5.24565816e-01,
        5.03047526e-01,
        4.81832951e-01,
        4.60917801e-01,
        4.40297842e-01,
        4.19968933e-01,
        3.99926960e-01,
        3.80167872e-01,
        3.60687643e-01,
        3.47853839e-01,
        3.28829736e-01,
        3.10074121e-01,
        2.97717690e-01,
        2.73353338e-01,
        2.61343271e-01,
        2.43540183e-01,
        2.25988388e-01,
        2.14425042e-01,
        1.97284177e-01,
        1.80385262e-01,
        1.69252038e-01,
        1.52748764e-01,
        1.41876206e-01,
        1.25759333e-01,
        1.09869942e-01,
        9.94018018e-02,
        8.38844329e-02,
        7.36613870e-02,
        5.85073233e-02,
        4.85236309e-02,
        3.37243713e-02,
        2.39744280e-02,
        9.52166691e-03,
        2.22044605e-16,
    ]
)


def find_closest_redshift_slice(z):
    return np.argmin(np.abs(_redshifts - z))


def find_massive_progenitor_branch(
    f: h5py._hl.files.File, haloID2indx: Dict[np.longlong, np.int32]
):
    idx = np.array([0])
    current_progID = f["FirstProgenitorID"][idx]
    main_prog = []
    main_prog_snapnum = []
    while current_progID.item() != -1:
        main_prog.append(current_progID)
        main_prog_snapnum.append(f["SnapNum"][idx])

        idx = haloID2indx[current_progID.item()]
        current_progID = f["FirstProgenitorID"][idx]
    main_progID2snapnum = {
        i.item(): s.item() - 1 for i, s in zip(main_prog, main_prog_snapnum)
    }
    return main_progID2snapnum


def find_most_massive_branch(
    f: h5py._hl.files.File, haloID2indx: Dict[np.longlong, np.int32]
):
    massive_progID2snapnum = {}
    for s in range(100):
        snapIdx = np.where(f["SnapNum"][:] == s)[0]
        most_mass = (f["SubhaloMassType"][snapIdx, :]).sum(-1)
        if len(most_mass) == 0:
            continue
        massidx = np.where(f["SubhaloMassType"][:, :].sum(-1) == most_mass.max())[0]
        idx = np.intersect1d(snapIdx, massidx)
        try:
            idx.item()
        except:
            print(s, idx, most_mass, most_mass.shape,)
            most_mass = (f["SubhaloMassType"][np.where(f["SnapNum"][:] == s)[0], :])
            print(most_mass.shape, most_mass)
            print(np.where(f['SnapNum'][:] == s))
            raise ValueError
        massive_progID2snapnum[f["SubhaloID"][idx.item()]] = s
    return massive_progID2snapnum


def find_closest_main_progenitor_snapnum(
    f: h5py._hl.files.File,
    subhaloID: np.longlong,
    haloID2indx: Dict[np.longlong, np.int32],
    main_progID2snapnum: Dict[np.longlong, np.int32],
    desc_dict: Dict,
):
    descID = subhaloID  # f['DescendantID'][ haloID2indx[subhaloID] ]
    traverse_desc = []
    while descID not in main_progID2snapnum:
        subhaloID = descID
        descID = f["DescendantID"][haloID2indx[subhaloID]]
        traverse_desc.append(descID)
        if descID in desc_dict:
            break
    final_z = (
        _redshifts[main_progID2snapnum[descID]]
        if descID not in desc_dict
        else desc_dict[descID]
    )
    desc_dict.update({d: final_z for d in traverse_desc})
    return final_z


def estimate_merging_redshift(
    filename: str,
    snapNum: int,
    n_most_massive: int = 20,
    haloID2indx=None,
    mpb2snapnum=None,
) -> np.ndarray:

    prog_dict = {}
    with h5py.File(filename, "r") as f:
        if haloID2indx is None or mpb2snapnum is None:
            haloID2indx = {sid: i for i, sid in enumerate(f["SubhaloID"][:])}
            # mpb2snapnum = find_massive_progenitor_branch(f, haloID2indx)
            mpb2snapnum = find_most_massive_branch(f, haloID2indx)

        subhalos_sn = f["SubhaloID"][f["SnapNum"][:] == snapNum]
        subhalos_sn_mass = f["SubhaloMassType"][
            np.where(f["SnapNum"][:] == snapNum)[0],
        ].sum(-1)

        subhalos_sn_pad = np.zeros(n_most_massive, dtype=np.float64)
        merge_z = np.zeros_like(subhalos_sn_pad, dtype=np.float64)

        # from most massive to least massive
        tmpidx = np.argsort(subhalos_sn_mass)[::-1]
        idx_ord = subhalos_sn[tmpidx][:n_most_massive]

        for i, s in enumerate(idx_ord):
            # prog_dict = {}
            merge_z[i] = find_closest_main_progenitor_snapnum(
                f, s, haloID2indx, mpb2snapnum, prog_dict
            )
            subhalos_sn_pad[i] = subhalos_sn_mass[tmpidx[i]]

    return haloID2indx, mpb2snapnum, merge_z, subhalos_sn_pad


def gather_attrs_by_redshifts(filename, snapnum, columns):
    """take relevent data from each lhalotree

    Args:
      filename: the filepath of the lhalotree
      snapnum:  the redshift slice to take
      columns: the data columns to extract from the lhalotree
    Return:
      col_extracted: the extracted column name
      data: the data extracted
    """

    data = []
    col_extracted = []
    with h5py.File(filename, "r") as f:
        idx = np.where(f["SnapNum"][:] == snapnum)[0]
        for k in columns:
            dim = f[k].shape[1] if len(f[k].shape) == 2 else 1
            if dim > 1:
                col_extracted += [f"{k}_{i}" for i in range(dim)]
                data.append(f[k][idx, :])
            else:
                col_extracted += [k]
                data.append(f[k][idx][:, np.newaxis])

    # print(data)
    data = np.hstack(data)
    return col_extracted, data


def sort_progenitor_by_columns(data, col_indx):
    sort_field = data[:, col_indx]
    idx_sorted = np.argsort(sort_field)[::-1]

    return data[idx_sorted]


def get_radial_features(
    data,
    threshold=750,
):
    """turn vectorial features to scalar features, i.e. radial distance/ velocities
    Args:
      data: the data to obtain radial_features
    return:
      edge_attr: the mean square difference
    """
    center_data = data[0]
    satellites = data[1:]
    edge_attr = np.zeros((len(data), 1))
    edge_attr[1:] = calculate_wrapped_distance(center_data, satellites, thres=threshold)
    return edge_attr


def calculate_wrapped_distance(r1, r2, thres):
    vdiff = np.abs(r1 - r2)
    flag = vdiff > thres / 2.0
    vdiff[flag] -= thres
    diff = (vdiff**2).sum(-1)
    return diff[:, np.newaxis]
    # def prepare_edge_features(edges, node_feat, thres=400):
    # # TODO: require extra care when dealing with more vectorial features
    # edge_attr = []
    # for n in node_feat:
    #   vdiff = ((n[edges[0]]- n[edges[1]]).abs())
    #   flag = vdiff > thres
    #   vdiff[flag] -= 750
    #   diff = (vdiff**2).sum(-1)
    #   if len(diff.shape) != 1:
    #     diff = diff.sum(-1)
    #   edge_attr.append(diff)
    # return torch.stack(edge_attr).transpose(1,0)


def pad_truncate(data, max_progenitors=10):
    if data.shape[0] > max_progenitors:
        return data[:max_progenitors]
    else:
        newdata = np.zeros((max_progenitors, data.shape[1]))
        newdata[: len(data)] = data
        return newdata


def extract_progenitors_data(
    filename,
    redshift_slice,
    features_to_extract=["SubhaloMassType"],
    sort_by_feature="SubhaloMassType_1",
    radial_features=["SubhaloPos", "SubhaloVel"],
    radial_features_thres=[75000, 1e12],
    max_progenitors=10,
):

    features_ = np.sort(np.unique(features_to_extract + radial_features))
    colname, subdata = gather_attrs_by_redshifts(filename, redshift_slice, features_)
    col2indx = {c: i for i, c in enumerate(colname)}



    sort_col_indx = col2indx[sort_by_feature]  # DM mass
    sorted_data = sort_progenitor_by_columns(subdata, sort_col_indx)
    sorted_data = pad_truncate(sorted_data, max_progenitors=max_progenitors)

    # aggregate edge featyures
    radial_feats = []
    radial_cols = []
    for f, thres in zip(radial_features, radial_features_thres):
        indx = [col2indx[f"{f}_{i}"] for i in range(3)]
        mask = (sorted_data[:, [sort_col_indx]] > 0).astype(int)
        _feat = get_radial_features(sorted_data[:, indx], threshold=thres) * mask
        # print(_feat)
        radial_feats.append(_feat)
        radial_cols += [f"{f}_{i}" for i in range(3)]

    cols = [c for c in colname if c not in radial_cols]
    # indx = [i for i, c in enumerate(colname) if c not in radial_cols]
    # sorted_data = sorted_data[:,indx]
    cols = colname

    # pos_indx = [col2indx[f"SubhaloPos_{i}"] for i in range(3)]
    # vel_indx = [col2indx[f"SubhaloVel_{i}"] for i in range(3)]

    # radial_distance = get_radial_features(sorted_data[:,pos_indx]/1e2, threshold=750 )
    # radial_velocity = get_radial_features(sorted_data[:,vel_indx]/1e3, threshold=1e9 )

    # filter out the vectoral information



    # get mass fraction
    mtype_idx = [col2indx[c] for c in ['SubhaloMassType_0','SubhaloMassType_1','SubhaloMassType_4','SubhaloMassType_5' ]]
    midx = [col2indx[c] for c in colname if c in ['SubhaloMass']]
    frac_data = sorted_data[:, mtype_idx] / (sorted_data[:, midx]+1e-10)

    data_inputs = np.hstack((sorted_data, *radial_feats, frac_data))
    cols += [f"Radial{r}" for r in radial_features] + ['GasFrac', 'DMFrac', 'StarWindFrac', 'BHFrac']
    return np.array(cols), data_inputs


def drop_columns(data, columns, column_to_drop):
    idx_keep = np.array([i for i, c in enumerate(columns) if c not in column_to_drop])
    return columns[idx_keep], data[:, idx_keep]


def write_prog_by_filename(fn, output_dir="/content/drive/MyDrive/TNG300/prog_max20"):
    try:
        dd = os.path.split(fn)
        if output_dir is None:
            output_dir = dd[0]
        outfile = os.path.join(output_dir, f"prog_{dd[-1].split('.')[0]}.npy")
        if os.path.exists(outfile):
            print(f"{outfile} exists")
            return

        progs = []
        # iterate over all redshifts
        for i in range(100):
            colname, prog0 = extract_progenitors_data(
                fn, i, max_progenitors=20, features_to_extract=feat_names
            )
            progs.append(prog0.transpose().flatten())

        output_arr = np.array(progs)
        np.save(outfile, output_arr)

        print(outfile)

    except:
        print(fn, "failed")
        return fn


def write_prog_by_filename_with_mergez(
    fn,
    output_dir="/content/drive/MyDrive/TNG300/prog_max20",
    max_progenitors=20,
    feat_names=subhalo_feature_cols,
    sort_by_feature="SubhaloMass",
    extract_z_slice=range(100),
):
    dd = os.path.split(fn)
    if output_dir is None:
        output_dir = dd[0]
    outfile = os.path.join(output_dir, f"prog_{dd[-1].split('.')[0]}.npy")
    if os.path.exists(outfile):
        print(f"{outfile} exists")
        return

    haloID2indx = None
    mpb2snapnum = None
    progs = []
    # iterate over all redshifts
    for i in extract_z_slice:
        colname, prog0 = extract_progenitors_data(
            fn, i, max_progenitors=max_progenitors, features_to_extract=feat_names,sort_by_feature=sort_by_feature
        )
        haloID2indx, mpb2snapnum, merge_z, mass = estimate_merging_redshift(
            fn,
            i,
            n_most_massive=max_progenitors,
            haloID2indx=haloID2indx,
            mpb2snapnum=mpb2snapnum,
        )
        prog0 = np.hstack((prog0, merge_z[:, np.newaxis]))
        progs.append(prog0.transpose().flatten())

    output_arr = np.array(progs)
    np.save(outfile, output_arr)

    #print(outfile)
    return outfile
    #except Exception as e:
    #    print(fn, "failed")
    #    print(e)
    #    return fn

def get_colname(fn, feat_names, max_progenitors=20, i = 0):
    colname, prog0 = extract_progenitors_data(
        fn, i, max_progenitors=max_progenitors, features_to_extract=feat_names
    )
    colname  = list(colname)
    colname += ["SubhaloMergeRedshift"]
    return colname


if __name__ == "__main__":
    preprocessed_loc = "/scratch/y89/kt9438/TNG300_preprocessed_data"
    sublink_loc = "/scratch/y89/kt9438/QueryTNGData/"

    existing_npy = glob.glob(f"{preprocessed_loc}/*.npy")
    existing_hid = [f.split("/")[-1].split(".")[0][13:] for f in existing_npy]

    filelist = glob.glob(f"{sublink_loc}/*.hdf5")
    all_hid = [f.split("/")[-1].split(".")[0][8:] for f in filelist]

    remaining_hid = sorted(list(set(all_hid) - set(existing_hid)))
    remaining_filenames = [
        f"{sublink_loc}/sublink_{h}.hdf5" for h in remaining_hid
    ]

    np.save("remaning_files.npy", np.array(remaining_filenames))

    column_name = get_colname(remaining_filenames[0], subhalo_feature_cols)
    np.save("subhalo_columns.npy", column_name)

    # identify the location of the slices
    initial_slice = find_closest_redshift_slice(2.0)
    final_slice = find_closest_redshift_slice(0.0)
    print(len(remaining_filenames))

    # Pool the data preprocessing
    pool = Pool()
    pool.map(
        lambda fn: write_prog_by_filename_with_mergez(
            fn,
            output_dir=preprocessed_loc,
            extract_z_slice=[initial_slice, final_slice],
            feat_names=subhalo_feature_cols,
            sort_by_feature="SubhaloMass",
            max_progenitors=20,
        ),
        remaining_filenames,
    )
