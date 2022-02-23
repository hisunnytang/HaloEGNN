import numpy as np
import torch
from typing import List

# Mapping from SnapNum to Redshifts
_redshifts = np.array([2.00464916e+01, 1.49891729e+01, 1.19802132e+01, 1.09756432e+01,
                       9.99659061e+00, 9.38877106e+00, 9.00234032e+00, 8.44947624e+00,
                       8.01217270e+00, 7.59510708e+00, 7.23627615e+00, 7.00541687e+00,
                       6.49159765e+00, 6.01075745e+00, 5.84661388e+00, 5.52976561e+00,
                       5.22758102e+00, 4.99593353e+00, 4.66451788e+00, 4.42803383e+00,
                       4.17683506e+00, 4.00794506e+00, 3.70877433e+00, 3.49086142e+00,
                       3.28303313e+00, 3.00813103e+00, 2.89578509e+00, 2.73314261e+00,
                       2.57729030e+00, 2.44422579e+00, 2.31611085e+00, 2.20792556e+00,
                       2.10326958e+00, 2.00202823e+00, 1.90408957e+00, 1.82268929e+00,
                       1.74357057e+00, 1.66666961e+00, 1.60423458e+00, 1.53123903e+00,
                       1.49551213e+00, 1.41409826e+00, 1.35757661e+00, 1.30237842e+00,
                       1.24847257e+00, 1.20625806e+00, 1.15460277e+00, 1.11415052e+00,
                       1.07445788e+00, 1.03551042e+00, 9.97294247e-01, 9.50531363e-01,
                       9.23000813e-01, 8.86896908e-01, 8.51470888e-01, 8.16709995e-01,
                       7.91068256e-01, 7.57441401e-01, 7.32636154e-01, 7.00106382e-01,
                       6.76110387e-01, 6.44641817e-01, 6.21428728e-01, 5.98543286e-01,
                       5.75980842e-01, 5.46392202e-01, 5.24565816e-01, 5.03047526e-01,
                       4.81832951e-01, 4.60917801e-01, 4.40297842e-01, 4.19968933e-01,
                       3.99926960e-01, 3.80167872e-01, 3.60687643e-01, 3.47853839e-01,
                       3.28829736e-01, 3.10074121e-01, 2.97717690e-01, 2.73353338e-01,
                       2.61343271e-01, 2.43540183e-01, 2.25988388e-01, 2.14425042e-01,
                       1.97284177e-01, 1.80385262e-01, 1.69252038e-01, 1.52748764e-01,
                       1.41876206e-01, 1.25759333e-01, 1.09869942e-01, 9.94018018e-02,
                       8.38844329e-02, 7.36613870e-02, 5.85073233e-02, 4.85236309e-02,
                       3.37243713e-02, 2.39744280e-02, 9.52166691e-03, 2.22044605e-16])

def extract_column_data( data: np.ndarray, column_names: List, max_progenitors: int, redshift_slice: int )-> np.ndarray:
    """extract the relavant column only over a particular redshift slice"""""
    col_indx = [np.where(colname == c)[0][0] for c in column_names]

    datas = []
    for i in col_indx:
        datas.append(data[redshift_slice, i*max_progenitors: (i+1)*max_progenitors])
        return np.array(datas)

def find_closest_redshift_slice(z):
    return np.argmin(np.abs(_redshifts-z))


def get_fully_connected_edgemask(num_prog, max_prog):
    """connected if there are halos else zero"""""
    edge_mask = np.zeros((max_prog,max_prog))
    for i in range(num_prog):
        for j in range(num_prog):
            if i == j: continue
            edge_mask[i,j] = 1
    return edge_mask.flatten()

def prepare_input_data(data,
                       initial_slice,
                       final_slice,
                       max_progenitors=20,
                       position_columns = ["SubhaloPos_0", "SubhaloPos_1", "SubhaloPos_2"],
                       feature_columns  = ["SubhaloMassType_1"],
                       condition_columns = ['SubhaloMassType_1']
                       ):

    position_data = extract_column_data(data, position_columns, max_progenitors, initial_slice )
    mass_data     = extract_column_data(data, feature_columns, max_progenitors, initial_slice )

    #_position_data = extract_column_data(data, position_columns, max_progenitors, final_slice )
    _mass_data     = extract_column_data(data, condition_columns, max_progenitors, final_slice )

    # get edge data too!
    node_mask = (mass_data >0).astype(int)
    edge_mask = get_fully_connected_edgemask( (node_mask).sum(), max_progenitors  )

    return [np.transpose(position_data), np.transpose(mass_data), np.transpose(node_mask), edge_mask], [_mass_data[:,0],]

class ProgenitorDataset(torch.utils.data.Dataset):
  def __init__(self,
               filenames,
               max_progenitors=20,
               first_slice=30,
               final_slice=99,
               position_columns = ["SubhaloPos_0", "SubhaloPos_1", "SubhaloPos_2"],
               feature_columns  = ["SubhaloMassType_1"],
               condition_columns = ['SubhaloMassType_1']
               ):
      super().__init__()
      self.filenames = filenames
      self.max_progenitors=max_progenitors
      self.initial_slice = initial_slice
      self.final_slice   = final_slice
      self.condition = condition_columns
      self.feature   = feature_columns
      self.position = position_columns

  def __len__(self):
      return len(self.filenames)

  def __getitem__(self, idx):
      data = np.load(self.filenames[idx])
      input_data, cond_data = prepare_input_data(data,
                                                 self.initial_slice,
                                                 self.final_slice,
                                                 max_progenitors=self.max_progenitors,
                                                 position_columns=self.position
                                                 feature_column=self.feature,
                                                 condition_column=self.condition,
                                                 )
      torch_data = [torch.from_numpy(i) for i in input_data]
      cond_data  = [torch.from_numpy(i) for i in cond_data]
      return torch_data, cond_data


