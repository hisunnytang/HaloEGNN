from torch.utils.data import Dataset
import numpy as np
import h5py

class HaloDataset(Dataset):
  scalar_field_cols = ['Group_M_Crit200', 'Group_M_Mean200', 'Group_R_Crit200', 'Group_R_Mean200',  'SubhaloVMax', 'SubhaloVelDisp']
  vector_field_cols = ['SubhaloPos', 'SubhaloVel','SubhaloSpin']
  target_field_cols = ['Subhalo']
  redshifts = np.array([2.00464916e+01, 1.49891729e+01, 1.19802132e+01, 1.09756432e+01,
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
  
  def __init__(self, 
               filenames, 
               input_redshift=2.0,
               target_redshift=0.0,
               scalar_features = ["SubhaloMassType",  
                                  "SubhaloVMax", 
                                  "SubhaloVelDisp"],
               coor_features   = ['SubhaloPos'],
               vector_features = ['SubhaloVel',
                                  'SubhaloSpin'],
               target_cols     = ['SubhaloMassType'],
               ):

    self.filenames = filenames
    self.input_redshift = input_redshift
    self.target_redshift = target_redshift


    self.snapNum_feat = np.argmin(np.abs(self.redshifts -self.input_redshift))
    self.snapNum_targ = np.argmin(np.abs(self.redshifts -self.target_redshift))

    self.scalar_features = scalar_features
    self.coor_features = coor_features
    self.vector_features = vector_features
    self.target_cols = target_cols

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    fn = self.filenames[idx]
    return self._read_data(fn)

  def _read_data(self, fn):
    with h5py.File(fn, 'r') as f:

      halo_featz_index   = np.where(f['SnapNum'][:] == self.snapNum_feat)[0]
      halo_targetz_index = np.where(f['SnapNum'][:] == self.snapNum_targ)[0]

      scalar_features = []
      for scalar_cols in self.scalar_features:
        if scalar_cols == 'SubhaloMassType':
          feat = f[scalar_cols][halo_featz_index,1]        
        else:
          feat = f[scalar_cols][halo_featz_index]
        scalar_features.append(feat)
      
      pos_features = []
      for pos in self.coor_features:
        pos_features.append(f[pos][halo_featz_index])
      
      vector_features = []
      for vel in self.vector_features:
        vector_features.append(f[vel][halo_featz_index])

      targets = []
      for t in self.target_cols:
        targets.append(f[t][halo_targetz_index])

      edges = self.prepare_edges(len(feat))

      return np.column_stack(scalar_features),\
       pos_features[0], \
       edges, \
       np.stack(vector_features,axis=-1),\
       targets[0]

  def prepare_edges(self,n_nodes):
    adj_mat = np.ones((n_nodes, n_nodes))
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
          if i != j:
            rows.append(i)
            cols.append(j)
    edges = [torch.tensor(rows), torch.tensor(cols)]
    return edges