from .EGNN_layers import *

class EGNN_Model(torch.nn.Module):
  def __init__(self,
               in_node_nf=8,
               in_edge_nf=3,
               hidden_nf=32,
               n_layers=4,
               target_shape= 1):
    super().__init__()
    self.egnn = EGNN(in_node_nf=in_node_nf,
                     in_edge_nf=in_edge_nf,
                     hidden_nf=hidden_nf,
                     n_layers=n_layers,
                     device='cpu')

    self.hidden_nf = hidden_nf
    act_fn = nn.LeakyReLU(0.2)
    self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                  act_fn,
                                  nn.Linear(self.hidden_nf, self.hidden_nf))

    self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                    act_fn,
                                    nn.Linear(self.hidden_nf, target_shape))

  def forward(self, nodes, pos_feat, edges, edge_attr):

    h,x = self.egnn(nodes, pos_feat, edges, edge_attr)
    h = self.node_dec(h)
    h = h.view(-1, len(nodes), self.hidden_nf)
    h = torch.mean(h, dim=1)
    pred = self.graph_dec(h)
    return pred
