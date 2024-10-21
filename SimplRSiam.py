
class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class Twins(nn.Module):
    def __init__(self, emb_width, twintype, module):        
        super().__init__()        
        self.twintype = twintype        
        self.module = module
        self.ens_size = len(self.module)
        self.merge = heads.ProjectionHead(
                        [                
                            (emb_width*self.ens_size, emb_width, None, None),                            
                        ])

    def forward(self, x):
        embeddings = [m(x).flatten(start_dim=1) for m in self.module]
        if self.twintype == 'cat':
            out = torch.concat(embeddings, dim=1)
        elif self.twintype == 'rand':
            out = self.merge(torch.concat(embeddings, dim=1))
        elif self.twintype == 'first':
            out = embeddings[0]
        elif self.twintype == 'avg':
            out = embeddings[0] / self.ens_size
            for i in range(1, self.ens_size):
                out += embeddings[i] / self.ens_size
        else:
            raise Exception("Twin Type Not Supported")
        return out