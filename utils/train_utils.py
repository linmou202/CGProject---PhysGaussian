import torch

class Young_Moudulous_Map(torch.nn.Module):
    
    def __init__(self, E_item, inverted_index, gs_num):
        super(Young_Moudulous_Map, self).__init__()
        self.E = torch.nn.Parameter(E_item)
        self.inverted_index = inverted_index
        self.gs_num = gs_num

    def forward(self):
        E_out = torch.zeros(self.gs_num, dtype=torch.float)
        for i in range(0, self.gs_num):
            E_out[i] = self.E[self.inverted_index[i]]
        return