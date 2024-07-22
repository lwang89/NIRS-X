from torch import nn


class NIRSiam(nn.Module):

    def __init__(self, base_encoder, embedding_size=256):
        super().__init__()
        self.embedding = nn.Sequential(
            base_encoder,
            nn.Flatten()
        )

        self.projection = nn.Sequential(
            nn.LazyLinear(embedding_size, bias=False),
            nn.ReLU(inplace=True),
            nn.LazyLinear(embedding_size, bias=False),
            nn.ReLU(inplace=True),
            nn.LazyLinear(embedding_size)
        )

        self.prediction = nn.Sequential(
            nn.LazyLinear(embedding_size),
            nn.ReLU(inplace=True),
            nn.LazyLinear(embedding_size)
        )

    def forward(self, x, no_projection=False, simsiam=True):
        x = self.embedding(x)

        if simsiam:
            proj = self.projection(x)
            pred = self.prediction(proj)
            
            return proj, pred
        
        elif no_projection:
            return x
        
        else:
            return self.projection(x)
