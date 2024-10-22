from torch import nn

from hpc2ml.nn.activations import Act


class Forces(nn.Module):
    """Forces net"""

    def __init__(self, output_dim, decoder_type="mlp", decoder_activation_str="ssp", ):
        super(Forces, self).__init__()
        self.decoder_type = decoder_type
        self.act = Act(decoder_activation_str)
        self.output_dim = output_dim

        if self.decoder_type == "linear":
            self.decoder = nn.Sequential(nn.Linear(self.output_dim, 3))
        elif self.decoder_type == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim),
                self.act,
                nn.Linear(self.output_dim, int(self.output_dim / 1)),
                self.act,
                nn.Linear(int(self.output_dim / 1), 3),
            )
        else:
            raise ValueError(f"Undefined forces decoder: {self.decoder_type}")

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.decoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.decoder[-1].weight.data = self.decoder[-1].weight.data / 10

    def forward(self, x):
        x = self.decoder(x)
        return x
