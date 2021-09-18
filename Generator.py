# nz1 = 100 is size of structual latent vector:
# nz2 = 10 is size of class latent vector.

# architecture based off https://arxiv.org/pdf/1610.09585
class Generator(torch.nn.Module):

    def __init__(self):

        super(Generator, self).__init__()
        self.fc1 = torch.nn.Linear(nz1+nz2, 768)
        
        self.main = torch.nn.Sequential(
            
            torch.nn.ConvTranspose2d(
                in_channels = 768,
                out_channels = 384,
                kernel_size = 5,
                stride = 2,
                padding = 0,
                bias = False
            ),
            torch.nn.BatchNorm2d(num_features = 384),
            torch.nn.ReLU(inplace = True),

            torch.nn.ConvTranspose2d(
                in_channels = 384,
                out_channels = 256,
                kernel_size = 5,
                stride = 2,
                padding = 0,
                bias = False
            ),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.ReLU(inplace=True),

            
            torch.nn.ConvTranspose2d(
                in_channels=256,
                out_channels=192,
                kernel_size=5,
                stride=2,
                padding=0,
                bias=False
            ),
            torch.nn.BatchNorm2d(num_features = 192),
            torch.nn.ReLU(inplace=True),

            
            torch.nn.ConvTranspose2d(
                in_channels = 192,
                out_channels = 64,
                kernel_size = 5,
                stride = 2,
                padding = 0,
                bias = False
            ),
            torch.nn.BatchNorm2d(num_features = 64),
            torch.nn.ReLU(inplace=True),

            
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=3,
                kernel_size=8,
                stride=2,
                padding=0,
                bias=False
            ),
            torch.nn.Tanh()
          
        )

    def forward(self, inputs, condition_latent_vec):
       
        concat_inputs = torch.cat((inputs, condition_latent_vec), dim=1)
        
        out1 = self.fc1(concat_inputs)
       
        out1 = out1.unsqueeze(2).unsqueeze(3)
        
        return self.main(out1)
