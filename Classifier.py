#VGG16 model: https://arxiv.org/abs/1409.1556

from torchvision.models import vgg16
model = vgg16()


class Classifier(torch.nn.Module):

    def __init__(self):

        super(Classifier, self).__init__()
        self.vgg = vgg16(pretrained = True)
        
        
        
        self.features = self.vgg.features
        self.avgpool = self.vgg.avgpool
        
        self.fc = torch.nn.Sequential (self.vgg.classifier[0], self.vgg.classifier[1], self.vgg.classifier[2], 
                                      self.vgg.classifier[3], self.vgg.classifier[4], self.vgg.classifier[5])
        
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(
                in_features= 4096,
                out_features= 1024,
                bias=True
            ),
            torch.nn.ReLU(), 
            torch.nn.Linear(
                in_features= 1024,
                out_features= 256,
                bias=True
            ),
            torch.nn.ReLU(), 
            torch.nn.Linear(
                in_features= 256,
                out_features= 64,
                bias=True
            ),
            torch.nn.ReLU(), 
            torch.nn.Linear(
                in_features= 64,
                out_features= 2,
                bias=True
            ),
            torch.nn.Softmax(dim=1)
        )
        
       

    def forward(self, input):
        features = self.features(input)
        features_avgpooled = self.avgpool(features)
        features_fc = self.fc(features_avgpooled.view(features.shape[0], -1))
        clf = self.clf(features_fc.view(features_fc.shape[0], -1))
        return clf, features
