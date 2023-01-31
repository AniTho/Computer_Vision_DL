import timm
import torch
from torch import nn

class MultiTaskModel(nn.Module):
    def __init__(self, base_model = 'resnet50', freeze = True, num_freeze = 100, dropout_p = 0.4):
        super(MultiTaskModel, self).__init__()
        model = timm.create_model(base_model, pretrained=True)
        lin_features = model.num_features
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        if freeze:
            self.freeze_layers(num_freeze)
        self.age_predictor = nn.Sequential(nn.Linear(lin_features, 512),
                                            nn.ReLU(),
                                            nn.BatchNorm1d(512),
                                            nn.Dropout(dropout_p),
                                            nn.Linear(512, 1),
                                            nn.Sigmoid())
        self.gender_predictor = nn.Sequential(nn.Linear(lin_features, 512),
                                                nn.ReLU(),
                                                nn.BatchNorm1d(512),
                                                nn.Dropout(dropout_p),
                                                nn.Linear(512, 1),
                                                nn.Sigmoid())
        self.gender_predictor.apply(self.initialize_weights)
        self.age_predictor.apply(self.initialize_weights)

    def initialize_weights(self, layer):
        '''
        Initialize weights with kaiming normal initialization technique

        Args:
        - layer(Torch nn layer): Current layer
        '''
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            layer.bias.data.fill_(0.0)

    def freeze_layers(self, num_freeze):
        '''
        Freeze the number of layers of model

        Args:
        - num_freeze (int): Number of layers to be freezed
        '''
        for idx, param in enumerate(self.feature_extractor.parameters()):
            if idx <= num_freeze:
                param.requires_grad = False

    def forward(self, x):
        '''
        Run a forward propagation step on the input batch

        Args:
        - x (torch.Tensor): Input Tensor of shape BxCxHxW
        '''
        features = self.feature_extractor(x)
        age = self.age_predictor(features)
        gender = self.gender_predictor(features)
        return age, gender

def create_model(base_model = 'resnet50', freeze = True, num_freeze = 100, dropout_p = 0.4):
    '''
    Create a pytorch model

    Args:
    - base_model (str): Name of pretrained architecture (list from timm library)
    - freeze (bool): To freeze the pretrained architecture or not
    - num_freeze (int): Number of layers to be frozen
    - dropout_p (float): Probability for dropout, has to be in range [0,1]
    Returns:
    - (nn.Module): Final Model for training
    '''

    model =  MultiTaskModel(base_model = 'resnet50', freeze = True, num_freeze = 100, dropout_p = 0.4)
    return model