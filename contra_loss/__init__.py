"""load networks parameters"""
from .DeepConvNet import DeepConvNet
from .EEGNet import EEGNet
from .NIRSformer import CNNEncoder, NIRSformer
from .NIRSiam import NIRSiam


def get_model(config):
    """
    Description:load networks parameters
    """
    if config.model == 'DeepConvNet':
        return DeepConvNet(feature_size=config.feature_size,
                           num_timesteps=config.num_timesteps,
                           num_classes=len(config.labels),
                           dropout=config.dropout)

    elif config.model == 'EEGNet':
        return EEGNet(feature_size=config.feature_size,
                      window_size=config.num_timesteps,
                      num_classes=len(config.labels),
                      F1=config.F1,
                      D=config.D,
                      F2=config.F2,
                      avgpool2d_1=config.avgpool2d_1,
                      avgpool2d_2=config.avgpool2d_2,
                      dropout=config.dropout)

    elif config.model == 'contra_DeepConvNet':
        return NIRSiam(DeepConvNet(feature_size=config.feature_size,
                                   num_timesteps=config.num_timesteps,
                                   num_classes=len(config.labels),
                                   dropout=config.dropout))

    elif config.model == 'NIRSformer':
        return NIRSformer(num_classes=len(config.labels),
                          dropout=config.dropout)

    elif config.model == 'contra_NIRSformer':
        return NIRSiam(NIRSformer(num_classes=len(config.labels),
                                  dropout=config.dropout))

    elif config.model == 'contra_EEGNet':
        return NIRSiam(EEGNet(feature_size=config.feature_size,
                              window_size=config.num_timesteps,
                              num_classes=len(config.labels),
                              F1=config.F1,
                              D=config.D,
                              F2=config.F2,
                              avgpool2d_1=config.avgpool2d_1,
                              avgpool2d_2=config.avgpool2d_2,
                              dropout=config.dropout))

    elif config.model == 'contra_SimCNN':
        return NIRSiam(CNNEncoder(num_classes=len(config.labels),
                                  dropout=config.dropout))

    elif config.model == 'SimCNN':
        return CNNEncoder(num_classes=len(config.labels),
                          dropout=config.dropout)

    else:
        raise NotImplementedError
