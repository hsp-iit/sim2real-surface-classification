
import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm

import timm
import torchvision

from typing import Optional, Tuple, List, Dict


# special models that are not available in torchvision and/or timm
SPECIAL_MODELS = ["dino_v2"]


class ComposedModel(nn.Module):

    def __init__(self, 
                 model_name: str, 
                 n_classes: int,
                 from_torchvision: bool = False,
                 input_shape: Tuple[int, int, int] = (3,224,224), 
                 bottleneck_dim: int = 256,
                 weight_norm: bool = False,
                 bias: bool = False) -> None:
        """
            A Composed model is in the form: 
            (Input image ->) FeatureExtractor -> Bottleneck -> Classifier (-> predicion)

            Args:
                model_name (str): the name of the feature extractor.
                n_classes (int): the number of the classes.
                from_torchvision (bool). True to load torchvision model, False
                to use timm. Default: False.
                input_shape (tuple): the input shape.
                bottleneck_dim (int): the size of the bottleneck.
                weight_norm (bool): True to use weight normalization on the 
                classifier.
                bias (bool): True to add the bias to the classifier.

        """
        
        super().__init__()

        self.name = model_name
        self.n_classes = n_classes
        self.from_torchvision = from_torchvision
        self.input_shape = input_shape
        self.bottleneck_dim = bottleneck_dim
        self.weight_norm = weight_norm

        self.backbone = _get_backbone(name=model_name, from_torchvision=from_torchvision)
        self.bottleneck = _get_bottleneck(model=self.backbone, 
                                          out_size=bottleneck_dim,
                                          input_shape=input_shape)
        
        self.classifier = _get_classifier(in_size=bottleneck_dim, 
                                          out_size=n_classes, 
                                          weight_norm=weight_norm,
                                          bias=bias)



    def forward_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """ Returns the features (after the backbone). """
        return self.backbone(x)


    def forward_bottleneck_features(self, x: torch.Tensor) -> torch.Tensor:
        """ Returns the features (after the bottlenck). """
        return self.bottleneck(self.backbone(x))
    

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """ Returns the logits. """
        return self.classifier(self.bottleneck(self.backbone(x)))
 

    def forward(self, x: torch.Tensor) -> Dict:
        """ Returns the logits and the features (after the bottlenck). """

        backbone_features = self.backbone(x)
        features = self.bottleneck(backbone_features)
        logits = self.classifier(features)

        return {"logits": logits, 
                "features": features, 
                "backbone_features": backbone_features}


    def get_features_dim(self) -> int:
        return self.bottleneck_dim


    def get_backbone_features_dim(self) -> int:
        return self.bottleneck[0].in_features
    

    def get_classifier_weights(self) -> torch.Tensor:
        return self.classifier.weight
    

    def get_param_groups(self, 
                         lr_model: float, 
                         lr_bottleneck: float, 
                         lr_classifier: float) -> List:
        """ Returns the param groups of the model with possibly different lr. """
        param_groups = []

        for _, v in self.backbone.named_parameters():
            param_groups += [{'params': v, 'lr': lr_model}]
        for _, v in self.bottleneck.named_parameters():
            param_groups += [{'params': v, 'lr': lr_bottleneck}]
        for _, v in self.classifier.named_parameters():
            param_groups += [{'params': v, 'lr': lr_classifier}]

        return param_groups
    

def _has_layer(module: nn.Module, layer: nn.Module) -> bool:
    """ 
        Function to check if a module has a layer.

        Args:
            module (nn.Module): the main module (the model).
            layer (nn.Module): the layer that need to be checked.

        Returns:
            True if module contains the layer, False otherwise.
    
    """
    
    # submodules 
    children = list(module.children())
    
    # base case
    if isinstance(module, layer): return True # type: ignore
    
    # check if layer is a child
    output = False

    for child in children:
        output = output or _has_layer(child, layer)

    return output


@torch.inference_mode()
def _get_output_dim(model: nn.Module,
                    input_shape: Tuple[int, int, int]) -> int:
    """
        Get the input dimentsion of a model.

        Args:
            model (nn.Module): the model.
            input_shape (tuple): a input shape triple (channels, height, width).

        Returns:
            the input dimension of the model.
    """


    # get device of the model
    device = list(model.parameters())[0].device

    # add batch dimension and creare a random array
    shape = [1] + list(input_shape)
    sample = torch.randn(*shape, device=device) 

    # compute the output
    out = model(sample)
    output_dim = out.shape[1]

    return output_dim


def _get_backbone(name: str, from_torchvision: Optional[bool] = False) -> nn.Module:
    """
        Get a backbone given its name. By default is returns the timm model.

        Args:
            name (str): the name of the model.
            from_torchvision (bool): True to load get torchvision model (default:False).

        Returns:
            the nn.Module of the model.
    """

    # CASE 1: Special model not available in torchvision/timm 
    # NOTE: just dino_v2 implemented right now.
    if name in SPECIAL_MODELS:
        if "dino_v2" in name.lower():
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        else:
            raise ValueError(f"Illegal special model: {name}")
        
    # CASE 2: model from TorchVision
    elif from_torchvision:
        getmodel_fnc = getattr(torchvision.models, name)
        model = getmodel_fnc(pretrained=True)
        model_list = list(model.children())
        model = nn.Sequential(*model_list[:-1], nn.Flatten())
    # CASE 3: model from timm 
    else:
        # num_classes=0 means not classifier, just feature extractor
        # (keeping last pooling).
        model = timm.create_model(name, num_classes=0, pretrained=True)   
       
    return model


def _get_bottleneck(model: nn.Module, 
                    out_size: int, 
                    input_shape: Tuple[int, int, int] = (3, 224, 224)) -> nn.Module:
    """
        Get a bottleneck for the given model.
        The bottleneck is a block: Linear -> Normalization -> Activation.
        The normalization can be BatchNorm or LayerNorm and it depends on the model
        (if model uses BatchNorm the bottleneck will use BatchNorm too).
        The activation can be ReLu and GeLu and depends on the model like the 
        normalization.

        Args:
            model (nn.Module): the model that will output the features that will
            be used by the bottleneck.
            out_size (int): output size of the bottleneck.
            input_shape (tuple): input shape of the model.

        Return:
            the bottlenck module.
    """
    if isinstance(input_shape, int):
        input_shape = (3, input_shape, input_shape)
    elif len(input_shape) == 2:
        input_shape = tuple([3] + input_shape)


    bottleneck_in  = _get_output_dim(model, input_shape=input_shape)

    normalization = nn.BatchNorm1d if _has_layer(model, nn.BatchNorm2d) else \
                    nn.LayerNorm
    activation    = nn.ReLU if _has_layer(model, nn.ReLU) else nn.GELU

    return nn.Sequential(nn.Linear(bottleneck_in, out_size),
                         normalization(out_size),
                         activation())


def _get_classifier(in_size: int, 
                    out_size: int, 
                    weight_norm: bool = False,
                    bias: bool = False) -> nn.Module:
    """
        Get a linear classifier.

        Args:
            in_size (int): the input size.
            out_size (int): the output size.
            weight_norm (bool): use weight norm (default: False).
            bias (bool): True to use the bias in the classifier (default: False)
        Returns:
            the linear classifier.
    """

    classifier = nn.Linear(in_size, out_size, bias=bias)

    if weight_norm: classifier = weightNorm(classifier, name="weight")

    return classifier





