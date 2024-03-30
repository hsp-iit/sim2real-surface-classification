from typing import Optional, Callable, List
from composed_model import ComposedModel
import torch.nn as nn
import torch

############################ GRADIENT REVERSAL ##########################################
class GradientReversalFunction(torch.autograd.Function):
    """ Identity function that reverses gradient. """

    @staticmethod
    def forward(ctx, input, coeff=1.):
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class GradientReversalLayer(nn.Module):

    def __init__(self, reversal_coeff: Optional[Callable[[int], float]] = None):
        """
            Initializes the reversal layar.
            This function computes the identity in the foward pass and inverts the sign
            of the gradient and muliply it by a coefficient.

            .. math::
                forward(x) = x
                backward(g) = -reversal_coeff(step) \cdot g

            By default the reversal_coeff is always equal to one, additionally a function
            to compute the coefficient can be used. The function should take an integer
            (the step) as input and should return a float (the coefficient) as output.
            To increase the step use the function step().

            Args:
                reversal_coeff (Callable, optional): function from int to float to 
                compute the reversal coefficient, given the step.

        """
        
        super().__init__()
        self.current_step = 0
        self.coeff_fn = reversal_coeff

    def forward(self, *input):
        return GradientReversalFunction.apply(*input, self.get_coeff())

    def step(self):
        self.current_step += 1

    def get_coeff(self):
        return self.coeff_fn(self.current_step) if self.coeff_fn is not None else 1.
    

#################################### DISCRIMINATOR ######################################
class BaseDiscriminator(nn.Module):
    """ A simple discriminator to distinguish domains. """

    def __init__(self, 
                 input_size: int, 
                 hidden_size: int,
                 leaky_slope: Optional[float] = 0.,
                 spectral_norm: Optional[bool] = False,
                 batch_norm: Optional[bool] = True,
                 layer_norm: Optional[bool] = False,
                 dropout: Optional[float] = 0.):
        """
            Initializes the module.

            Args:
                input_size (int): the input size.
                hidden_size (int): the hidden size.
                leaky_slope (float, optional): if 0 ReLU will be used. Otherwise leaky
                ReLU with given slope.
                spectral_norm (bool, optional): True to use spectral normalization.
                batch_norm (bool, optional): True to use batch normalization layers.
                layer_norm (bool, optional): True to use layer normalization layers.
                dropout (float, optional): the dropout rate.

            Raises:
                ValueError if more than one between [batch_norm, layer_norm, dropout] are
                selected.
        """
        if sum([int(dropout > 0), int(batch_norm), int(layer_norm)]) > 1:
            raise ValueError("Select max one: batch norm, layer norm or dropout!")

        super().__init__()

        self.activation = nn.ReLU(inplace=True) if leaky_slope == 0 else \
                          nn.LeakyReLU(inplace=True, negative_slope=0.01)

        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, 1)

        if spectral_norm:
            self.linear_1 = nn.utils.spectral_norm(self.linear_1)
            self.linear_2 = nn.utils.spectral_norm(self.linear_2)
            self.linear_3 = nn.utils.spectral_norm(self.linear_3)

        self.normalization_1 = nn.Identity()
        self.normalization_2 = nn.Identity()

        if dropout > 0:
            self.normalization_1 = nn.Dropout(dropout)
            self.normalization_2 = nn.Dropout(dropout)
        elif batch_norm:
            self.normalization_1 = nn.BatchNorm1d(hidden_size)
            self.normalization_2 = nn.BatchNorm1d(hidden_size)
        elif layer_norm:
            self.normalization_1 = nn.LayerNorm(hidden_size)
            self.normalization_2 = nn.LayerNorm(hidden_size)


    def forward(self, x):

        x = self.linear_1(x)
        x = self.normalization_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.normalization_2(x)
        x = self.activation(x)
        x = self.linear_3(x)

        return x
    


def _has_layer(module: nn.Module, type) -> bool:

    children = list(module.children())

    if isinstance(module, type):
        return True
    
    output = False

    for child in children:
        output = output or _has_layer(child, type)

    return output


def get_discriminator(model: ComposedModel,
                      input_dim: int,
                      hidden_size: int,
                      spectral_norm: Optional[bool] = False,
                      leaky_slope: Optional[float] = 0.) -> BaseDiscriminator:

    """
        Get a discriminator for a given ComposedModel.

        Args:
            input_dim (int): the input size of the discriminator.
            hidden_size (int): the hidden size of the discriminator.
            spectral_norm (bool): True to use spectral normalization on linear layers.
            leacky_slope (float): float to use leaky ReLU activation.

    """
    if _has_layer(model.backbone, nn.BatchNorm2d):
        batch_norm = True
        layer_norm = False
    else:
        batch_norm = False
        layer_norm = True

    discriminator = BaseDiscriminator(input_size=input_dim, 
                                      hidden_size=hidden_size,
                                      leaky_slope=leaky_slope, 
                                      spectral_norm=spectral_norm, 
                                      batch_norm=batch_norm, 
                                      layer_norm=layer_norm)

    return discriminator



class DannModel(nn.Module):
    def __init__(self,
                 model: ComposedModel,
                 reversal: GradientReversalLayer,
                 discriminator: BaseDiscriminator):
        """
            Initializes the DANN Model.

            Args:

                model (ComposedModel): the backbone/bottleneck/classifier inside a
                ComposedModel.
                reversal (GradientReversalLayer): the layer to reverse gradients.
                discriminator (BaseDiscriminator): the discriminator.
        """

        assert model.classifier is not None, "ComposedModel should have classifier"

        super().__init__()
        self.model = model
        self.reversal = reversal
        self.discriminator = discriminator

    # def forward(self, x: torch.Tensor) -> dict:
    #     return self.model(x)

    def forward(self, x: torch.Tensor) -> dict:
        """
            Args:
                x (torch.Tensor): the input tensor

            Returns:
                (feats, model_outs, discriminator_outs)
        """

        model_outs = self.model(x)
        features = model_outs["features"]
        logits   = model_outs["logits"]
        backbone_features = model_outs["backbone_features"]

        dis_out = self.discriminator(self.reversal(features))

        return {"logits": logits,
                "features": features,
                "backbone_features": backbone_features,
                "discriminator": dis_out}


    def get_param_groups(self, 
                         lr_model: float,
                         lr_discriminator: float, 
                         lr_classifier: float, 
                         lr_bottleneck: Optional[float] = None) -> List[dict]:
        """
            Returns the param group list setting different learning rates for the
            backbone, classifier, bottleneck and discriminator.

            Args:
                lr_model (float): the learning rate for the backbone.
                lr_discriminator (float): the learning rate for the discriminator.
                lr_classifier (float): the learning rate for the classifier,
                lr_bottleneck (float, optional): the learning rate for the bottlenck.

            Returns:
                a list with the param groups.
        """

        param_groups = self.model.get_param_groups(lr_model=lr_model,
                                                   lr_bottleneck=lr_bottleneck,
                                                   lr_classifier=lr_classifier)

        param_groups.append(
            {"params": self.discriminator.parameters(), "lr": lr_discriminator}
        )

        return param_groups
    





