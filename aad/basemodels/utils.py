import copy

import torch.nn as nn

from .model_iris import IrisNN


def copy_model(model,
               num_features=None,
               hidden_nodes=None,
               num_classes=None,
               pretrained=False):
    """Create a copy of a model"""
    assert isinstance(model, nn.Module)
    Model = model.__class__
    loss_fn = model.loss_fn
    optimizer = model.optimizer
    optim_params = model.optim_params
    scheduler = model.scheduler
    scheduler_params = model.scheduler_params
    from_logits = model.from_logits
    if Model == IrisNN:
        clone = IrisNN(
            loss_fn=loss_fn,
            optimizer=optimizer,
            optim_params=optim_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            from_logits=from_logits,
            num_features=num_features,
            hidden_nodes=hidden_nodes,
            num_classes=num_classes,
        )
    else:
        clone = Model(
            loss_fn=loss_fn,
            optimizer=optimizer,
            optim_params=optim_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            from_logits=from_logits,
        )

    if pretrained:
        state_dict = copy.deepcopy(model.state_dict())
        clone.load_state_dict(state_dict)

    return clone
