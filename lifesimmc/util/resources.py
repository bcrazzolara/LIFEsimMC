from typing import Union, Callable

from lifesimmc.core.modules.base_module import BaseModule


def get_transformations_from_resource_name(module: BaseModule, resource_name: Union[str, tuple[str]]) -> list[Callable]:
    if resource_name is None:
        transformations = []
    elif isinstance(resource_name, tuple):
        transformations = [module.get_resource_from_name(
            name).transformation for name in resource_name]
    else:
        transformations = [module.get_resource_from_name(
            resource_name).transformation]
    return transformations


def get_dof_lost_from_resource_name(module: BaseModule, resource_name: Union[str, tuple[str]]) -> int:
    """Sum the degrees of freedom removed by the named transformation resource(s) (e.g. mean-subtraction in
    ZCAWhiteningModule), for use in dof/covariance corrections in downstream statistical tests."""
    if resource_name is None:
        names = []
    elif isinstance(resource_name, tuple):
        names = list(resource_name)
    else:
        names = [resource_name]
    return sum(module.get_resource_from_name(name).dof_lost for name in names)
