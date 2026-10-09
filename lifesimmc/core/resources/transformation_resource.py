from dataclasses import dataclass
from typing import Callable

from torch import Tensor


@dataclass
class TransformationResource:
    """Class representation of the transformation resource.

    Parameters
    ----------
    name : str
        The name of the transformation.
    transformation : Callable[[Tensor], Tensor]
        The transformation function that takes a Tensor as input and returns a Tensor.
    dof_lost : int
        Number of degrees of freedom this transformation removes from the data (e.g. from a mean-subtraction
        projection). Statistical tests that assume a full-rank residual (e.g. chi-squared dof, least-squares
        covariance scaling) should subtract this from their degrees of freedom. Default is 0.
    """
    name: str
    transformation: Callable[[Tensor], Tensor]
    dof_lost: int = 0
