from models.classifier import classifier
from models.functional_counterfactual import functional_counterfactual, ClassifierFn, Critic, Abductor, Mechanism

Array = Union[jnp.ndarray, np.ndarray, Any]
Shape = Tuple[int, ...]
PRNGKey = KeyArray
InitFn = Callable[[PRNGKey, Shape], Tuple[Shape, Params]]
ApplyFn = Callable
StaxLayer = Tuple[InitFn, ApplyFn]
StaxLayerConstructor = Callable[..., StaxLayer]
UpdateFn = Callable[[int, OptimizerState, Any, PRNGKey], Tuple[OptimizerState, Array, Any]]
InitOptimizerFn = Callable[[Params], Tuple[OptimizerState, UpdateFn, ParamsFn]]
Model = Tuple[InitFn, ApplyFn, InitOptimizerFn]

__all__ = [classifier, functional_counterfactual, ClassifierFn, Critic, Abductor, Mechanism]
