import dataclasses
from typing import Optional, Any

import numpy as np


@dataclasses.dataclass
class BoxTreeNode:
    k: Optional[float] = None
    left: Optional["BoxTreeNode"] = None
    right: Optional["BoxTreeNode"] = None
    parent: Optional["BoxTreeNode"] = None
    interval: Optional[np.ndarray] = None
    f_min: Optional[float] = None
    f_best: Optional[float] = None
@dataclasses.dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = dataclasses.field(compare=False)
