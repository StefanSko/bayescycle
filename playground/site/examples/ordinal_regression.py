from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Ordered
from bayeswire.distributions import Normal, OrderedLogistic


@model
class OrdinalRegression:
    n_cutpoints = Data.scalar()
    x = Data.vector()

    beta = Param(Normal(0.0, 1.0))
    cutpoints = Param(Normal(0.0, 2.0), size=n_cutpoints, constraint=Ordered())

    eta = beta * x
    y = Observed(OrderedLogistic(eta, cutpoints))
