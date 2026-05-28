import numpy as np
from designer import HPADesigner


class HPA303(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(n_div=n_div, max_plys=20, level=level, AIRFOIL=AIRFOIL, WIRE=True, DIHEDRAL=True, PAYLOAD=True)
        self.nf = 3
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3
        f = np.array([self.drag + g, self.y_aoa[0] + g, -self.payload + 10 * g])
        return f


def objective_function(x, defaults, whether_to_optimize):
    x = np.array(x).flatten()
    x_new = defaults.copy()

    j = 0
    for i in range(len(whether_to_optimize)):
        if whether_to_optimize[i] == 1:
            x_new[i] = x[j]
            j += 1

    func = HPA303(n_div=7, level=1, NORMALIZED=True)

    return func(x_new)