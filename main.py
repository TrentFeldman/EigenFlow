import math
import numpy as np

class Func:
    def __init__(self, c, p):
        self.mult = c
        self.exp = p
        self.label = 1 if p > 1 else 0   # 1 = converges, 0 = diverges
        self.values = []

        x = 1
        for slot in range(20):
            # log(c / x^p) = log(c) - p*log(x), so no huge numbers are ever built
            result = math.log(self.mult) - self.exp * math.log(x)
            self.values.append(result)
            x *= 2

    def __repr__(self):
        return (
            f"{self.mult:.3f} / x^{self.exp:.3f}  label={self.label}\n"
            f"Log values: {[round(v, 2) for v in self.values]}\n"
        )


rng = np.random.default_rng()
data = []

for counter in range(100000):
    c = 10 ** rng.uniform(-2.0, 2.0)   # log-uniform between 0.01 and 100
    p = rng.uniform(0.0, 2.0)
    data.append(Func(c, p))



# Extract the inputs and labels
X = np.array([func.values for func in data], dtype=np.float32)
y = np.array([func.label for func in data], dtype=np.int64)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("First sample:", X[0])
print("First label:", y[0])