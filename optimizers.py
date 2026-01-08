
import numpy as np

class Optimizer:
    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        raise NotImplementedError

    def reset(self):
        pass


class SGD(Optimizer):
    def step(self, params, grads):
        return params - self.lr * grads


class Momentum(Optimizer):
    def __init__(self, lr, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.momentum * self.v - self.lr * grads
        return params + self.v

    def reset(self):
        self.v = None


class Nesterov(Optimizer):
    def __init__(self, lr, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.v = None

    def step(self, params, grads_fn):
        if self.v is None:
            self.v = np.zeros_like(params)

        lookahead = params + self.momentum * self.v
        grads = grads_fn(lookahead)

        self.v = self.momentum * self.v - self.lr * grads
        return params + self.v

    def reset(self):
        self.v = None


class AdaGrad(Optimizer):
    def __init__(self, lr, eps=1e-8):
        super().__init__(lr)
        self.eps = eps
        self.h = None

    def step(self, params, grads):
        if self.h is None:
            self.h = np.zeros_like(params)
        self.h += grads ** 2
        return params - self.lr * grads / (np.sqrt(self.h) + self.eps)

    def reset(self):
        self.h = None


class RMSProp(Optimizer):
    def __init__(self, lr, beta=0.9, eps=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self.h = None

    def step(self, params, grads):
        if self.h is None:
            self.h = np.zeros_like(params)
        self.h = self.beta * self.h + (1 - self.beta) * grads ** 2
        return params - self.lr * grads / (np.sqrt(self.h) + self.eps)

    def reset(self):
        self.h = None


class Adam(Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        self.m, self.v, self.t = None, None, 0
