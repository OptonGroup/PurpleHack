"""
Модуль-заглушка для тестирования без PyTorch.

Этот модуль предоставляет минимальные заглушки для классов и функций PyTorch,
чтобы можно было запустить код без фактической установки PyTorch.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.warning("Используется mock_torch! Это только для демонстрации, реальное обучение невозможно.")

class Tensor:
    """Заглушка для torch.Tensor."""
    
    def __init__(self, data):
        if isinstance(data, (list, tuple, np.ndarray)):
            self.data = np.array(data, dtype=np.float32)
        else:
            self.data = np.array([data], dtype=np.float32)
        self.shape = self.data.shape
    
    def __repr__(self):
        return f"MockTensor({self.data})"
    
    def item(self):
        if self.data.size == 1:
            return float(self.data.flatten()[0])
        raise ValueError("Невозможно преобразовать тензор размерности > 1 в скаляр")
    
    def numpy(self):
        return self.data
    
    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self.data, axis=dim))
    
    def squeeze(self, dim=None):
        if dim is None:
            return Tensor(np.squeeze(self.data))
        return Tensor(np.squeeze(self.data, axis=dim))
    
    def __getitem__(self, idx):
        return Tensor(self.data[idx])
    
    def max(self, dim):
        values = np.max(self.data, axis=dim)
        indices = np.argmax(self.data, axis=dim)
        return Tensor(values), Tensor(indices)
    
    def argmax(self):
        return Tensor(np.argmax(self.data))
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        return Tensor(self.data + other)
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        return Tensor(self.data * other)
    
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        return Tensor(self.data - other)
    
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        return Tensor(self.data / other)
    
    def backward(self):
        logger.debug("Вызван метод backward() (заглушка)")

def FloatTensor(data):
    """Заглушка для torch.FloatTensor."""
    return Tensor(data)

def LongTensor(data):
    """Заглушка для torch.LongTensor."""
    return Tensor(data)

def ones_like(tensor):
    """Заглушка для torch.ones_like."""
    return Tensor(np.ones_like(tensor.data))

def zeros_like(tensor):
    """Заглушка для torch.zeros_like."""
    return Tensor(np.zeros_like(tensor.data))

def argmax(tensor, dim=None):
    """Заглушка для torch.argmax."""
    if dim is None:
        return Tensor(np.argmax(tensor.data))
    return Tensor(np.argmax(tensor.data, axis=dim))

def no_grad():
    """Заглушка для torch.no_grad()."""
    class NoGradContext:
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
    
    return NoGradContext()

class nn:
    """Заглушка для torch.nn."""
    
    class Module:
        """Заглушка для torch.nn.Module."""
        
        def __init__(self):
            self.training = True
        
        def parameters(self):
            return []
        
        def __call__(self, x):
            return self.forward(x)
        
        def forward(self, x):
            raise NotImplementedError
        
        def train(self):
            self.training = True
        
        def eval(self):
            self.training = False
        
        def to(self, device):
            return self
        
        def load_state_dict(self, state_dict):
            logger.debug("Вызван метод load_state_dict() (заглушка)")
        
        def state_dict(self):
            return {}
    
    class Linear(Module):
        """Заглушка для torch.nn.Linear."""
        
        def __init__(self, in_features, out_features):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = Tensor(np.random.randn(out_features, in_features) * 0.1)
            self.bias = Tensor(np.zeros(out_features))
        
        def forward(self, x):
            # Простая линейная операция: y = x * W^T + b
            return Tensor(np.dot(x.data, self.weight.data.T) + self.bias.data)

class functional:
    """Заглушка для torch.nn.functional."""
    
    @staticmethod
    def relu(x):
        """Заглушка для torch.nn.functional.relu."""
        return Tensor(np.maximum(x.data, 0))
    
    @staticmethod
    def smooth_l1_loss(input, target):
        """Заглушка для torch.nn.functional.smooth_l1_loss."""
        diff = input.data - target.data
        abs_diff = np.abs(diff)
        mask = abs_diff < 1.0
        loss = np.where(mask, 0.5 * diff * diff, abs_diff - 0.5)
        return Tensor(np.mean(loss))

class optim:
    """Заглушка для torch.optim."""
    
    class Adam:
        """Заглушка для torch.optim.Adam."""
        
        def __init__(self, params, lr=0.001):
            self.params = list(params)
            self.lr = lr
        
        def zero_grad(self):
            logger.debug("Вызван метод zero_grad() (заглушка)")
        
        def step(self):
            logger.debug("Вызван метод step() (заглушка)")
        
        def state_dict(self):
            return {"lr": self.lr}
        
        def load_state_dict(self, state_dict):
            if "lr" in state_dict:
                self.lr = state_dict["lr"]

def save(obj, path):
    """Заглушка для torch.save."""
    logger.debug(f"Вызвана функция save() с путем {path} (заглушка)")

def load(path):
    """Заглушка для torch.load."""
    logger.debug(f"Вызвана функция load() с путем {path} (заглушка)")
    return {} 