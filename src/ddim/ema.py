import torch


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.state_dict().items():
            self.shadow[name] = param.clone().detach()

    def update(self, model):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                self.shadow[name].copy_(self.decay * self.shadow[name] + (1.0 - self.decay) * param)

    def store(self, model):
        self.backup = {name: param.clone() for name, param in model.state_dict().items()}

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        if self.backup:
            model.load_state_dict(self.backup, strict=True)
            self.backup = {}

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.shadow = {k: v.clone() for k, v in state_dict["shadow"].items()}
