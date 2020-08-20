from torch.nn import Module


class FreezableModule(Module):
    """torch.nn.Module with an instance method freeze() for freezing parameters.
    """

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
