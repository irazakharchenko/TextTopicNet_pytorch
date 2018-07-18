class SelectiveSequential(nn.Module):
    def __init__(self, to_select, modules_dict):
        super(SelectiveSequential, self).__init__()
        for key, module in modules_dict.items():
            self.add_module(key, module)
        self._to_select = to_select
    
    def forward(x):
        list = []
        for name, module in self._modules.iteritems():
            x = module(x)
            if name in self._to_select:
                list.append(x)
        return list

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = SelectiveSequential(
            ['conv1', 'conv3'],
            {'conv1': nn.Conv2d(1, 1, 3),
             'conv2': nn.Conv2d(1, 1, 3),
             'conv3': nn.Conv2d(1, 1, 3)}
        )

    def forward(self, x):
        return self.features(x)