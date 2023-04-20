class Model(object):

    def __init__(self, num_visible, num_hidden=[256]):
        self.num_visible = 0
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_layer = len(num_hidden)

    def log_val(self, v):
        pass

    def log_val_diff(self, v1, v2):
        pass

    def derlog(self, v, size):
        pass

    def get_parameters(self):
        pass

    def visualize_param(self):
        pass

    def is_real(self):
        pass

    def is_complex(self):
        pass

    def is_probability(self):
        return False