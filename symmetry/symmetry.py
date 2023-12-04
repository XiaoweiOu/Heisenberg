class Symmetry:
    """
        Base class of symmetry operations
        """
    def __init__(self):
        """
        Define some basic properties of the group
        e.g. order
        """
        raise NotImplementedError

    def symmetrize_config(self, x):
        raise NotImplementedError