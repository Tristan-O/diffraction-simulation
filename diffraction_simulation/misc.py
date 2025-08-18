class KeyAwareDefaultDict(dict):
    def __init__(self, factory, **kwargs):
        self.factory = factory
        self.update(**kwargs)
    def __missing__(self,key:str):
        self[key] = self.factory(key)
        return self[key]