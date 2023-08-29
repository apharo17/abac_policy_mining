
class Biclique():

    #A class for a biclique

    def __init__(self, id, usrids, resids):
        #id is a non-negative integer
        #usrids is a list of user ids
        #resids is a list of resource ids
        self.id = id
        self.usrids = usrids
        self.resids = resids
        self.pattern = None

    def get_id(self):
        return self.id

    def get_usrids(self):
        return self.usrids

    def get_resids(self):
        return self.resids

    def get_pattern(self):
        return self.pattern

    def set_pattern(self, pattern):
        #pattern is an attribute homogeneous pattern
        self.pattern = pattern
