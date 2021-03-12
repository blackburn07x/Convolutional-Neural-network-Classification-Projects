class DisjointSet:
    def __init__(self, nodes):
        self.nodes = nodes
        self.parent = [-1 for i in range(nodes)]  # make every pixel a paretn of itself
        self.size = [1 for i in range(nodes)]  # size
        self.rank = [0 for i in range(nodes)]  # rank is used for union as smaller nodes merge with bigger ones

    def find(self, i):  # f() to find parent/root recursively
        if self.parent[i] == -1:  # if parent return that pixel
            return i
        self.parent[i] = self.find(self.parent[i])  # else serach for root
        return self.parent[i]

    def union(self, x, y):
        # find parent
        xp = self.find(x)
        yp = self.find(y)

        if xp != yp:  # union based on rank
            if self.rank[xp] > self.rank[yp]:
                self.parent[yp] = xp
                self.size[xp] += self.size[yp]
            else:
                self.parent[xp] = yp
                self.size[yp] += self.size[xp]
            if self.rank[xp] == self.rank[yp]:
                self.rank[yp] += 1
