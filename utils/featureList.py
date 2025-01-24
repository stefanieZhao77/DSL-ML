class FeatureList:
    def __init__(self, parent, dataset, feature, operator, value):
        self.dataset = dataset
        self.feature = feature
        self.operator = operator if operator else None
        self.value = value if value else None
        
    