class NaiveBayesModelPrototype():
    def __init__(self) -> None:
        self.n = 1
        self.k = 1

    def loadData(self) -> None:
        '''
        Params: pd.DataFrame
        self.n -> preprocessing will set group column to be at position [-1]. Find the amount of values in that column.
        self.k -> if there is an id column (this should be set as index?), self.k = cols - 2. else, self.k = cols - 1
        '''
        self.n = 2  #These will be set using the parameters of this method
        self.k = 5
        for i in range(self.n):
            self.__dict__['class_' + str(i) + '_prob'] = 1/self.n
            for j in range(self.k):
                self.__dict__['attr_' + str(j) + '_given_' + str(i)] = 1/self.k

    def _classProbability(self) -> None:
        pass

    def _attrProbGivenClass(self) -> None:
        pass
    
    def trainModel(self) -> None:
        pass

    


trainModel = NaiveBayesModelPrototype()
trainModel.loadData()
trainModel.trainModel()