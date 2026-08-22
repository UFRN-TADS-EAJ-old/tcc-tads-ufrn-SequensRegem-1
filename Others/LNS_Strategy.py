from abc import ABC, abstractmethod
import random
import numpy as np
import numpy.typing as npt

class Insertion_methods(ABC):
    @abstractmethod
    def insertion (individual,indexes,labels) -> np.ndarray:
        pass

class Removal_methods(ABC):
    @abstractmethod
    def removal (individual,n_remove) -> list:
        pass

class RandomInsertion(Insertion_methods):
    def insertion (self,individual,indexes,labels) -> npt.NDArray[np.int64]:
        values = random.choices(labels,k=len(indexes))
        for index, _ in enumerate(indexes):
            individual[_] = values[index]
        
        return(individual)

class ModaInsertion(Insertion_methods):

    def __init__(self, columns):
        self.modas = columns

    def insertion (self,individual,indexes,labels) -> npt.NDArray[np.int64]:
        if np.issubdtype(self.modas.dtype, np.floating):
            sum_ = sum(self.modas)
            trends = []
            for i in range(len(indexes)):
                trends.append((sum_ + random.random())/(len(individual)+1))

            for index,value in enumerate(indexes):
                individual[value] = trends[index]

            return(individual)
        else:     
            trends = [max(moda,key=moda.count) for moda in self.modas if moda]

            # for i in range(len(indexes)):
            #     tiny_list = []
            #     for _ in range(5):
            #         tiny_list.append(random.choice(labels))
            #     parts_list.append(tiny_list)

            for i in indexes:
                # moda = max(set(parts_list[i]),key=parts_list[i].count)
                individual[i] = trends[i]
            return(individual)

class RouteRemoval(Removal_methods):
    def removal (self,individual,n_remove) -> list:
        indexes = []
        chosen = random.choice(np.unique(individual))

        indexes = np.where(individual == chosen)[0].tolist()

        if n_remove != -1 and len(indexes) > n_remove:
            indexes = indexes[:n_remove]

        return(indexes)

class SequenceRemoval(Removal_methods):
    def removal (self,individual,n_remove) -> list:
        index = random.randrange(len(individual))

        try:
            final_number = index + n_remove
            value = individual[final_number]
            indexes = list(range(index,final_number))

        except IndexError:
            final_number = index - n_remove
            indexes = list(range(index,final_number,-1))

        return(indexes)

class ClientProximityRemoval(Removal_methods):
    def removal (self,individual,n_remove) -> list:
        index = random.randrange(len(individual))
        chosen = individual[index]
        list_ = []
        indexes = []
        value = 10
        
        for index_2,_ in enumerate(individual):
            if _ == chosen:
                list_.append(abs(index_2-index))

        for i_ in range(n_remove):
            for index_3,_ in enumerate(list_):
                if _ <= value:
                    indexes.append(index_3)
                    list_[index_3] = np.nan
                    break

        
        return(indexes)