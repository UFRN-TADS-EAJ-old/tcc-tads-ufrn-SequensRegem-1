from abc import ABC, abstractmethod
import random
import numpy as np
import numpy.typing as npt
from itertools import islice
from collections import Counter

class Insertion_methods(ABC):
    @abstractmethod
    def insertion (individual,indexes,labels) -> np.ndarray:
        pass

class Removal_methods(ABC):
    @abstractmethod
    def removal (individual,n_remove) -> list:
        pass

class RandomInsertion_big(Insertion_methods):
    # def __init__(self,small_or_big):
    #     self.small_or_big = small_or_big

    def insertion (self,individual,indexes,labels) -> npt.NDArray[np.int64]:
        values = random.choices(labels,k=len(indexes))
        for index, _ in enumerate(indexes):
            individual[_] = values[index]
        
        return(individual)

class RandomInsertion_small(Insertion_methods):
    def insertion (self,individual,indexes,labels) -> npt.NDArray[np.int64]:
        values = [random.random() for _ in range(len(indexes))]
        for index, _ in enumerate(indexes):
            individual[_] = values[index]
        
        return(individual)

class ModaInsertion_big(Insertion_methods):
    def __init__(self, columns):
        self.modas = columns

    def insertion (self,individual,indexes,labels) -> npt.NDArray[np.int64]:

        trends = [Counter(moda).most_common(1)[0][0] for moda in self.modas]

        # for i in range(len(indexes)):
        #     tiny_list = []
        #     for _ in range(5):
        #         tiny_list.append(random.choice(labels))
        #     parts_list.append(tiny_list)

        for i in indexes:
            # moda = max(set(parts_list[i]),key=parts_list[i].count)
            individual[i] = trends[i]
        return(individual)

class ModaInsertion_small(Insertion_methods):
    def __init__(self, columns):
        self.modas = columns

    def insertion (self,individual,indexes,labels) -> npt.NDArray[np.int64]:
        sum_ = sum(self.modas)
        trends = []
        for i in range(len(indexes)):
            trends.append((sum_ + random.random())/(len(individual)+1))

        for index,value in enumerate(indexes):
            # moda = max(set(parts_list[i]),key=parts_list[i].count)
            individual[value] = trends[index]

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

class ClientProximityRemoval_big(Removal_methods):
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

class ClienteProximityRemoval_small(Removal_methods):
    def removal (self,individual,n_remove) -> list:
        flag = True

        while flag == True:
            space_1 = random.random()
            space_2 = random.random()
            if space_1 != space_2:
                flag = False

        if space_1 > space_2:
            indexes_all = (i for i, val in enumerate(individual) if space_1 <= val <= space_2)
            indexes = np.fromiter(islice(indexes_all, n_remove), dtype=int)
        else:
           indexes_all = (i for i, val in enumerate(individual) if space_1 <= val <= space_2)
           indexes = np.fromiter(islice(indexes_all, n_remove), dtype=int)

        return(indexes)