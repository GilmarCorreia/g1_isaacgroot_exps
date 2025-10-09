from abc import ABC, abstractmethod

class LaunchInterface(ABC):

    def __init__(self, name):
        self.__setInterfaceName(name)
        self.__setPkgName("g1_isaacgroot_exps")
        self.__nodes = []

        self.createInterfaceNodes()

    ########### SETTERS ###########
    def __setPkgName(self, pkg_name):
        self.__pkg_name = pkg_name

    def __setInterfaceName(self, name):
        self.__interfaceName = name

    ########### GETTERS ###########
    def getPkgName(self):
        return self.__pkg_name

    def getInterfaceName(self):
        return self.__interfaceName
    
    def getInterfaceNodes(self):
        return self.__nodes

    ########### METHODS ###########
    def addNode(self, node):
        self.__nodes.append(node)

    @abstractmethod
    def createInterfaceNodes(self):
        pass