#This is the maze class. It is used to generate all mazes in simulation
from euclid import Circle, Point2, Vector2, LineSegment2

class Maze(object):
    def __init__(self):
        self.heroPos = 0
        self.goalPos = 0
        self.mazes = []
        self.makeMazes()
       
    def getMaze(self,i):
        return self.mazes[i]
    
    def getGoalPos(self):
        return self.goalPos
    
    def getHeroPos(self):
        return self.heroPos
    
    def makeMazes(self):
        for i in range(0,2):
            self.mazes.append(self.makeMaze(i))
    
    def makeMaze(self,i):
        if(i == 0):
            return self.wallsOnly()
        if(i == 1):
            return self.oneObst()
    
    def wallsOnly(self):
        maze = []
        self.heroPos = Point2(45,250)
        self.goalPos = Point2(80,250)
        for j in range(0,35):
            maze.append(Point2(-5 + (j*20),-5))
        for j in range(0,35):
            maze.append(Point2(-5 + (j*20),500))
            
        for j in range(0,25):
            maze.append(Point2(-5,-5+(j*20)))
        for j in range(0,26):
            maze.append(Point2(700,-5+(j*20)))
        return maze

    def oneObst(self):
        maze = self.wallsOnly()
        maze.append(Point2(350,250))
        return maze