#This is the maze class. It is used to generate all mazes in simulation
from euclid import Circle, Point2, Vector2, LineSegment2
import random as r
import copy as cp

class Maze(object):
    def __init__(self,x,y,d):
        self.heroPos = 0
        self.goalPos = 0
        self.mazes = []
        self.smazes = []
        self.x = x
        self.y = y
        self.divider = d
        self.makeMazes()
       
    def getMaze(self,i):
        self.setPositions(i)
        return self.mazes[int(i)]
    
    def getSMaze(self,i):
        return self.smazes[i]
    
    def getGoalPos(self):
        return self.goalPos
    
    def getHeroPos(self):
        return cp.copy(self.heroPos)
    
    def setPositions(self,i):
        if(i == 0.0 or i == 10.0):
            self.heroPos = Point2(45,260)
            self.goalPos = Point2(655,250)
        if(i == 1.0 or i == 0.1):
            self.heroPos = Point2(550,250)
            self.goalPos = Point2(45,250)
        if(i == 2.0):
            self.heroPos = Point2(45,60)
            self.goalPos = Point2(655,60)
        if(i == 3.0 or i == 0.2 or i == 9.1):
            self.heroPos = Point2(670,450)
            self.goalPos = Point2(35,50)
        if(i == 4.0):
            self.heroPos = Point2(45,450)
            self.goalPos = Point2(670,50)
        if(i == 5.0):
            self.heroPos = Point2(500,260)
            self.goalPos = Point2(300,60)
        if(i == 6.0):
            self.heroPos = Point2(655,260)
            self.goalPos = Point2(350,450)
        if(i == 7.0):
            self.heroPos = Point2(350,260)
            self.goalPos = Point2(45,400)
        if(i == 8.0 or i == 9.2):
            self.heroPos = Point2(45,50)
            self.goalPos = Point2(190,420)
        if(i == 9.0):
            self.heroPos = Point2(45,260)
            self.goalPos = Point2(655,400)
        if(i == 11):
            self.heroPos = Point2(self.x - 20,int(self.y/2))
            self.goalPos = Point2(20,int(self.y/2))
           
    def saveMaze(self,maze,name):
        path = "tf_rl/simulation/crazyMazes/" + name
        wtf = open(path,"w")
        wtf.write(str(maze))
        wtf.close()
    
    #not functioning properly
    def reloadMaze(self,name):
        path = "tf_rl/simulation/crazyMazes/" + name
        wtf = open(path,"r")
        maze = wtf.read()
        wtf.close()
        return maze
        
    def makeMazes(self):
        for i in range(0,12):
            self.mazes.append(self.makeMaze(i))
            self.smazes.append(self.makeSMaze(i))
            
    def makeMaze(self,i):
        if(i == 0):
            #return self.wallsOnly()
            return []
        if(i == 1):
            return self.oneObst()
        if(i == 2):
            return self.wallsOnly()
        if(i == 3):
            return self.grid()
        if(i == 4):
            return self.wallsOnly()
        if(i == 5):
            return self.wallsOnly()
        if(i == 6):
            return self.cMaze()
        if(i == 7):
            return self.cMaze1()
        if(i == 8):
            return self.cMaze2()
        if(i == 9):
            return self.crazyCircle()#return self.cMaze3()
        if(i == 10):
            return self.wallsOnly()#self.cMaze3()[:130]
        if(i == 11):
            return self.oneObst()
    
    def makeSMaze(self,i):
        blank = []
        if(i == 0):
            return blank
        if(i == 1):
            return blank
        if(i == 2):
            return self.tightFit()
        if(i == 3):
            return blank
        if(i == 4):
            return self.zigZag()
        if(i == 5):
            return self.twoDoor()
        if(i == 6):
            return self.sMaze()
        if(i == 7):
            return self.sMaze1()
        if(i == 8):
            return self.sMaze2()
        if(i == 9):
            return []
            #return self.sMaze3()
        if(i == 10):
            return []
        if(i == 11):
            return []

    #Circle mazes
    def wallsOnly(self):
        maze = []
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
        maze = []#self.wallsOnly()
        maze.append(Point2(int(self.x/2),int(self.y/2)))
        return maze
    
    def grid(self):
        maze = []
        y = 10
        x = 0
        for i in range(0,108):
            x = i % 14
            if(i % 18==0):
                y += 80
                
            maze.append(Point2(10 + (x*50),y))
            
        for j in range(0,35):
            maze.append(Point2(-5 + (j*20),-5))
        for j in range(0,35):
            maze.append(Point2(-5 + (j*20),500))
            
        for j in range(0,25):
            maze.append(Point2(-5,-5+(j*20)))
        for j in range(0,26):
            maze.append(Point2(700,-5+(j*20)))
        return maze
    
    def crazyCircle(self):
        #maze = self.wallsOnly()
        maze = []
        for i in range(0,30):
            maze.append(Point2(r.randint(70,640),r.randint(0,500)))
        #self.saveMaze(maze,"crazycircle3.txt")
        return maze
    # End Circle Mazes
    
    #Square Mazes

    def tightFit(self):
        maze = []
        for i in range (0,25):
            if(i != 12):
                if(i == 11):
                    maze.append(Point2(350,20 + (i*19)))
                else:
                    maze.append(Point2(350,20+ (i * 20)))
        return maze
    
    def zigZag(self):
        maze = []
        for j in range(0,3):
            for i in range (0,20):
                maze.append(Point2(100 + j*400,20+ (i * 20)))
         
        for j in range(0,1):
            for i in range (0,20):
                maze.append(Point2(300 + j*400,120+ (i * 20)))
        return maze
    
    def twoDoor(self):
        maze = []
        for i in range (0,25):
            if(i != 5 and i!=6 and i!=16 and i!=17):
                maze.append(Point2(350,20+ (i * 20)))
        return maze        
    def crazySquare(self):
        maze = []
        for i in range(0,30):
            maze.append(Point2(r.randint(70,640),r.randint(0,500)))
        #self.saveMaze(maze,"crazysquare3.txt")
        return maze
         
    #End Square Mazes
    
    
    #By Hand mazes
    def cMaze(self):
        return [Point2(-5.00, -5.00), Point2(15.00, -5.00), Point2(35.00, -5.00), Point2(55.00, -5.00), Point2(75.00, -5.00), Point2(95.00, -5.00), Point2(115.00, -5.00), Point2(135.00, -5.00), Point2(155.00, -5.00), Point2(175.00, -5.00), Point2(195.00, -5.00), Point2(215.00, -5.00), Point2(235.00, -5.00), Point2(255.00, -5.00), Point2(275.00, -5.00), Point2(295.00, -5.00), Point2(315.00, -5.00), Point2(335.00, -5.00), Point2(355.00, -5.00), Point2(375.00, -5.00), Point2(395.00, -5.00), Point2(415.00, -5.00), Point2(435.00, -5.00), Point2(455.00, -5.00), Point2(475.00, -5.00), Point2(495.00, -5.00), Point2(515.00, -5.00), Point2(535.00, -5.00), Point2(555.00, -5.00), Point2(575.00, -5.00), Point2(595.00, -5.00), Point2(615.00, -5.00), Point2(635.00, -5.00), Point2(655.00, -5.00), Point2(675.00, -5.00), Point2(-5.00, 500.00), Point2(15.00, 500.00), Point2(35.00, 500.00), Point2(55.00, 500.00), Point2(75.00, 500.00), Point2(95.00, 500.00), Point2(115.00, 500.00), Point2(135.00, 500.00), Point2(155.00, 500.00), Point2(175.00, 500.00), Point2(195.00, 500.00), Point2(215.00, 500.00), Point2(235.00, 500.00), Point2(255.00, 500.00), Point2(275.00, 500.00), Point2(295.00, 500.00), Point2(315.00, 500.00), Point2(335.00, 500.00), Point2(355.00, 500.00), Point2(375.00, 500.00), Point2(395.00, 500.00), Point2(415.00, 500.00), Point2(435.00, 500.00), Point2(455.00, 500.00), Point2(475.00, 500.00), Point2(495.00, 500.00), Point2(515.00, 500.00), Point2(535.00, 500.00), Point2(555.00, 500.00), Point2(575.00, 500.00), Point2(595.00, 500.00), Point2(615.00, 500.00), Point2(635.00, 500.00), Point2(655.00, 500.00), Point2(675.00, 500.00), Point2(-5.00, -5.00), Point2(-5.00, 15.00), Point2(-5.00, 35.00), Point2(-5.00, 55.00), Point2(-5.00, 75.00), Point2(-5.00, 95.00), Point2(-5.00, 115.00), Point2(-5.00, 135.00), Point2(-5.00, 155.00), Point2(-5.00, 175.00), Point2(-5.00, 195.00), Point2(-5.00, 215.00), Point2(-5.00, 235.00), Point2(-5.00, 255.00), Point2(-5.00, 275.00), Point2(-5.00, 295.00), Point2(-5.00, 315.00), Point2(-5.00, 335.00), Point2(-5.00, 355.00), Point2(-5.00, 375.00), Point2(-5.00, 395.00), Point2(-5.00, 415.00), Point2(-5.00, 435.00), Point2(-5.00, 455.00), Point2(-5.00, 475.00), Point2(700.00, -5.00), Point2(700.00, 15.00), Point2(700.00, 35.00), Point2(700.00, 55.00), Point2(700.00, 75.00), Point2(700.00, 95.00), Point2(700.00, 115.00), Point2(700.00, 135.00), Point2(700.00, 155.00), Point2(700.00, 175.00), Point2(700.00, 195.00), Point2(700.00, 215.00), Point2(700.00, 235.00), Point2(700.00, 255.00), Point2(700.00, 275.00), Point2(700.00, 295.00), Point2(700.00, 315.00), Point2(700.00, 335.00), Point2(700.00, 355.00), Point2(700.00, 375.00), Point2(700.00, 395.00), Point2(700.00, 415.00), Point2(700.00, 435.00), Point2(700.00, 455.00), Point2(700.00, 475.00), Point2(700.00, 495.00), Point2(411.00, 377.00), Point2(501.00, 388.00), Point2(470.00, 76.00), Point2(84.00, 450.00), Point2(612.00, 481.00), Point2(340.00, 227.00), Point2(291.00, 53.00), Point2(337.00, 143.00), Point2(139.00, 177.00), Point2(233.00, 115.00), Point2(597.00, 103.00), Point2(627.00, 196.00), Point2(581.00, 6.00), Point2(226.00, 40.00), Point2(507.00, 345.00), Point2(565.00, 331.00), Point2(618.00, 5.00), Point2(376.00, 340.00), Point2(92.00, 135.00), Point2(202.00, 9.00), Point2(365.00, 165.00), Point2(390.00, 387.00), Point2(182.00, 347.00), Point2(611.00, 230.00), Point2(617.00, 482.00), Point2(566.00, 101.00), Point2(614.00, 153.00), Point2(416.00, 369.00), Point2(234.00, 13.00), Point2(471.00, 352.00)]
    
    def sMaze(self):
        return [Point2(123.00, 144.00), Point2(368.00, 96.00), Point2(439.00, 444.00), Point2(354.00, 361.00), Point2(451.00, 399.00), Point2(119.00, 142.00), Point2(555.00, 471.00), Point2(435.00, 380.00), Point2(496.00, 9.00), Point2(419.00, 421.00), Point2(560.00, 286.00), Point2(548.00, 167.00), Point2(160.00, 132.00), Point2(277.00, 482.00), Point2(331.00, 7.00), Point2(519.00, 58.00), Point2(387.00, 403.00), Point2(81.00, 24.00), Point2(636.00, 84.00), Point2(602.00, 137.00), Point2(446.00, 467.00), Point2(74.00, 105.00), Point2(374.00, 499.00), Point2(441.00, 379.00), Point2(213.00, 328.00), Point2(444.00, 474.00), Point2(296.00, 90.00), Point2(510.00, 7.00), Point2(92.00, 351.00), Point2(503.00, 174.00)]
    
    def cMaze1(self):
        return [Point2(-5.00, -5.00), Point2(15.00, -5.00), Point2(35.00, -5.00), Point2(55.00, -5.00), Point2(75.00, -5.00), Point2(95.00, -5.00), Point2(115.00, -5.00), Point2(135.00, -5.00), Point2(155.00, -5.00), Point2(175.00, -5.00), Point2(195.00, -5.00), Point2(215.00, -5.00), Point2(235.00, -5.00), Point2(255.00, -5.00), Point2(275.00, -5.00), Point2(295.00, -5.00), Point2(315.00, -5.00), Point2(335.00, -5.00), Point2(355.00, -5.00), Point2(375.00, -5.00), Point2(395.00, -5.00), Point2(415.00, -5.00), Point2(435.00, -5.00), Point2(455.00, -5.00), Point2(475.00, -5.00), Point2(495.00, -5.00), Point2(515.00, -5.00), Point2(535.00, -5.00), Point2(555.00, -5.00), Point2(575.00, -5.00), Point2(595.00, -5.00), Point2(615.00, -5.00), Point2(635.00, -5.00), Point2(655.00, -5.00), Point2(675.00, -5.00), Point2(-5.00, 500.00), Point2(15.00, 500.00), Point2(35.00, 500.00), Point2(55.00, 500.00), Point2(75.00, 500.00), Point2(95.00, 500.00), Point2(115.00, 500.00), Point2(135.00, 500.00), Point2(155.00, 500.00), Point2(175.00, 500.00), Point2(195.00, 500.00), Point2(215.00, 500.00), Point2(235.00, 500.00), Point2(255.00, 500.00), Point2(275.00, 500.00), Point2(295.00, 500.00), Point2(315.00, 500.00), Point2(335.00, 500.00), Point2(355.00, 500.00), Point2(375.00, 500.00), Point2(395.00, 500.00), Point2(415.00, 500.00), Point2(435.00, 500.00), Point2(455.00, 500.00), Point2(475.00, 500.00), Point2(495.00, 500.00), Point2(515.00, 500.00), Point2(535.00, 500.00), Point2(555.00, 500.00), Point2(575.00, 500.00), Point2(595.00, 500.00), Point2(615.00, 500.00), Point2(635.00, 500.00), Point2(655.00, 500.00), Point2(675.00, 500.00), Point2(-5.00, -5.00), Point2(-5.00, 15.00), Point2(-5.00, 35.00), Point2(-5.00, 55.00), Point2(-5.00, 75.00), Point2(-5.00, 95.00), Point2(-5.00, 115.00), Point2(-5.00, 135.00), Point2(-5.00, 155.00), Point2(-5.00, 175.00), Point2(-5.00, 195.00), Point2(-5.00, 215.00), Point2(-5.00, 235.00), Point2(-5.00, 255.00), Point2(-5.00, 275.00), Point2(-5.00, 295.00), Point2(-5.00, 315.00), Point2(-5.00, 335.00), Point2(-5.00, 355.00), Point2(-5.00, 375.00), Point2(-5.00, 395.00), Point2(-5.00, 415.00), Point2(-5.00, 435.00), Point2(-5.00, 455.00), Point2(-5.00, 475.00), Point2(700.00, -5.00), Point2(700.00, 15.00), Point2(700.00, 35.00), Point2(700.00, 55.00), Point2(700.00, 75.00), Point2(700.00, 95.00), Point2(700.00, 115.00), Point2(700.00, 135.00), Point2(700.00, 155.00), Point2(700.00, 175.00), Point2(700.00, 195.00), Point2(700.00, 215.00), Point2(700.00, 235.00), Point2(700.00, 255.00), Point2(700.00, 275.00), Point2(700.00, 295.00), Point2(700.00, 315.00), Point2(700.00, 335.00), Point2(700.00, 355.00), Point2(700.00, 375.00), Point2(700.00, 395.00), Point2(700.00, 415.00), Point2(700.00, 435.00), Point2(700.00, 455.00), Point2(700.00, 475.00), Point2(700.00, 495.00), Point2(353.00, 215.00), Point2(378.00, 308.00), Point2(357.00, 136.00), Point2(136.00, 473.00), Point2(481.00, 237.00), Point2(439.00, 469.00), Point2(462.00, 201.00), Point2(540.00, 385.00), Point2(347.00, 281.00), Point2(508.00, 74.00), Point2(213.00, 226.00), Point2(84.00, 4.00), Point2(459.00, 402.00), Point2(431.00, 98.00), Point2(250.00, 158.00), Point2(202.00, 36.00), Point2(491.00, 63.00), Point2(234.00, 358.00), Point2(432.00, 21.00), Point2(633.00, 216.00), Point2(225.00, 323.00), Point2(519.00, 222.00), Point2(85.00, 72.00), Point2(582.00, 168.00), Point2(596.00, 130.00), Point2(428.00, 451.00), Point2(270.00, 165.00), Point2(305.00, 159.00), Point2(304.00, 249.00), Point2(73.00, 95.00)]
    
    def sMaze1(self):
        return [Point2(624.00, 29.00), Point2(589.00, 490.00), Point2(457.00, 117.00), Point2(508.00, 436.00), Point2(607.00, 153.00), Point2(287.00, 217.00), Point2(359.00, 233.00), Point2(343.00, 211.00), Point2(166.00, 313.00), Point2(525.00, 299.00), Point2(603.00, 1.00), Point2(243.00, 469.00), Point2(158.00, 385.00), Point2(144.00, 444.00), Point2(490.00, 343.00), Point2(355.00, 329.00), Point2(326.00, 163.00), Point2(162.00, 301.00), Point2(465.00, 362.00), Point2(478.00, 370.00), Point2(585.00, 436.00), Point2(533.00, 484.00), Point2(304.00, 224.00), Point2(622.00, 295.00), Point2(410.00, 43.00), Point2(629.00, 136.00), Point2(522.00, 143.00), Point2(254.00, 338.00), Point2(501.00, 184.00), Point2(564.00, 329.00)]
    
    def cMaze2(self):
        return [Point2(-5.00, -5.00), Point2(15.00, -5.00), Point2(35.00, -5.00), Point2(55.00, -5.00), Point2(75.00, -5.00), Point2(95.00, -5.00), Point2(115.00, -5.00), Point2(135.00, -5.00), Point2(155.00, -5.00), Point2(175.00, -5.00), Point2(195.00, -5.00), Point2(215.00, -5.00), Point2(235.00, -5.00), Point2(255.00, -5.00), Point2(275.00, -5.00), Point2(295.00, -5.00), Point2(315.00, -5.00), Point2(335.00, -5.00), Point2(355.00, -5.00), Point2(375.00, -5.00), Point2(395.00, -5.00), Point2(415.00, -5.00), Point2(435.00, -5.00), Point2(455.00, -5.00), Point2(475.00, -5.00), Point2(495.00, -5.00), Point2(515.00, -5.00), Point2(535.00, -5.00), Point2(555.00, -5.00), Point2(575.00, -5.00), Point2(595.00, -5.00), Point2(615.00, -5.00), Point2(635.00, -5.00), Point2(655.00, -5.00), Point2(675.00, -5.00), Point2(-5.00, 500.00), Point2(15.00, 500.00), Point2(35.00, 500.00), Point2(55.00, 500.00), Point2(75.00, 500.00), Point2(95.00, 500.00), Point2(115.00, 500.00), Point2(135.00, 500.00), Point2(155.00, 500.00), Point2(175.00, 500.00), Point2(195.00, 500.00), Point2(215.00, 500.00), Point2(235.00, 500.00), Point2(255.00, 500.00), Point2(275.00, 500.00), Point2(295.00, 500.00), Point2(315.00, 500.00), Point2(335.00, 500.00), Point2(355.00, 500.00), Point2(375.00, 500.00), Point2(395.00, 500.00), Point2(415.00, 500.00), Point2(435.00, 500.00), Point2(455.00, 500.00), Point2(475.00, 500.00), Point2(495.00, 500.00), Point2(515.00, 500.00), Point2(535.00, 500.00), Point2(555.00, 500.00), Point2(575.00, 500.00), Point2(595.00, 500.00), Point2(615.00, 500.00), Point2(635.00, 500.00), Point2(655.00, 500.00), Point2(675.00, 500.00), Point2(-5.00, -5.00), Point2(-5.00, 15.00), Point2(-5.00, 35.00), Point2(-5.00, 55.00), Point2(-5.00, 75.00), Point2(-5.00, 95.00), Point2(-5.00, 115.00), Point2(-5.00, 135.00), Point2(-5.00, 155.00), Point2(-5.00, 175.00), Point2(-5.00, 195.00), Point2(-5.00, 215.00), Point2(-5.00, 235.00), Point2(-5.00, 255.00), Point2(-5.00, 275.00), Point2(-5.00, 295.00), Point2(-5.00, 315.00), Point2(-5.00, 335.00), Point2(-5.00, 355.00), Point2(-5.00, 375.00), Point2(-5.00, 395.00), Point2(-5.00, 415.00), Point2(-5.00, 435.00), Point2(-5.00, 455.00), Point2(-5.00, 475.00), Point2(700.00, -5.00), Point2(700.00, 15.00), Point2(700.00, 35.00), Point2(700.00, 55.00), Point2(700.00, 75.00), Point2(700.00, 95.00), Point2(700.00, 115.00), Point2(700.00, 135.00), Point2(700.00, 155.00), Point2(700.00, 175.00), Point2(700.00, 195.00), Point2(700.00, 215.00), Point2(700.00, 235.00), Point2(700.00, 255.00), Point2(700.00, 275.00), Point2(700.00, 295.00), Point2(700.00, 315.00), Point2(700.00, 335.00), Point2(700.00, 355.00), Point2(700.00, 375.00), Point2(700.00, 395.00), Point2(700.00, 415.00), Point2(700.00, 435.00), Point2(700.00, 455.00), Point2(700.00, 475.00), Point2(700.00, 495.00), Point2(552.00, 305.00), Point2(546.00, 423.00), Point2(119.00, 385.00), Point2(562.00, 353.00), Point2(623.00, 296.00), Point2(334.00, 81.00), Point2(404.00, 29.00), Point2(250.00, 461.00), Point2(188.00, 448.00), Point2(208.00, 43.00), Point2(235.00, 242.00), Point2(222.00, 409.00), Point2(610.00, 161.00), Point2(436.00, 417.00), Point2(472.00, 427.00), Point2(126.00, 93.00), Point2(487.00, 481.00), Point2(366.00, 270.00), Point2(549.00, 467.00), Point2(156.00, 313.00), Point2(533.00, 356.00), Point2(600.00, 251.00), Point2(560.00, 355.00), Point2(261.00, 422.00), Point2(86.00, 431.00), Point2(334.00, 83.00), Point2(274.00, 221.00), Point2(524.00, 410.00), Point2(352.00, 289.00), Point2(198.00, 343.00)]
    
    def sMaze2(self):
        return [Point2(559.00, 411.00), Point2(223.00, 237.00), Point2(484.00, 371.00), Point2(235.00, 237.00), Point2(454.00, 115.00), Point2(613.00, 214.00), Point2(520.00, 363.00), Point2(112.00, 175.00), Point2(70.00, 122.00), Point2(273.00, 213.00), Point2(182.00, 49.00), Point2(369.00, 251.00), Point2(234.00, 232.00), Point2(611.00, 223.00), Point2(432.00, 337.00), Point2(512.00, 113.00), Point2(402.00, 197.00), Point2(537.00, 444.00), Point2(475.00, 76.00), Point2(202.00, 41.00), Point2(519.00, 214.00), Point2(73.00, 299.00), Point2(191.00, 398.00), Point2(233.00, 256.00), Point2(486.00, 367.00), Point2(339.00, 284.00), Point2(583.00, 104.00), Point2(111.00, 328.00), Point2(222.00, 383.00), Point2(148.00, 72.00)]
    
    def cMaze3(self):
        return [Point2(-5.00, -5.00), Point2(15.00, -5.00), Point2(35.00, -5.00), Point2(55.00, -5.00), Point2(75.00, -5.00), Point2(95.00, -5.00), Point2(115.00, -5.00), Point2(135.00, -5.00), Point2(155.00, -5.00), Point2(175.00, -5.00), Point2(195.00, -5.00), Point2(215.00, -5.00), Point2(235.00, -5.00), Point2(255.00, -5.00), Point2(275.00, -5.00), Point2(295.00, -5.00), Point2(315.00, -5.00), Point2(335.00, -5.00), Point2(355.00, -5.00), Point2(375.00, -5.00), Point2(395.00, -5.00), Point2(415.00, -5.00), Point2(435.00, -5.00), Point2(455.00, -5.00), Point2(475.00, -5.00), Point2(495.00, -5.00), Point2(515.00, -5.00), Point2(535.00, -5.00), Point2(555.00, -5.00), Point2(575.00, -5.00), Point2(595.00, -5.00), Point2(615.00, -5.00), Point2(635.00, -5.00), Point2(655.00, -5.00), Point2(675.00, -5.00), Point2(-5.00, 500.00), Point2(15.00, 500.00), Point2(35.00, 500.00), Point2(55.00, 500.00), Point2(75.00, 500.00), Point2(95.00, 500.00), Point2(115.00, 500.00), Point2(135.00, 500.00), Point2(155.00, 500.00), Point2(175.00, 500.00), Point2(195.00, 500.00), Point2(215.00, 500.00), Point2(235.00, 500.00), Point2(255.00, 500.00), Point2(275.00, 500.00), Point2(295.00, 500.00), Point2(315.00, 500.00), Point2(335.00, 500.00), Point2(355.00, 500.00), Point2(375.00, 500.00), Point2(395.00, 500.00), Point2(415.00, 500.00), Point2(435.00, 500.00), Point2(455.00, 500.00), Point2(475.00, 500.00), Point2(495.00, 500.00), Point2(515.00, 500.00), Point2(535.00, 500.00), Point2(555.00, 500.00), Point2(575.00, 500.00), Point2(595.00, 500.00), Point2(615.00, 500.00), Point2(635.00, 500.00), Point2(655.00, 500.00), Point2(675.00, 500.00), Point2(-5.00, -5.00), Point2(-5.00, 15.00), Point2(-5.00, 35.00), Point2(-5.00, 55.00), Point2(-5.00, 75.00), Point2(-5.00, 95.00), Point2(-5.00, 115.00), Point2(-5.00, 135.00), Point2(-5.00, 155.00), Point2(-5.00, 175.00), Point2(-5.00, 195.00), Point2(-5.00, 215.00), Point2(-5.00, 235.00), Point2(-5.00, 255.00), Point2(-5.00, 275.00), Point2(-5.00, 295.00), Point2(-5.00, 315.00), Point2(-5.00, 335.00), Point2(-5.00, 355.00), Point2(-5.00, 375.00), Point2(-5.00, 395.00), Point2(-5.00, 415.00), Point2(-5.00, 435.00), Point2(-5.00, 455.00), Point2(-5.00, 475.00), Point2(700.00, -5.00), Point2(700.00, 15.00), Point2(700.00, 35.00), Point2(700.00, 55.00), Point2(700.00, 75.00), Point2(700.00, 95.00), Point2(700.00, 115.00), Point2(700.00, 135.00), Point2(700.00, 155.00), Point2(700.00, 175.00), Point2(700.00, 195.00), Point2(700.00, 215.00), Point2(700.00, 235.00), Point2(700.00, 255.00), Point2(700.00, 275.00), Point2(700.00, 295.00), Point2(700.00, 315.00), Point2(700.00, 335.00), Point2(700.00, 355.00), Point2(700.00, 375.00), Point2(700.00, 395.00), Point2(700.00, 415.00), Point2(700.00, 435.00), Point2(700.00, 455.00), Point2(700.00, 475.00), Point2(700.00, 495.00), Point2(145.00, 285.00), Point2(537.00, 171.00), Point2(435.00, 1.00), Point2(448.00, 149.00), Point2(242.00, 372.00), Point2(245.00, 95.00), Point2(495.00, 12.00), Point2(172.00, 383.00), Point2(562.00, 102.00), Point2(527.00, 186.00), Point2(361.00, 483.00), Point2(98.00, 284.00), Point2(329.00, 186.00), Point2(85.00, 191.00), Point2(577.00, 203.00), Point2(576.00, 214.00), Point2(505.00, 478.00), Point2(502.00, 259.00), Point2(218.00, 422.00), Point2(223.00, 297.00), Point2(272.00, 271.00), Point2(501.00, 454.00), Point2(557.00, 493.00), Point2(636.00, 97.00), Point2(410.00, 377.00), Point2(395.00, 70.00), Point2(184.00, 45.00), Point2(510.00, 69.00), Point2(103.00, 156.00), Point2(624.00, 458.00)]
    
    def sMaze3(self):
        return [Point2(91.00, 137.00), Point2(211.00, 387.00), Point2(319.00, 8.00), Point2(481.00, 313.00), Point2(254.00, 88.00), Point2(456.00, 167.00), Point2(206.00, 255.00), Point2(161.00, 454.00), Point2(606.00, 6.00), Point2(290.00, 457.00), Point2(629.00, 375.00), Point2(324.00, 295.00), Point2(200.00, 261.00), Point2(296.00, 41.00), Point2(283.00, 65.00), Point2(102.00, 302.00), Point2(530.00, 490.00), Point2(119.00, 174.00), Point2(159.00, 290.00), Point2(347.00, 245.00), Point2(541.00, 257.00), Point2(424.00, 459.00), Point2(350.00, 111.00), Point2(395.00, 30.00), Point2(435.00, 162.00), Point2(434.00, 252.00), Point2(628.00, 467.00), Point2(506.00, 286.00), Point2(438.00, 124.00), Point2(605.00, 491.00)]
