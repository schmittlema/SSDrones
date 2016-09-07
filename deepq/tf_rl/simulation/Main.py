import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from tf_rl.simulation import Mazes
import os

from collections import defaultdict
from euclid import Circle, Point2, Vector2, LineSegment2

import tf_rl.simulate
import tf_rl.utils.svg as svg
import tf_rl.utils.geometry as geo

from matplotlib.backends.backend_pdf import PdfPages

class GameObject(object):
    def __init__(self, position, speed, acceleration, obj_type, settings):
        """Esentially represents circles of different kinds, which have
        position and speed."""
        self.settings= settings
        self.maximum_speed = settings["maximum_speed"]
        self.radius = self.settings["object_radius"]
        self.obj_type = obj_type
        self.position = position
        self.speed = speed 
        self.acceleration= acceleration
        self.bounciness = 1.0

    def wall_collisions(self):
        """Update speed upon collision with the wall."""
        world_size = self.settings["world_size"]

        for dim in range(2):
            if self.position[dim] - self.radius       <= 0               and self.speed[dim] < 0:
                self.speed[dim] = - self.speed[dim] * self.bounciness
            elif self.position[dim] + self.radius + 1 >= world_size[dim] and self.speed[dim] > 0:
                self.speed[dim] = - self.speed[dim] * self.bounciness

    def capSpeed(self,speed):
        if(speed[0] > self.maximum_speed[0] or speed[0] < -self.maximum_speed[0]):
            n = 1
            if speed[0] < 0:
                n = -1
            speed[0] =  n * self.maximum_speed[0]
        if(speed[1] > self.maximum_speed[1] or speed[1] < -self.maximum_speed[1]):
            n = 1
            if speed[1] < 0:
                n = -1
            speed[1] = n * self.maximum_speed[1]
        return speed

    def speeding(self,speed):
        if(speed[0] > self.maximum_speed[0] or speed[0] < -self.maximum_speed[0]):
            return True
        if(speed[1] > self.maximum_speed[1] or speed[1] < -self.maximum_speed[1]):
            return True
        return False

    def move(self, dt):
        """Move as if dt seconds passed"""
        #implement new physics
        if self.settings["add_physics"]: 
            if not self.speeding(self.speed+dt*self.acceleration):
                self.position+= self.speed*dt+(dt**2)/2*self.acceleration #updated physics
                self.speed+= dt*self.acceleration
                self.position = Point2(*self.position)
            else:      #don't allow agent to accelerate if its speed exceeds max
                self.position+=self.speed*dt
                self.position=Point2(*self.position)
        #revert to the physics of the original experiment  
        else:
            """Move as if dt seconds passed"""
            self.speed = self.capSpeed(self.speed)
            self.position +=(self.speed*dt)
            self.position = Point2(*self.position)

    def step(self, dt):
        """Move and bounce of walls."""
        self.move(dt)

    def as_square(self):
        return Square(self.position,float(self.radius),float(self.radius))
    
    def as_circle(self):
        return Circle(self.position, float(self.radius))

    def draw(self):
        """Return svg object for this item."""
        color = self.settings["colors"][self.obj_type]
        if(self.obj_type == "square"):
            return svg.Rectangle(self.position - Point2(10,10),Point2(2*self.radius,2*self.radius),color=color)
        else:
            return svg.Circle(self.position + Point2(10, 10), self.radius, color=color)

class Main(object):
    def __init__(self, settings,brainName):
        """Initiallize game simulator with settings"""
        self.settings = settings
        self.size  = self.settings["world_size"]
        self.walls = [LineSegment2(Point2(0,0),                        Point2(0,self.size[1])),
                      LineSegment2(Point2(0,self.size[1]),             Point2(self.size[0], self.size[1])),
                      LineSegment2(Point2(self.size[0], self.size[1]), Point2(self.size[0], 0)),
                      LineSegment2(Point2(self.size[0], 0),            Point2(0,0))]

        self.hero = GameObject(Point2(*self.settings["hero_initial_position"]),
                               Vector2(*self.settings["hero_initial_speed"]),
                               Vector2(*self.settings["hero_initial_accel"]),
                               "hero",
                               self.settings)
        self.brainName = brainName
        self.maze = []
        self.mazeindex = 0
        #Change mazeIterator to change which maze to start on
        self.mazeIterator = 1 
        self.mazeObject = Mazes.Maze()
        self.startTime = time.strftime("%d:%m:%Y:%H")
        self.offset = 0.0
        
        self.display = False
        self.counter = 0
        
        self.successArray = []
        self.successRate = 0.0
        self.crashRate = 0.0
        self.timeOut = 0
        self.timeStart = time.time()
        self.timeoutArray = []
        self.runs = 0
        
        #Stats for overall
        self.averageRuns = []
        self.averageSuccessRate = []
        self.averageTimeout = []
        
        #Squares
        self.smaze = []
        self.smazeindex = 0
        self.makeMaze()
        #print self.maze
            
        #plot run reward
        self.runReward = []
        fileName = brainName + ".pdf"
        print(fileName)
        print(brainName + ".txt")
        self.pp = PdfPages("../RunData/" + fileName)
        self.currReward = 0
        
        #Timing
        self.learntime = 0
        self.time = 0
        self.minTime = 0

        #Rotation 
        self.heading = 0

        if not self.settings["hero_bounces_off_walls"]:
            self.hero.bounciness = 0.0

        self.objects = []
        for obj_type, number in settings["num_objects"].items():
            if(obj_type == "enemy"):
                number = len(self.maze)
            if(obj_type == "square"):
                number = len(self.smaze)
            for _ in range(number):
                self.spawn_object(obj_type)

        self.observation_lines = self.generate_observation_lines()

        self.object_reward = 0
        self.collected_rewards = []

        # every observation_line sees the nearest friend or enemy
        #edit: agent is no longer able to see walls, nearest friend, or the speed of the nearest object bc these things have been taken out
        if self.settings["mod_observation"]:
            self.eye_observation_size = len(self.settings["objects"])-1
        else:
            self.eye_observation_size= len(self.settings["objects"])+3
        # additionally there are two numbers representing the heading vector  and the objects position.
        self.observation_size = self.eye_observation_size * len(self.observation_lines) + 2 + 2
        #directions of movement  
        self.directions = [Vector2(*d) for d in [[1,0], [0,1], [-1,0],[0,-1],[0.0,0.0]]]
        self.num_actions = len(self.directions)
        self.moves = [-60,60,0,0,0]

        self.objects_eaten = defaultdict(lambda: 0)
    
    def makeMaze(self):
        self.maze = self.mazeObject.getMaze(float(self.mazeIterator))
        self.smaze = self.mazeObject.getSMaze(self.mazeIterator)
        self.hero.position = self.mazeObject.getHeroPos()
        self.mazeIterator += 1
    
    def convert_to_cartesian(self,heading):
        return [math.cos(heading),math.sin(heading)]
        
    def perform_action(self, action_id):
        """Change speed to one of hero vectors"""
        assert 0 <= action_id < self.num_actions #check to see if valid action
        if self.settings["Rotation"]:
            self.heading = self.heading + self.moves[action_id]
            self.hero.speed += self.convert_to_cartesian(self.heading) * self.settings["delta_v"]
            
        elif self.settings["add_physics"]: 
            self.hero.acceleration= self.directions[action_id]*self.settings["accel"]
        else:
            self.hero.speed+=self.directions[action_id] * self.settings["delta_v"] 

    def empty(self,array):
        return len(array) == 0
    
    def spawn_object(self, obj_type):
        """Spawn object of a given type and add it to the objects array"""
        empty = False
        radius = self.settings["object_radius"]
        if(obj_type == 'friend'):
            position = self.mazeObject.getGoalPos()
        if(obj_type == "square"):
            if(not(self.empty(self.smaze))):
                position = self.smaze[self.smazeindex]
                self.smazeindex += 1
            else:
                empty = True
        if(obj_type == "enemy"):
            if(not(self.empty(self.maze))):
                position = self.maze[self.mazeindex]
                self.mazeindex += 1 
            else:
                empty = True
        

        max_speed = np.array(self.settings["maximum_speed"])
        speed = Vector2(0, 0)
        acceleration = Vector2(0,0)
        if(not empty):
            self.objects.append(GameObject(position, speed, acceleration,  obj_type, self.settings))
        self.minTime =  self.mazeObject.heroPos.distance(self.mazeObject.goalPos)/(self.settings["maximum_speed"][0])

    def step(self, dt,fps,actionEvery):
        """Simulate all the objects for a given ammount of time.

        Also resolve collisions with the hero"""
        self.counter += 1
        self.time = self.time + dt
        for obj in self.objects + [self.hero] :
            obj.step(dt)
        self.resolve_collisions(fps,actionEvery)
        
    def runComplete(self):
        self.timeOut = (self.time)/self.minTime
        self.timeoutArray.append(self.timeOut)
        self.time = 0
        
    def runFail(self):
        self.timeStart = time.time()
            

    def squared_distance(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def updateSuccess(self,ob):
        if(len(self.successArray) == 100):
                self.successArray = self.successArray[1:]
        if ob.obj_type == "friend":
            self.successArray.append(1)
        self.runComplete()
        if ob.obj_type == "enemy" or ob.obj_type == "square":
            self.successArray.append(0)
            self.runFail()
        self.successRate = sum(self.successArray,0.0)/float(len(self.successArray))
        self.crashRate = 1.0 - self.successRate
    
    def saveData(self):
        dir = "../RunData/"
        name =dir + self.brainName + ".txt"
        wtf = open(name,"a")
        wtf.write("Maze: " + str(self.mazeIterator -1) + "\n")
        wtf.write("Runs: " + str(self.runs) + "\n")
        self.averageRuns.append(self.runs)
        wtf.write("SuccessRate: " + str(self.successRate) + "\n")
        self.averageSuccessRate.append(self.successRate)
        wtf.write("CrashRate: " + str(self.crashRate) + "\n")
        try:
            time = sum(self.timeoutArray,0.0)/float(len(self.timeoutArray))
        except ZeroDivisionError:
            time = 0
        self.averageTimeout.append(time)
        wtf.write("AverageTimout: " + str(time) + "\n"+"\n")
        wtf.close()

    def saveTotals(self):
        if(self.mazeIterator == 1):
            self.saveData()
        dir = "../RunData/"
        name =dir + self.brainName + ".txt"
        wtf = open(name,"a")
        wtf.write("Overall:"+ "\n")
        try:
            wtf.write("Runs: " + str(sum(self.averageRuns,0.0)/float(len(self.averageRuns))) + "\n")
        except(ZeroDivisionError):
            wtf.write("Runs: " + "\n")
        try:
            success = sum(self.averageSuccessRate,0.0)/float(len(self.averageSuccessRate))
            wtf.write("SuccessRate: " + str(success) + "\n")
        except(ZeroDivisionError):
            wtf.write("SuccessRate: " + "\n")
            success = 1
        wtf.write("CrashRate: " + str(1-success) + "\n")
        
        try:
            wtf.write("AverageTimout: " + str(sum(self.averageTimeout,0.0)/float(len(self.averageTimeout))) + "\n"+"\n")
        except(ZeroDivisionError):
            wtf.write("AverageTimeout: " + "\n" + "\n")
 
        wtf.close()    

    def random_selector(self,a,b):
        if a > b:
            return random.randint(b,a)    
        else:
            return random.randint(a,b)
    
    def reassign(self):
        collision_distance = 2 * self.settings["object_radius"]
        collision_distance2 = collision_distance ** 2
        done = False
        self.mazeObject.heroPos = Point2(random.randint(10,650),random.randint(10,450))
        self.mazeObject.goalPos = Point2(random.randint(10,650),random.randint(10,450))
        obstpos = Point2(self.random_selector(self.mazeObject.heroPos[0],self.mazeObject.goalPos[0]),self.random_selector(self.mazeObject.heroPos[1],self.mazeObject.goalPos[1]))
        while(not(done)):
            for obj in self.objects:
                if (obj.obj_type == "enemy"):
                    obj.position = obstpos
                if self.squared_distance(self.mazeObject.heroPos, obj.position) < collision_distance2 + 30:
                    done = False
                    self.mazeObject.heroPos = Point2(random.randint(10,650),random.randint(10,450))
                    obstpos = Point2(self.random_selector(self.mazeObject.heroPos[0],self.mazeObject.goalPos[0]),self.random_selector(self.mazeObject.heroPos[1],self.mazeObject.goalPos[1]))
                if self.squared_distance(self.mazeObject.goalPos, obj.position) < collision_distance2 + 30:
                    done = False
                    self.mazeObject.goalPos = Point2(random.randint(10,650),random.randint(10,450))
                    obstpos = Point2(self.random_selector(self.mazeObject.heroPos[0],self.mazeObject.goalPos[0]),self.random_selector(self.mazeObject.heroPos[1],self.mazeObject.goalPos[1]))
                else:
                    done = True
        self.minTime =  self.mazeObject.heroPos.distance(self.mazeObject.goalPos)/(self.settings["maximum_speed"][0])
                    

    def nextMaze(self, obj):
        if obj.obj_type == "friend":
            self.reassign()
            self.hero.position = self.mazeObject.getHeroPos()
            obj.position = self.mazeObject.getGoalPos()
            for obj in self.objects:
                if(obj.obj_type == "hero"):
                    obj.position = self.hero.position
        """see if the agent has met the criteria for advancement, advance if it has"""
        '''if(self.runs >= 100 and self.successRate >= self.settings["minimum_success_rate"]):
            self.offset = 0.0
            self.saveData()
            self.timeoutArray = []
            self.runs = 0
            self.objects_eaten["friend"] = 0
            self.objects_eaten["enemy"] = 0
            self.successRate = 0.0
            self.successArray = []
            self.makeMaze()
            self.mazeindex = 0
            self.plot_run_reward(smoothing=100)
            self.runReward = []
            self.objects = []
            for obj_type, number in self.settings["num_objects"].items():
                if(obj_type == "enemy"):
                    number = len(self.maze)
                if(obj_type == "square"):
                    number = len(self.smaze)
                for _ in range(number):
                    self.spawn_object(obj_type)
        elif((self.mazeIterator - 1.0 < 1 ) and  obj.obj_type == "friend"):
            self.mazeObject.heroPos = Point2(random.randint(20,600),random.randint(10,450))
            self.mazeObject.goalPos = Point2(random.randint(20,600),random.randint(10,450))
            obj.position = self.mazeObject.getGoalPos()
            self.hero.position = self.mazeObject.getHeroPos()
        '''

    def interSquare(self,hPos,oPos):
        """Returns wether or not circle intersect rectangle"""
        rectangleLeft = oPos[0] - self.settings["object_radius"]
        rectangleRight = oPos[0] + self.settings["object_radius"]
        rectangleTop = oPos[1] - self.settings["object_radius"]
        rectangleBottom = oPos[1] + self.settings["object_radius"]
        
        point = np.array([hPos[0]+ 10 ,hPos[1]+10])
        recs = np.array([rectangleLeft,rectangleTop])
        rece = np.array([rectangleRight,rectangleTop])
        recsb = np.array([rectangleLeft,rectangleBottom])
        receb = np.array([rectangleRight,rectangleBottom])
        
        recsh = np.array([rectangleLeft,rectangleTop])
        receh = np.array([rectangleLeft, rectangleBottom])
        recshr = np.array([rectangleRight, rectangleTop])
        recehr = np.array([rectangleRight,rectangleBottom])
        
        closestXT = geo.point_segment_distance(recs, rece,point)
        closestXB = geo.point_segment_distance(recsb, receb,point)
        
        closestYL = geo.point_segment_distance(recsh, receh,point)
        closestYR = geo.point_segment_distance(recshr, recehr,point)

        x = (closestXT < self.settings["object_radius"]) or (closestXB < self.settings["object_radius"]) or (closestYL < self.settings["object_radius"]) or (closestYR < self.settings["object_radius"])
        return x

    def resolve_collisions(self,fps,actionEvery):
        """If hero touches, hero restarts and reward is updated."""
        collision_distance = 2 * self.settings["object_radius"]
        collision_distance2 = collision_distance ** 2
        to_remove = []
        if(self.time >= self.minTime * self.settings["Timeout"]):
            obj = GameObject(Point2(200,200), 0.0, 0.0,  "enemy", self.settings)
            to_remove.append(obj)
            self.updateSuccess(obj)
            self.time = 0
            self.runs += 1
            self.object_reward += self.settings["object_reward"]["enemy"]
            print("timeout")
        for obj in self.objects:
            if(obj.obj_type == "square"):
                if(self.interSquare(self.hero.position,obj.position)):
                    to_remove.append(obj)
                    self.updateSuccess(obj)
                    self.runs += 1
            elif self.squared_distance(self.hero.position, obj.position) < collision_distance2:
                to_remove.append(obj)
                self.updateSuccess(obj)
                self.runs += 1
        for obj in to_remove:
            if(obj.obj_type == "square"):
                self.objects_eaten["enemy"] += 1
            else:
                self.objects_eaten[obj.obj_type] += 1
            self.object_reward += self.settings["object_reward"][obj.obj_type]
            self.hero.position = self.mazeObject.getHeroPos()
            self.hero.speed = Vector2(*self.settings["hero_initial_speed"])
            self.hero.acceleration = Vector2(*self.settings["hero_initial_accel"])
            self.counter = 0
            self.nextMaze(obj)

            

    def inside_walls(self, point):
        """Check if the point is inside the walls"""
        EPS = 1e-4
        return (EPS <= point[0] < self.size[0] - EPS and
                EPS <= point[1] < self.size[1] - EPS)

    def squareDistance(self,square,line):
        x1 = line.p1[0]
        y1 = line.p1[1]
        x2 = line.p2[0]
        y2 = line.p2[1]
        minX = square.position[0] - self.settings["object_radius"]
        maxX = square.position[0] + self.settings["object_radius"]
        minY = square.position[1] - self.settings["object_radius"]
        maxY = square.position[1] + self.settings["object_radius"]
        if ((x1 <= minX and x2 <= minX) or (y1 <= minY and y2 <= minY) or (x1 >= maxX and x2 >= maxX) or (y1 >= maxY and y2 >= maxY)):
            return False

        m = (y2 - y1) / (x2 - x1)
        y = m * (minX - x1) + y1
        if (y > minY and y < maxY):
            return True
        y = m * (maxX - x1) + y1
        if (y > minY and y < maxY):
            return True
        x = (minY - y1) / m + x1
        if (x > minX and x < maxX):
            return True
        x = (maxY - y1) / m + x1
        if (x > minX and x < maxX):
            return True

    def observe(self):
        print(self.successRate)
        #print(self.timeoutArray)
        #print("minTime " + str(self.minTime))
        #print("dt total" + str(self.time))
        """Return observation vector. For all the observation directions it returns representation
        of the closest object to the hero - might be nothing, another object or a wall.
        Representation of observation for all the directions will be concatenated.
        """
        if self.settings["mod_observation"]:
            return self.mod_observation()
        if self.settings["Rotation"]:
            return self.rotation_observation()
        else:
            return self.old_observation()
        

    def rotation_observation(self):
        #use modified observation method  / vector  
        num_obj_types = len(self.settings["objects"])
        observable_distance = self.settings["observation_line_length"]

        relevant_objects = [obj for obj in self.objects
                            if obj.position.distance(self.hero.position) < observable_distance and obj.obj_type !="friend"]
        # objects sorted from closest to furthest
        relevant_objects.sort(key=lambda x: x.position.distance(self.hero.position))

        observation_size = 10
        observation = np.ones(observation_size)
        for i, observation_line in enumerate(self.observation_lines):
            # shift to hero position
            observation_line = LineSegment2(self.hero.position + Vector2(*observation_line.p1),
                                            self.hero.position + Vector2(*observation_line.p2))
            observed_object = None
            
            for obj in relevant_objects:
                if observation_line.distance(obj.position) < self.settings["object_radius"]:
                    observed_object = obj
                    break
            object_type_id = None
            proximity = observable_distance
            if observed_object is not None: # object seen
                object_type_id = self.settings["objects"].index(observed_object.obj_type)
                speed_x, speed_y = tuple(observed_object.speed)
                if(observed_object.obj_type != "square"):
                    intersection_segment = obj.as_circle().intersect(observation_line)
                    #assert intersection_segment is not None
                    try:
                        proximity = min(intersection_segment.p1.distance(self.hero.position),
                                    intersection_segment.p2.distance(self.hero.position))
                    except AttributeError:
                        proximity = observable_distance
                else:
                    try:
                        proximity = self.squareDistance(observed_object,observation_line)
                    except(ZeroDivisionError):
                        proximity = 0
            

            observation[i] = proximity / observable_distance
        
        #add hero velocity to the  observation vector
        observation[observation_size-4]     = self.hero.speed[0] 
        observation[observation_size-3] = self.hero.speed[1]
        
        # add heading to the observation vector       
        observation[observation_size-2] = self.mazeObject.getGoalPos()[0]-self.hero.position[0]
        observation[observation_size-1] = self.mazeObject.getGoalPos()[1]-self.hero.position[1]
        return observation

    def mod_observation(self):
        #use modified observation method  / vector  
        num_obj_types = len(self.settings["objects"])
        observable_distance = self.settings["observation_line_length"]

        relevant_objects = [obj for obj in self.objects
                            if obj.position.distance(self.hero.position) < observable_distance and obj.obj_type !="friend"]
        # objects sorted from closest to furthest
        relevant_objects.sort(key=lambda x: x.position.distance(self.hero.position))

        observation = np.ones(self.observation_size)
        #observation_offset = 0
        for i, observation_line in enumerate(self.observation_lines):
            # shift to hero position
            observation_line = LineSegment2(self.hero.position + Vector2(*observation_line.p1),
                                            self.hero.position + Vector2(*observation_line.p2))
            observed_object = None
            
            for obj in relevant_objects:
                if observation_line.distance(obj.position) < self.settings["object_radius"]:
                    observed_object = obj
                    break
            object_type_id = None
            proximity = observable_distance
            if observed_object is not None: # object seen
                object_type_id = self.settings["objects"].index(observed_object.obj_type)
                speed_x, speed_y = tuple(observed_object.speed)
                if(observed_object.obj_type != "square"):
                    intersection_segment = obj.as_circle().intersect(observation_line)
                    #assert intersection_segment is not None
                    try:
                        proximity = min(intersection_segment.p1.distance(self.hero.position),
                                    intersection_segment.p2.distance(self.hero.position))
                    except AttributeError:
                        proximity = observable_distance
                else:
                    try:
                        proximity = self.squareDistance(observed_object,observation_line)
                    except(ZeroDivisionError):
                        proximity = 0
            

            observation[i] = proximity / observable_distance
        
        #add hero velocity to the  observation vector
        observation[self.observation_size-4]     = self.hero.speed[0] 
        observation[self.observation_size-3] = self.hero.speed[1]
        
        # add heading to the observation vector       
        observation[self.observation_size-2] = self.mazeObject.getGoalPos()[0]-self.hero.position[0]
        observation[self.observation_size-1] = self.mazeObject.getGoalPos()[1]-self.hero.position[1]
        return observation

    def old_observation(self):
        num_obj_types = len(self.settings["objects"]) + 1 # and wall
        max_speed_x, max_speed_y = self.settings["maximum_speed"]

        observable_distance = self.settings["observation_line_length"]

        relevant_objects = [obj for obj in self.objects
                            if obj.position.distance(self.hero.position) < observable_distance]
        # objects sorted from closest to furthest
        relevant_objects.sort(key=lambda x: x.position.distance(self.hero.position))

        observation        = np.zeros(self.observation_size)
        observation_offset = 0
        for i, observation_line in enumerate(self.observation_lines):
            # shift to hero position
            observation_line = LineSegment2(self.hero.position + Vector2(*observation_line.p1),
                                            self.hero.position + Vector2(*observation_line.p2))

            observed_object = None
            # if end of observation line is outside of walls, we see the wall.
            if not self.inside_walls(observation_line.p2):
                observed_object = "**wall**"
            for obj in relevant_objects:
                if observation_line.distance(obj.position) < self.settings["object_radius"]:
                    observed_object = obj
                    break
            object_type_id = None
            speed_x, speed_y = 0, 0
            proximity = 0
            if observed_object == "**wall**": # wall seen
                object_type_id = num_obj_types - 1
                # a wall has fairly low speed...
                speed_x, speed_y = 0, 0
                # best candidate is intersection between
                # observation_line and a wall, that's
                # closest to the hero
                best_candidate = None
                for wall in self.walls:
                    candidate = observation_line.intersect(wall)
                    if candidate is not None:
                        if (best_candidate is None or
                                best_candidate.distance(self.hero.position) >
                                candidate.distance(self.hero.position)):
                            best_candidate = candidate
                if best_candidate is None:
                    # assume it is due to rounding errors
                    # and wall is barely touching observation line
                    proximity = observable_distance
                else:
                    proximity = best_candidate.distance(self.hero.position)
            elif observed_object is not None: # agent seen
                object_type_id = self.settings["objects"].index(observed_object.obj_type)
                speed_x, speed_y = tuple(observed_object.speed)
                intersection_segment = obj.as_circle().intersect(observation_line)
                assert intersection_segment is not None
                try:
                    proximity = min(intersection_segment.p1.distance(self.hero.position),
                                    intersection_segment.p2.distance(self.hero.position))
                except AttributeError:
                    proximity = observable_distance
            for object_type_idx_loop in range(num_obj_types):
                observation[observation_offset + object_type_idx_loop] = 1.0
            if object_type_id is not None:
                observation[observation_offset + object_type_id] = proximity / observable_distance
            observation[observation_offset + num_obj_types] =     speed_x   / max_speed_x
            observation[observation_offset + num_obj_types + 1] = speed_y   / max_speed_y
            assert num_obj_types + 2 == self.eye_observation_size
            observation_offset += self.eye_observation_size

        observation[observation_offset]     = self.hero.speed[0] / max_speed_x
        observation[observation_offset + 1] = self.hero.speed[1] / max_speed_y
        observation_offset += 2
        
        # add normalized locaiton of the hero in environment        
        observation[observation_offset]     = self.hero.position[0] / 350.0 - 1.0
        observation[observation_offset + 1] = self.hero.position[1] / 250.0 - 1.0
        
        assert observation_offset + 2 == self.observation_size

        return observation
    
    def distance_to_walls(self):
        """Returns distance of a hero to walls"""
        res = float('inf')
        for wall in self.walls:
            res = min(res, self.hero.position.distance(wall))
        return res - self.settings["object_radius"]
    
    
    def distance_to_goal(self):
        return self.hero.position.distance(self.mazeObject.getGoalPos())
    
                                           
    def collect_reward(self):
        """Return accumulated object eating score + current distance to goal score"""
        togoal = 1000 -self.distance_to_goal()
        total_reward = self.object_reward + togoal
        self.runReward.append(total_reward)
        reward = int((total_reward - self.currReward))
        if self.object_reward != 0:
            self.currReward = 0
        else:
            self.currReward = total_reward

        self.object_reward = 0
        self.collected_rewards.append(reward)
        return reward

    def plot_reward(self, smoothing = 30):
        """Plot evolution of reward over time."""
        plottable = self.collected_rewards[:]
        while len(plottable) > 1000:
            for i in range(0, len(plottable) - 1, 2):
                plottable[i//2] = (plottable[i] + plottable[i+1]) / 2
            plottable = plottable[:(len(plottable) // 2)]
        x = []
        for  i in range(smoothing, len(plottable)):
            chunk = plottable[i-smoothing:i]
            x.append(sum(chunk) / len(chunk))
        plt.plot(list(range(len(x))), x)
        plt.xlabel('Frames')
        plt.ylabel('Reward')
        plt.title('Average Reward Overall')
        #plt.savefig(self.pp, format='pdf')
        #self.pp.close()

   
    def plot_run_reward(self, smoothing = 30):
        """Plot evolution of reward over time."""
        plottable = self.runReward[:]
        while len(plottable) > 1000:
            for i in range(0, len(plottable) - 1, 2):
                plottable[i//2] = (plottable[i] + plottable[i+1]) / 2
            plottable = plottable[:(len(plottable) // 2)]
        x = []
        for  i in range(smoothing, len(plottable)):
            chunk = plottable[i-smoothing:i]
            x.append(sum(chunk) / len(chunk))
        plt.plot(list(range(len(x))), x)
        plt.xlabel('Frames')
        plt.ylabel('Reward')
        plt.title(str(self.mazeIterator -2))
        plt.savefig(self.pp, format='pdf')

    def plot_timing(self, smoothing = 30):
        """Plot evolution of reward over time."""
        plottable = self.timeoutArray[:]
        while len(plottable) > 1000:
            for i in range(0, len(plottable) - 1, 2):
                plottable[i//2] = (plottable[i] + plottable[i+1]) / 2
            plottable = plottable[:(len(plottable) // 2)]
        x = []
        for  i in range(smoothing, len(plottable)):
            chunk = plottable[i-smoothing:i]
            x.append(sum(chunk) / len(chunk))
        plt.plot(self.timeoutArray,"ro")
        plt.axis([0,len(self.timeoutArray),0,10])
        plt.xlabel('Runs')
        plt.ylabel('Relative time to Min')
        plt.title('Timing Graph')
        plt.savefig(self.pp, format='pdf')
        self.pp.close()

    def generate_observation_lines(self):
        """Generate observation segments in settings["num_observation_lines"] directions"""
        result = []
        start = Point2(0.0, 0.0)
        end   = Point2(self.settings["observation_line_length"],
                       self.settings["observation_line_length"])
        num = 2*np.pi
        lines = self.settings["num_observation_lines"]

        if self.settings["Rotation"]:
            num = np.pi
            lines = 10
        
        for angle in np.linspace(0, num, lines, endpoint=False):
            rotation = Point2(math.cos(angle+ self.heading), math.sin(angle))
            current_start = Point2(start[0] * rotation[0], start[1] * rotation[1])
            current_end   = Point2(end[0]   * rotation[0], end[1]   * rotation[1])
            result.append(LineSegment2(current_start, current_end))
        return result

    def _repr_html_(self):
        return self.to_html()

    def to_html(self, stats=[]):
        """Return svg representation of the simulator"""

        stats = stats[:]
        recent_reward = self.collected_rewards[-100:] + [0]
        objects_eaten_str = ', '.join(["%s: %s" % (o,c) for o,c in self.objects_eaten.items()])
        stats.extend([
            "Reward       = %1f" % (self.collected_rewards[-1],),#sum(recent_reward)/len(recent_reward),),
            "Objects Eaten => %s" % (objects_eaten_str,),
            "Percent Success (last 10) => %1f " % self.successRate,
            "Runs => %d" % self.runs,
            "Maze # : %d" % (self.mazeIterator -1)
            #"seconds: %.1f" % (self.counter/400)
        ])

        scene = svg.Scene((self.size[0] + 20, self.size[1] + 20 + 20 * len(stats)))
        scene.add(svg.Rectangle((10, 10), self.size))


        for line in self.observation_lines:
            scene.add(svg.Line(line.p1 + self.hero.position + Point2(10,10),
                               line.p2 + self.hero.position + Point2(10,10)))

        for obj in self.objects + [self.hero]:
            scene.add(obj.draw())

        offset = self.size[1] + 15
        for txt in stats:
            scene.add(svg.Text((10, offset + 20), txt, 15))
            offset += 20

        return scene

