import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import Mazes
import os

from collections import defaultdict
from euclid import Circle, Point2, Vector2, LineSegment2

import tf_rl.simulate
import tf_rl.utils.svg as svg

class GameObject(object):
    def __init__(self, position, speed, acceleration, obj_type, settings):
        """Esentially represents circles of different kinds, which have
        position and speed."""
        self.settings = settings
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

    def move(self, dt):
        """Move as if dt seconds passed"""
        self.speed+= dt*self.acceleration
        self.position+= dt*self.speed
        self.position = Point2(*self.position)

    def step(self, dt):
        """Move and bounce of walls."""
        self.wall_collisions()
        self.move(dt)

    def as_circle(self):
        return Circle(self.position, float(self.radius))

    def draw(self):
        """Return svg object for this item."""
        color = self.settings["colors"][self.obj_type]
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
        self.mazeIterator = 0
        self.mazeObject = Mazes.Maze()
        self.makeMaze()
        self.startTime = time.strftime("%d:%m:%Y:%H")
        
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

        if not self.settings["hero_bounces_off_walls"]:
            self.hero.bounciness = 0.0

        self.objects = []
        for obj_type, number in settings["num_objects"].items():
            if(obj_type == "enemy"):
                number = len(self.maze)
            for _ in range(number):
                self.spawn_object(obj_type)

        self.observation_lines = self.generate_observation_lines()

        self.object_reward = 0
        self.collected_rewards = []

        # every observation_line sees the nearest friend or enemy
        #edit: agent is no longer able to see walls, nearest friend, or the speed of the nearest object bc these things have been taken out
        self.eye_observation_size = len(self.settings["objects"])-1
        # additionally there are two numbers representing the heading vector  and the objects position.
        self.observation_size = self.eye_observation_size * len(self.observation_lines) + 2 + 2
        #directions of movement  
        self.directions = [Vector2(*d) for d in [[1,0], [0,1], [-1,0],[0,-1],[0.0,0.0]]]
        self.num_actions = len(self.directions)

        self.objects_eaten = defaultdict(lambda: 0)
    
    def makeMaze(self):
        self.maze = self.mazeObject.getMaze(self.mazeIterator)            
        self.hero.position = self.mazeObject.getHeroPos()
        self.mazeIterator += 1
        
    def perform_action(self, action_id):
        """Change speed to one of hero vectors"""
        assert 0 <= action_id < self.num_actions #check to see if valid action
        #self.hero.speed *= 0.5
        #self.hero.speed += self.directions[action_id] * self.settings["delta_v"]
        self.hero.acceleration= self.directions[action_id]*self.settings["accel"]
        

    def spawn_object(self, obj_type):
        """Spawn object of a given type and add it to the objects array"""
        radius = self.settings["object_radius"]
        if(obj_type == 'friend'):
            position = self.mazeObject.getGoalPos()
        else:
            position = self.maze[self.mazeindex]
            self.mazeindex += 1 

        max_speed = np.array(self.settings["maximum_speed"])
        speed = Vector2(0, 0)
        acceleration = Vector2(0,0)
        self.objects.append(GameObject(position, speed, acceleration,  obj_type, self.settings))

    def step(self, dt):
        """Simulate all the objects for a given ammount of time.

        Also resolve collisions with the hero"""
           
        for obj in self.objects + [self.hero] :
            obj.step(dt)
        self.resolve_collisions()
        
    def runComplete(self):
        self.timeOut = time.time() - self.timeStart
        if(len(self.timeoutArray) == 10):
            self.timeoutArray = self.timeoutArray[1:]
        self.timeoutArray.append(self.timeOut)
        self.timeStart = time.time() 
        
    def runFail(self):
        self.timeStart = time.time()
            

    def squared_distance(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def updateSuccess(self,ob):
        if(len(self.successArray) == 10):
                self.successArray = self.successArray[1:]
        if ob.obj_type == "friend":
            self.successArray.append(1)
            self.runComplete()
        if ob.obj_type == "enemy":
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
        time = sum(self.timeoutArray,0.0)/float(len(self.timeoutArray))
        self.averageTimeout.append(time)
        wtf.write("AverageTimout: " + str(time) + "\n"+"\n")
        wtf.close()

    def saveTotals(self):
        dir = "../RunData/"
        name =dir + self.brainName + ".txt"
        wtf = open(name,"a")
        wtf.write("Overall:"+ "\n")
        wtf.write("Runs: " + str(sum(self.averageRuns,0.0)/float(len(self.averageRuns))) + "\n")
        success = sum(self.averageSuccessRate,0.0)/float(len(self.averageSuccessRate))
        wtf.write("SuccessRate: " + str(success) + "\n")
        wtf.write("CrashRate: " + str(1-success) + "\n")
        wtf.write("AverageTimout: " + str(sum(self.averageTimeout,0.0)/float(len(self.averageTimeout))) + "\n"+"\n")
        wtf.close()    

    def nextMaze(self):
        """see if the agent has met the criteria for advancement, advance if it has"""
        if(self.runs >= 100 and self.successRate >= self.settings["minimum_success_rate"]):
            self.saveData()
            self.timeoutArray = []
            self.runs = 0
            self.objects_eaten["friend"] = 0
            self.objects_eaten["enemy"] = 0
            self.successRate = 0.0
            self.successArray = []
            self.makeMaze()
            self.mazeindex = 0
            for obj_type, number in self.settings["num_objects"].items():
                if(obj_type == "enemy"):
                    number = len(self.maze)
                for _ in range(number):
                    self.spawn_object(obj_type)

    
    def resolve_collisions(self):
        """If hero touches, hero restarts and reward is updated."""
        collision_distance = 2 * self.settings["object_radius"]
        collision_distance2 = collision_distance ** 2
        to_remove = []
        for obj in self.objects:
            if self.squared_distance(self.hero.position, obj.position) < collision_distance2:
                to_remove.append(obj)
                self.updateSuccess(obj)
                self.runs += 1
        for obj in to_remove:
            self.objects_eaten[obj.obj_type] += 1
            self.object_reward += self.settings["object_reward"][obj.obj_type]
            self.hero.position = self.mazeObject.getHeroPos()
            #reset the speed and acceleration of the hero, not sure if this is in the right place 
            self.speed=Vector2(self.settings["hero_initial_speed"])
            self.acceleration=Vector2(self.settings["hero_initial_accel"])
            self.nextMaze()

            

    def inside_walls(self, point):
        """Check if the point is inside the walls"""
        EPS = 1e-4
        return (EPS <= point[0] < self.size[0] - EPS and
                EPS <= point[1] < self.size[1] - EPS)

    def observe(self):
        """Return observation vector. For all the observation directions it returns representation
        of the closest object to the hero - might be nothing, another object or a wall.
        Representation of observation for all the directions will be concatenated.
        """
        num_obj_types = len(self.settings["objects"])
        observable_distance = self.settings["observation_line_length"]

        relevant_objects = [obj for obj in self.objects
                            if obj.position.distance(self.hero.position) < observable_distance]
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
            proximity = 0
            if observed_object is not None: # object seen
                object_type_id = self.settings["objects"].index(observed_object.obj_type)
                intersection_segment = obj.as_circle().intersect(observation_line)
                #assert intersection_segment is not None
                try:
                    proximity = min(intersection_segment.p1.distance(self.hero.position),
                                    intersection_segment.p2.distance(self.hero.position))
                except AttributeError:
                    proximity = observable_distance
                    
            observation[i] = proximity / observable_distance
            #assert num_obj_types + 2 == self.eye_observation_size
            #observation_offset += self.eye_observation_size
        
        
        #add hero velocity to the  observation vector
        observation[self.observation_size-4]     = self.hero.speed[0] 
        observation[self.observation_size-3] = self.hero.speed[1]
        
        # add heading to the observation vector       
        observation[self.observation_size-2] = self.mazeObject.getGoalPos()[0]-self.hero.position[0]
        observation[self.observation_size-1] = self.mazeObject.getGoalPos()[1]-self.hero.position[1]
        
        #assert observation_offset + 2 == self.observation_size
        print(observation)
        return observation
    
    '''def get_heading(self):
        """calculate heading vector"""
        dx=self.mazeObject.getGoalPos()[0]-self.hero.position[0]
        dy=self.mazeObject.getGoalPos()[1]-self.hero.position[1]
        #m=math.sqrt(dx**2+dy**2)
        return (dx , dy)'''
        
    
    def distance_to_walls(self):
        """Returns distance of a hero to walls"""
        res = float('inf')
        for wall in self.walls:
            res = min(res, self.hero.position.distance(wall))
        return res - self.settings["object_radius"]
    
    
    def distance_to_goal(self):
        return -self.hero.position.distance(self.mazeObject.getGoalPos())
    
                                           
    def collect_reward(self):
        """Return accumulated object eating score + current distance to walls score"""
        #wall_reward =  self.settings["wall_distance_penalty"] * \
        #               np.exp(-self.distance_to_walls() / self.settings["tolerable_distance_to_wall"])
        #assert wall_reward < 1e-3, "You are rewarding hero for being close to the wall!"
        togoal = self.distance_to_goal()/1000
        total_reward = self.object_reward + togoal
        self.collected_rewards.append(total_reward)
        self.object_reward = 0
        return total_reward

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

    def generate_observation_lines(self):
        """Generate observation segments in settings["num_observation_lines"] directions"""
        result = []
        start = Point2(0.0, 0.0)
        end   = Point2(self.settings["observation_line_length"],
                       self.settings["observation_line_length"])
        for angle in np.linspace(0, 2*np.pi, self.settings["num_observation_lines"], endpoint=False):
            rotation = Point2(math.cos(angle), math.sin(angle))
            current_start = Point2(start[0] * rotation[0], start[1] * rotation[1])
            current_end   = Point2(end[0]   * rotation[0], end[1]   * rotation[1])
            result.append( LineSegment2(current_start, current_end))
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
        ])

        scene = svg.Scene((self.size[0] + 20, self.size[1] + 20 + 20 * len(stats)))
        scene.add(svg.Rectangle((10, 10), self.size))


        for line in self.observation_lines:
            scene.add(svg.Line(line.p1 + self.hero.position + Point2(10,10),
                               line.p2 + self.hero.position + Point2(10,10)))

        for obj in self.objects + [self.hero] :
            scene.add(obj.draw())

        offset = self.size[1] + 15
        for txt in stats:
            scene.add(svg.Text((10, offset + 20), txt, 15))
            offset += 20

        return scene

