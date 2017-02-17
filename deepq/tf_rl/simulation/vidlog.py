import math
from collections import defaultdict
import numpy as np
from graphics import *
from euclid import Point2,Vector2,LineSegment2

#This is meant to replay a experiment from the log file

class GameObject(object):
    def __init__(self, position, radius, obj_type):
        """Esentially represents circles of different kinds, which have
        position and speed."""
        self.radius = radius 
        self.obj_type = obj_type
        self.position = position

def generate_observation_lines(num_lines,line_length,rotation_in,heading):
    """Generate observation segments in settings["num_observation_lines"] directions"""
    result = []
    start = Point2(0.0, 0.0)
    end   = Point2(line_length,
                   line_length)
    num = 2*np.pi
    nums = 0
    h = 0
    lines = num_lines
    obstlines = np.linspace(nums, num, lines, endpoint=False)

    if rotation_in:
        nums = 0 
        num = np.pi
        lines = 7
        h = ((heading * np.pi)/180) + np.pi/2
        obstlines = [0,np.pi/6,np.pi/3,np.pi/2,2*np.pi/3,5*np.pi/6,np.pi]
    
    for angle in obstlines:
        rotation = Point2(math.cos(angle+ h), math.sin(angle + h))
        current_start = Point2(start[0] * rotation[0], start[1] * rotation[1])
        current_end   = Point2(end[0]   * rotation[0], end[1]   * rotation[1])
        result.append((current_start,current_end))
    return result

#Variables for simulation. Will take from log file
size = (700,650)
hero_pos = Point2(50,50)
heading = 0
hero = GameObject(hero_pos,10.0,"hero")
obj1 = GameObject(Point2(100,100),10.0,"enemy")
objects = [obj1]
line_length = 120
rotation = False
num_lines = 32
observation_lines = generate_observation_lines(num_lines,line_length,rotation,heading);
objects_eaten = defaultdict(lambda: 0)
successRate = 0
runs = 0
mazeIterator = 0 
collected_rewards = [0]
        
def draw(obj,win):
    """Return svg object for this item."""
    if obj.obj_type == "enemy":
        color = "red"
    if obj.obj_type == "friend":
        color = 'green'
    if obj.obj_type == "hero":
        color = 'yellow'
    if(obj.obj_type == "square"):
        color = "red"
        print("Currently Unsupported")
    else:
        circle = Circle(obj.position,obj.radius)
        circle.setFill(color)
        circle.draw(win)

def draw_line(start,end,win):
    #have to do this to prevent a clone error
    s = Point(start[0],start[1])
    e = Point(end[0],end[1])
    l = Line(s,e)
    l.setWidth(1)
    l.draw(win)

def makeRect(corner, width, height):
    '''Return a new Rectangle given one corner Point and the dimensions.'''
    corner2 = corner.clone()
    corner2.move(width, height)
    return Rectangle(corner, corner2)

def convert_to_cartesian(heading):
    h = (heading * math.pi)/(200-20)
    return [math.cos(h),math.sin(h)]

def addText(txt,win,offset):
    info = Text(Point(350,20 + offset),txt)
    info.setSize(12)
    info.draw(win)

def to_html(successRate,runs,mazeIterator,collected_rewards,objects,hero,observation_lines,rotation,heading,objects_eaten,stats=[]):
    """Return draw the simulator"""
    stats = stats[:]
    #recent_reward = self.collected_rewards[-100:] + [0]
    objects_eaten_str = ', '.join(["%s: %s" % (o,c) for o,c in objects_eaten.items()])
    stats.extend([
        "Reward       = %1f" % (collected_rewards[-1],),#sum(recent_reward)/len(recent_reward),),
        "Objects Eaten => %s" % (objects_eaten_str,),
        "Percent Success (last 10) => %1f " % successRate,
        "Runs => %d" % runs,
        "Maze # : %d" % (mazeIterator -1)
        #"seconds: %.1f" % (self.counter/400)
    ])

    win = GraphWin('Replay',size[0],size[1])
    win.setBackground('white')
    rect = makeRect(Point(10,10),680,500)
    rect.draw(win)


    for line in observation_lines:
        line = (line[0]+hero_pos,line[1] + hero_pos)
        draw_line(line[0],line[1],win)

    if (rotation):
        headingend = convert_to_cartesian(heading) 
        headingend[0] = headingend[0] * 50*3
        headingend[1] = headingend[1] * 50*3
        draw_line(hero_pos,hero_pos-Point2(headingend[0],headingend[1]))

    for obj in objects + [hero]:
        draw(obj,win)

    offset = 500 + 15
    for txt in stats:
        addText(txt,win,offset)
        offset += 20

    win.getMouse()
    win.close()


to_html(successRate,runs,mazeIterator,collected_rewards,objects,hero,observation_lines,rotation,heading,objects_eaten,[])

