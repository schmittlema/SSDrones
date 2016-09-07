from __future__ import print_function
# coding: utf-8

# In[1]:

import tensorflow as tf

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib auto')

# In[2]:

import numpy as np
import tempfile
import time
from threading import Thread
from tf_rl.controller import DiscreteDeepQ
from tf_rl.simulation import Main
from tf_rl import simulate
from tf_rl.models import MLP




# In[3]:

LOG_DIR = tempfile.mkdtemp()
print(LOG_DIR)


# In[4]:

current_settings = {
    'objects': [
        'friend',
        'enemy',
        'square',
    ],
    'colors': {
        'hero':   'yellow',
        'friend': 'green',
        'enemy':  'red',
        'square': 'red',
    },
    'object_reward': {
        'friend': 1000,
        'enemy': -1000,
        'square': -100
    },
    'hero_bounces_off_walls': False,
    'add_physics':False,
    'mod_observation': False,
    'world_size': (700,500),
    'hero_initial_position': [600, 440],
    'hero_initial_speed':    [0,   0],
    'hero_initial_accel': [0, 0],
    "maximum_speed":         [100, 100],
    "object_radius": 10.0,
    "num_objects": {
        "friend" : 1,
        "enemy" :  1,
        "square" : 1,
    },
    "num_observation_lines" : 32,
    "observation_line_length": 120.,
    "tolerable_distance_to_wall": 700,
    "wall_distance_penalty":  -0.0,
    "delta_v": 50,
    "speed":0,
    "accel":50,
    "minimum_success_rate": 1.0,
    "Timeout":5 ,
    "Rotation" :True 
}


# In[5]:

notString = False
notRight = True
saver = False
while(notRight):
    try:
        choice = input("(A)Start new Experiment" + "\n"+"(B)Reload from file?" + "\n")
        if(choice == "A" or choice =="a"):
            notString = True
            notRight = False
        if(choice == "B" or choice == "b"):
            notRight = False
            notString = True
            saver = True
        if (not notString):
            print("Please enter A or B")
    except (NameError,SyntaxError):
        print("Please enter as a string")
        
brainName = "null"
while(notString):
    try:
        if(not saver):
            brainName = input("Enter new brain file name: ")
        else:
            brainName = input("Enter brain file name: (.ckpt not included) ")
        notString = False
    except (NameError,SyntaxError):
        print("Please enter as a string")
notString = True
gpu = False
while(notString):
    try:
        g = input("GPU ? ")
        if(g == "y" or g == "Y"):
            gpu = True
            print("Running on GPU")
        else:
            gpu = False
        notString = False
    except (NameError,SyntaxError):
        print("Please enter as a string")


# In[6]:

g = Main(current_settings,brainName)# create the game simulator


# In[ ]:

human_control = False

if human_control:
    # WSAD CONTROL (requires extra setup - check out README)
    current_controller = HumanController({b"w": 3, b"d": 0, b"s": 1,b"a": 2,}) 
else:
    # Tensorflow business - it is always good to reset a graph before creating a new controller.
    tf.python.framework.ops.reset_default_graph()
    if gpu:
        config = tf.ConfigProto(allow_soft_placement=True)
    else:
        config = tf.ConfigProto()
    session = tf.InteractiveSession(config = config)

    # This little guy will let us run tensorboard
    #      tensorboard --logdir [LOG_DIR]
    journalist = tf.train.SummaryWriter(LOG_DIR)

    # Brain maps from observation to Q values for different actions.
    # Here it is a done using a multi layer perceptron with hidden
    # layers
    size = g.observation_size
    if g.settings["Rotation"]:
        size = 10
    brain = MLP([size,], [300,300, g.num_actions], 
                [tf.tanh, tf.tanh,tf.identity])
    
    # The optimizer to use. Here we use RMSProp as recommended
    # by the publication
    optimizer = tf.train.RMSPropOptimizer(learning_rate= 0.001, decay=0.9)

    # DiscreteDeepQ object
    current_controller = DiscreteDeepQ(size, g.num_actions, brain, optimizer, session,brainName,
                                       discount_rate=0.99, exploration_period=5000, max_experience=10000, 
                                       store_every_nth=4, train_every_nth=4,
                                       summary_writer=journalist)

    
    session.run(tf.initialize_all_variables())
    session.run(current_controller.target_network_update)
    # graph was not available when journalist was created  
    journalist.add_graph(session.graph)
    
    if(saver):
        notFound = True
        while(notFound):
            string = "../saved_brains/" + brainName + ".ckpt"
            try:
                current_controller.saver.restore(session,string)
                notFound = False
                current_controller.bN = brainName
                g.brainName = brainName
                journalist.add_graph(session.graph_def)
                print("Running: " + brainName + ".ckpt")
            except(tf.errors.NotFoundError):
                brainName = input("File does not exist try again: ")

    
# In[ ]:

FPS          = 100
ACTION_EVERY = 3
    
fast_mode = True
if fast_mode:
    WAIT, VISUALIZE_EVERY = False, 10
else:
    WAIT, VISUALIZE_EVERY = True, 1

    
try:
    start_time = time.time()
    if(gpu):
        session.run(simulate(simulation=g,
             controller=current_controller,
                 fps=FPS,
                 visualize_every=VISUALIZE_EVERY,
                 action_every=ACTION_EVERY,
                 wait=WAIT,
                 disable_training=False,
                 simulation_resolution=0.001,
                 save_path="/tmp/"))
    else:
        with tf.device("/cpu:0"):
            simulate(simulation=g,
                 controller=current_controller,
                     fps=FPS,
                     visualize_every=VISUALIZE_EVERY,
                     action_every=ACTION_EVERY,
                     wait=WAIT,
                     disable_training=False,
                     simulation_resolution=0.001,
                     save_path="/tmp/")

except (KeyboardInterrupt,IndexError):
    g.plot_timing(smoothing=100)
    g.saveTotals()
    print("Complete")
    runTime = time.time() - start_time
    print("Total RunTime: " + str(runTime))
    print("LearningTime: " + str(g.learntime))
    print("SimulationTime: " + str(runTime - g.learntime))



session.run(current_controller.target_network_update)



current_controller.q_network.input_layer.Ws[0].eval()



current_controller.target_q_network.input_layer.Ws[0].eval()

