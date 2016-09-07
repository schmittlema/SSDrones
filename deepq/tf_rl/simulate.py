import math
import sys
import time
from IPython.display import clear_output, display, HTML
from itertools import count
from os.path import join, exists
from os import makedirs
import  tf_rl.utils.getch as g 
import threading
import tensorflow as tf

running = True

def simulate(simulation,
             controller= None,
             fps=60,
             visualize_every=1,
             action_every=1,
             simulation_resolution=None,
             wait=False,
             disable_training=False,
             save_path=None):
    """Start the simulation. Performs three tasks

        - visualizes simulation in iPython notebook
        - advances simulator state
        - reports state to controller and chooses actions
          to be performed.

    Parameters
    -------
    simulation: tr_lr.simulation
        simulation that will be simulated
    controller: tr_lr.controller
        controller used
    fps: int
        frames per seconds
    visualize_every: int
        visualize every `visualize_every`-th frame.
    action_every: int
        take action every `action_every`-th frame
    simulation_resolution: float
        simulate at most 'simulation_resolution' seconds at a time.
        If None, the it is set to 1/FPS (default).
    wait: boolean
        whether to intentionally slow down the simulation
        to appear real time.
    disable_training: bool
        if true training_step is never called.
    save_path: str
        save svg visualization (only tl_rl.utils.svg
        supported for the moment)
    """
    global running
    # prepare path to save simulation images
    if save_path is not None:
        if not exists(save_path):
            makedirs(save_path)
    last_image = 0

    # calculate simulation times
    chunks_per_frame = 1
    chunk_length_s   = 1.0 / fps

    if simulation_resolution is not None:
        frame_length_s = 1.0 / fps
        chunks_per_frame = int(math.ceil(frame_length_s / simulation_resolution))
        chunks_per_frame = max(chunks_per_frame, 1)
        chunk_length_s = frame_length_s / chunks_per_frame

    # state transition bookkeeping
    last_observation = None
    last_action      = None


    thread = threading.Thread(target=input,args=(simulation,))
    thread.deamon = True
    thread.start()

    simulation_started_time = time.time()
    count = 0
    frame_no = 0
    running = True
    timeStart = simulation_started_time
    while running:
        for _ in range(chunks_per_frame):
            simulation.step(chunk_length_s,fps,action_every)

        if frame_no % action_every == 0:
            new_observation = simulation.observe()
            reward          = simulation.collect_reward()
            # store last transition
            if last_observation is not None:
                controller.store(last_observation, last_action, reward, new_observation)

            # act
            new_action = controller.action(new_observation)
            simulation.perform_action(new_action)

            #train
            start = time.time()
            if not disable_training:
                controller.training_step()
                
            elapsed_time = time.time() - start
            #print elapsed_time
            simulation.learntime += elapsed_time
            start = 0

            
            # update current state as last state.
            last_action = new_action
            last_observation = new_observation
            
                
        # adding 1 to make it less likely to happen at the same time as
        # action taking.
        if (frame_no + 1) % visualize_every == 0:
            fps_estimate = frame_no / (time.time() - simulation_started_time)
            #print(fps_estimate)
            #sumfps = tf.scalar_summary("fps",fps_estimate)
            #preward = tf.scalar_summary("Reward",simulation.currReward)
            #kcontroller.summary_writer.add_summary(controller.s.run(preward),controller.iteration)
            #controller.summary_writer.add_summary(controller.s.run(sumfps),controller.iteration)

            clear_output(wait=True)
            svg_html = simulation.to_html(["fps = %.1f" % (fps_estimate,)])
            if save_path is not None:
                img_path = join(save_path, "screen.svg")
                if(simulation.display):
                    with open(img_path, "w") as f:
                        svg_html.write_svg(f,img_path)
        
       
        

        time_should_have_passed = frame_no / fps
        time_passed = (time.time() - simulation_started_time)
        if wait and (time_should_have_passed > time_passed):
            time.sleep(time_should_have_passed - time_passed)
            
        frame_no +=1

    raise KeyboardInterrupt

def input(simulation):
    getch = g._Getch()   
    global running
    print("To quit press q then hold ctl + c")
    while True:
        c = getch.impl()
        if c == "q":
            running = False
            sys.exit()
        if c == "d":
            if(simulation.display):
                simulation.display = False
                print("no longer displaying...")
            else:
                simulation.display = True
                print("displaying...")



