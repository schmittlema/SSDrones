import tf_rl.utils.svg as svg

def replay(logfile):
    log = open("../RunData/" + logfile,"r")
    string = log.read()
    #print string

