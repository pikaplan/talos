# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        TRAINING QUEUE MAIN
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
from TALOS.Experiments import ExperimentQueueSystem

oSystem = ExperimentQueueSystem()
oSystem.LoopExecution("TrainCNN.py")

