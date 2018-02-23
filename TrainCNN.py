# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        CNN LEARNING SESSION WITH EXCEPTION HANDLING 
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
from TALOS.Session import CNNTrainingSession

try:
    oSession = CNNTrainingSession()
    oSession.Run()
    nRunResult = 0
except Exception as e:
    print("EXCEPTION:", e)
    nRunResult = 1
finally:
    exit(nRunResult)
     
        
                