# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        NEURAL NETWORK MODEL FACTORY
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
from .AlexNetCNN import AlexNet
from .ZFNetCNN import ZFNet
from .VGGCNN import VGG16, VGG19
from .RetinaCNN import ARNN4
from TALOS.NeuralNetworks import NNImageClasifierSettings



#==================================================================================================
class CNNModelFactory(object):
    Models = {    
                 "AlexNet"  : AlexNet
                ,"ZFNet"    : ZFNet
                ,"VGG16"    : VGG16
                ,"VGG19"    : VGG19
                ,"RetinaCNN": ARNN4
             }
            
    #------------------------------------------------------------------------------------
    @classmethod
    def AddModel(cls, p_sModelName, p_cModelClass):
        CNNModelFactory.Models[p_sModelName] = p_cModelClass
    #------------------------------------------------------------------------------------
    @classmethod
    def Create(cls, p_sModelName, oDataSet, oExperiment):
        oNet=CNNModelFactory.Models[p_sModelName](NNImageClasifierSettings(oDataSet.ClassCount, [227,227], oExperiment.BatchSize), oExperiment)
        return oNet        
    #------------------------------------------------------------------------------------
#==================================================================================================



