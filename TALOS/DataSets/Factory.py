# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        DATASET FACTORY
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
from .CrowdDetection import DroneCrowd60
from .LITE import ImageDataSetLITE


#==================================================================================================
class DataSetFactory(object):
    Models={   
               "LITE20"   : ImageDataSetLITE
              ,"LITE30"   : ImageDataSetLITE
              ,"LITE50"   : ImageDataSetLITE
              ,"LITE100"  : ImageDataSetLITE
              ,"LITE200"  : ImageDataSetLITE
           }
    #------------------------------------------------------------------------------------
    @classmethod
    def AddDataSet(cls, p_sDataSetName, p_cDataSetClass):
        DataSetFactory.Models[p_sDataSetName] = p_cDataSetClass
    #------------------------------------------------------------------------------------
    @classmethod
    def Create(cls, p_sDataSetName, p_sDataSetFolder):
        oDataSet=DataSetFactory.Models[p_sDataSetName](p_sDataSetFolder)
        return oDataSet   
    #------------------------------------------------------------------------------------
    @classmethod
    def CreateVariation(cls, p_sDataSetName, p_nVariationNumber, p_sDataSetFolder):
        oDataSet=DataSetFactory.Models[p_sDataSetName](p_sDataSetFolder, p_nVariationNumber)
        return oDataSet          
    #------------------------------------------------------------------------------------
#==================================================================================================



