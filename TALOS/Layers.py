# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        ADDITIONAL PROCESSING AND NORMALIZATION LAYERS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import tensorflow as tf



#==================================================================================================
class MinMax(object):
    #------------------------------------------------------------------------------------
    @classmethod
    def NormalizationLayer(cls, p_tInput):
        nInputShape=p_tInput.get_shape()
    
        if nInputShape[0].value is None:
            nBatchCount = -1
        else:
            nBatchCount=int(nInputShape[0].value)
    
        if nInputShape[3].value is None:
            nFeatureCount = -1
        else:
            nFeatureCount=int(nInputShape[3].value)
    
        with tf.name_scope("MinMaxNormalization"):
            tMin=tf.reduce_min(p_tInput, reduction_indices=(1,2))
            tMax=tf.reduce_max(p_tInput, reduction_indices=(1,2))
            tMin=tf.reshape(tMin, (nBatchCount,1,1,nFeatureCount))
            tMax=tf.reshape(tMax, (nBatchCount,1,1,nFeatureCount))
            tOut=(p_tInput-tMin)/(tMax-tMin+1e-4)
        return tOut  
    #------------------------------------------------------------------------------------
    @classmethod
    def NormalizationRescaledLayer(cls, p_tInput, p_nNewMin, p_nNewMax):
        nInputShape=p_tInput.get_shape()
    
        if nInputShape[0].value is None:
            nBatchCount = -1
        else:
            nBatchCount=int(nInputShape[0].value)
    
        if nInputShape[3].value is None:
            nFeatureCount = -1
        else:
            nFeatureCount=int(nInputShape[3].value)
    
        with tf.name_scope("MinMaxNormalizationRescaled"):
            tMin=tf.reduce_min(p_tInput, reduction_indices=(1,2))
            tMax=tf.reduce_max(p_tInput, reduction_indices=(1,2))
            tMin=tf.reshape(tMin, (nBatchCount,1,1,nFeatureCount))
            tMax=tf.reshape(tMax, (nBatchCount,1,1,nFeatureCount))
            tMinMax=(p_tInput-tMin)/(tMax-tMin+1e-4)
    
            #            (newMax-newMin)(x - min)
            #    f(x) = ------------------------- + newMin
            #            max - min
            tNewMin=tf.constant(p_nNewMin, tf.float32)
            tNewMax=tf.constant(p_nNewMax, tf.float32 )
            tOut=(tNewMax-tNewMin)*tMinMax + tNewMin
        return tOut
    #------------------------------------------------------------------------------------
#==================================================================================================