# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        NEURAL NETWORK MODEL PARAMETERS AND WEIGHT INITIALIZATORS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import numpy as np
import tensorflow as tf
import math
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import dtypes















#==================================================================================================
class ParamsItem(object):
    __verboseLevel = 1
    
    #------------------------------------------------------------------------------------
    def __init__(self, p_nIndex):
        #........ |  Instance Attributes | ..............................................
        self.Index = p_nIndex     
        self.VarTensors = [] 
        self.Count = None
        self.MemoryUsed = None
        
        #................................................................................
    #------------------------------------------------------------------------------------
    def __bytesByDataType(self, p_tTensor):
        dtype = p_tTensor.dtype
        
        if (dtype == tf.uint8) or (dtype == np.uint8) or (dtype == dtypes.uint8):
            return 1
        elif (dtype == tf.int32) or (dtype == np.int32) or (dtype == dtypes.int32_ref) or (dtype == dtypes.int32):
            return 4
        if (dtype == tf.float32) or (dtype == np.float32) or (dtype == dtypes.float32_ref) or (dtype == dtypes.float32):
            return 4
        if (dtype == tf.float64) or (dtype == np.float64) or (dtype == dtypes.float64_ref) or (dtype == dtypes.float64):
            return 8
        if (dtype == tf.int16) or (dtype == np.int16) or (dtype == dtypes.int16_ref) or (dtype == dtypes.int16):
            return 2
        else:
            raise Exception("Unsupported tensor data type %s" % p_tTensor.dtype)
    #------------------------------------------------------------------------------------
    def Calculate(self):
        self.Count = 0
        self.MemoryUsed = 0
        for tParam in self.VarTensors:
            nShape = tParam.get_shape().as_list()
            
            dBytes = self.__bytesByDataType(tParam)
            
            if len(nShape) == 4:
                # Kernel and bias
                nParams = nShape[0]*nShape[1]*nShape[2]*nShape[3]
            elif len(nShape) == 2:
                # Weights or Biases
                nParams = nShape[0]*nShape[1]
            elif len(nShape) == 1:
                nParams = nShape[0]                
            else:
                nParams = 0
                
            self.Count += nParams
            self.MemoryUsed += nParams * dBytes
            if type(self).__verboseLevel >= 2:
                print("[%d] %d %.3f KB" % (self.Index, nParams, (nParams * dBytes) / (1024.0*1024.0)), nShape)
    #------------------------------------------------------------------------------------
    
#==================================================================================================











#==================================================================================================
class ConvolutionalLayerParamsItem(ParamsItem):
    #------------------------------------------------------------------------------------
    def __init__(self, p_nIndex):
        super(ConvolutionalLayerParamsItem, self).__init__(p_nIndex)
        #........ |  Instance Attributes | ..............................................        
        #................................................................................
    #------------------------------------------------------------------------------------
    def GetNames(self):
        sWeightVarName = "conv.W%02i" % self.Index
        sBiasVarName   = "conv.B%02i" % self.Index
        return sWeightVarName, sBiasVarName
    #------------------------------------------------------------------------------------
#==================================================================================================




    
    
    
#==================================================================================================
class FullyConnectedParamsItem(ParamsItem):
    #------------------------------------------------------------------------------------
    def __init__(self, p_nIndex):
        super(FullyConnectedParamsItem, self).__init__(p_nIndex)
        #........ |  Instance Attributes | ..............................................        
        #................................................................................
                    
    #------------------------------------------------------------------------------------
    def GetNames(self):
        sWeightVarName = "fc.W%02i" % self.Index
        sBiasVarName   = "fc.B%02i" % self.Index  
        return sWeightVarName, sBiasVarName
    #------------------------------------------------------------------------------------
#==================================================================================================
                                
        
        
        
        
        
      
        
        
#==================================================================================================
class ModelParamCollection(object):
    """
    Resuable learnable params generator. The same weights can be ported to other Tensorflow implementations
    """
    
    PSEUDO_RANDOM_SEED=2017
    __verboseLevel=0
    
    #------------------------------------------------------------------------------------
    def __init__(self):
        #........ |  Instance Attributes | ..............................................        
        self.RandomSeed=type(self).PSEUDO_RANDOM_SEED
        self.Trainable=[]
        self.Predefined=[]
        self.Items=[]
        #................................................................................            
            
    #------------------------------------------------------------------------------------
    @classmethod
    def SetRandomSeed(cls, p_nRandomSeed):
        type(cls).PSEUDO_RANDOM_SEED = p_nRandomSeed
    #------------------------------------------------------------------------------------
    def TotalParamCount(self):
        if type(self).__verboseLevel >= 3: 
            print("Calculating params ...")
        nCountOfTrainableLayers=0
        nCountOfTrainableParams = 0
        for nIndex, nParam in enumerate(self.Trainable):
            nShape = nParam.get_shape().as_list()
            
            if len(nShape) == 4:
                # Kernel and bias
                nParams = nShape[0]*nShape[1]*nShape[2]*nShape[3]
            elif len(nShape) == 2:
                # Weights or Biases
                nParams = nShape[0]*nShape[1]
            elif len(nShape) == 1:
                nParams = nShape[0]                
            else:
                nParams = 0
            
            if nIndex % 2 == 0:
                nCountOfTrainableLayers += 1
            nCountOfTrainableParams += nParams

            
        nCountOfPreDefinedParams = 0
        for nParam in self.Predefined:
            nShape = nParam.get_shape().as_list()
            
            if len(nShape) == 4:
                # Kernel
                nParams = nShape[0]*nShape[1] * nShape[2] * nShape[3]
            elif len(nShape) == 2:
                # Weights or Biases
                nParams = nShape[0]*nShape[1]
            elif len(nShape) == 1:
                nParams = nShape[0]                
            else:
                nParams = 0
            nCountOfPreDefinedParams += nParams
            
            
        return nCountOfPreDefinedParams, nCountOfTrainableParams, nCountOfTrainableLayers
    #------------------------------------------------------------------------------------
    def ConvolutionalLayerParams(self, p_nModuleIndex, p_nKernelShape, p_bIsWeightInitUniformScale=True, bIsBiasInitZero=True, p_nInitialKernelWeights=None, p_nInitialBiasesWeights=None, p_bIsAddingBias=True):
        oItem = ConvolutionalLayerParamsItem(p_nModuleIndex)
        self.Items.append(oItem)

        nFeatureDepth=p_nKernelShape[3]
        sWeightVarName, sBiasVarName = oItem.GetNames()
                                
        with tf.variable_scope("LearnedParams"):
            with tf.variable_scope("Module%02i" % p_nModuleIndex):
                # Initialize the convolution kernel weights
                if p_nInitialKernelWeights is None:
                    if p_bIsWeightInitUniformScale:
                        oWeightInitializer=uniform_scaling(shape=p_nKernelShape)
                    else:
                        #[MyMNIST]
                        oWeightInitializer=tf.truncated_normal(p_nKernelShape, stddev=0.1)
                    
                    tKernel = tf.get_variable(sWeightVarName, dtype=tf.float32, initializer=oWeightInitializer, trainable=True)
                else:
                    #[2017-06-08] PANTELIS
                    #oWeightInitializer=tf.constant(p_nInitialKernelWeights)
                    
                    oWeightInitializer=tf.constant_initializer(p_nInitialKernelWeights)
                    tKernel = tf.get_variable(sWeightVarName, shape=p_nKernelShape, dtype=tf.float32, initializer=oWeightInitializer, trainable=True)

                
                if p_bIsAddingBias:
                    if p_nInitialBiasesWeights is None:
                        if bIsBiasInitZero:
                            oBiasInitializer=tf.zeros(shape=[nFeatureDepth])
                            #oBiasInitializer=tf.constant_initializer(0.0, tf.float32)
                        else:
                            #[MyMNIST]
                            oBiasInitializer=tf.constant(0.1, dtype=tf.float32, shape=[nFeatureDepth])
                            #oBiasInitializer=tf.constant_initializer(0.1, tf.float32)
                    else:
                        oBiasInitializer=tf.constant(p_nInitialBiasesWeights)
                        
                    tBias = tf.get_variable(sBiasVarName, dtype=tf.float32, initializer=oBiasInitializer, trainable=True)
                else:
                    tBias = None
                    
        self.Trainable.append(tKernel)
        oItem.VarTensors.append(tKernel)
        if tBias is not None:
            self.Trainable.append(tBias)
            oItem.VarTensors.append(tBias)
        
        oItem.Calculate()
                           
        return tKernel, tBias
    #------------------------------------------------------------------------------------
    def FullyConnectedParams(self, p_nModuleIndex, p_nWeightShape, p_bAreZeroCenteredMeanWeights=True, p_bIsBiasInitZero=True):
        oItem = FullyConnectedParamsItem(p_nModuleIndex)
        self.Items.append(oItem)
        
        nFeaturesCount = p_nWeightShape[1]
        sWeightVarName, sBiasVarName = oItem.GetNames() 

        with tf.variable_scope("LearnedParams"):
            with tf.variable_scope("Module%02i" % p_nModuleIndex):
                if p_bAreZeroCenteredMeanWeights:
                    oWeightInitializer=tf.truncated_normal(shape=p_nWeightShape, mean=0.0, stddev=0.02, seed=self.RandomSeed)
                else:
                    #[MyMNIST]
                    oWeightInitializer=tf.truncated_normal(shape=p_nWeightShape, stddev=0.1, seed=self.RandomSeed)
                tWeights = tf.get_variable(sWeightVarName, dtype=tf.float32, initializer=oWeightInitializer, trainable=True)
                
                
                            
            
                if p_bIsBiasInitZero:
                    oBiasInitializer=tf.zeros(shape=[nFeaturesCount], dtype=tf.float32)
                    #oBiasInitializer=tf.constant_initializer(0.0, tf.float32)
                else:
                    #[MyMNIST]
                    oBiasInitializer=tf.constant(0.1, dtype=tf.float32, shape=[nFeaturesCount])
                    #oBiasInitializer=tf.constant_initializer(0.1, tf.float32)
                    
                
                tBias = tf.get_variable(sBiasVarName, dtype=tf.float32, initializer=oBiasInitializer, trainable=True)
                                                
        self.Trainable.append(tWeights)
        self.Trainable.append(tBias)
        
        oItem.VarTensors.append(tWeights)
        oItem.VarTensors.append(tBias)
        oItem.Calculate()
            
        return tWeights, tBias
    #------------------------------------------------------------------------------------
    def AdjustAfterReuse(self, p_nReuseDivisor=2):
        nMaxToKeep = int(len(self.Trainable) / p_nReuseDivisor)
        self.Trainable=self.Trainable[:nMaxToKeep]
        
        nMaxToKeep = int(len(self.Predefined) / p_nReuseDivisor)
        self.Predefined=self.Predefined[:nMaxToKeep]        
    #------------------------------------------------------------------------------------

#==================================================================================================







#------------------------------------------------------------------------------------
def uniform_scaling(shape=None, factor=1.0, dtype=tf.float32, seed=None, p_oNameScope=None):
    """ Uniform Scaling.

    Initialization with random values from uniform distribution without scaling
    variance.

    When initializing a deep network, it is in principle advantageous to keep
    the scale of the input variance constant, so it does not explode or diminish
    by reaching the final layer. If the input is `x` and the operation `x * W`,
    and we want to initialize `W` uniformly at random, we need to pick `W` from

      [-sqrt(3) / sqrt(dim), sqrt(3) / sqrt(dim)]

    to keep the scale intact, where `dim = W.shape[0]` (the size of the input).
    A similar calculation for convolutional networks gives an analogous result
    with `dim` equal to the product of the first 3 dimensions.  When
    nonlinearities are present, we need to multiply this by a constant `factor`.
    See [Sussillo et al., 2014](https://arxiv.org/abs/1412.6558)
    ([pdf](http://arxiv.org/pdf/1412.6558.pdf)) for deeper motivation, experiments
    and the calculation of constants. In section 2.3 there, the constants were
    numerically computed: for a linear layer it's 1.0, relu: ~1.43, tanh: ~1.15.

    Arguments:
        shape: List of `int`. A shape to initialize a Tensor (optional).
        factor: `float`. A multiplicative factor by which the values will be
            scaled.
        dtype: The tensor data type. Only float are supported.
        seed: `int`. Used to create a random seed for the distribution.

    Returns:
        The Initializer, or an initialized `Tensor` if shape is specified.

    """
    if shape:
        input_size = 1.0
        for dim in shape[:-1]:
            input_size *= float(dim)
        max_val = math.sqrt(3 / input_size) * factor
        
        return random_ops.random_uniform(shape, -max_val, max_val, dtype, seed=seed)
    else:
        return tf.uniform_unit_scaling_initializer(seed=seed, dtype=dtype)
#------------------------------------------------------------------------------------

















# DCNN Deprecated #

#------------------------------------------------------------------------------------
def avg_conv2d(p_oScope, p_nWeightID, p_nInputTensor, p_nInhibitionRate, p_nKernelShape, p_nConvStride, p_sPadding="VALID", p_bIsWeightInitUniformScale=True, p_bIsBiasInitZero=True, p_sName=None):

    w = p_nInhibitionRate / p_nKernelShape[0]
    print("avg mul", w)
    
    #TEMP: This is custom to 2x2 and li=2.0
    w_conv = tf.ones(p_nKernelShape, dtype=tf.float32)
    print("avg_conv_w", w_conv)
    
    
    #w_conv = tf.constant(w, dtype = tf.float32)
    
    #tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + p_oScope, w_conv)
    
    # convolution synaptic sum
    s_conv = tf.nn.conv2d(p_nInputTensor, w_conv, strides=[1, p_nConvStride, p_nConvStride, 1], padding=p_sPadding, name=p_sName)
    y_conv = tf.nn.tanh(s_conv)
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS + "/" + p_oScope, y_conv)
        
                
    return y_conv


#------------------------------------------------------------------------------------
def sum_conv2d(p_oScope, p_nWeightID, p_nInputTensor, p_nKernelShape, p_nConvStride, p_sPadding="VALID", p_bIsWeightInitUniformScale=True, p_bIsBiasInitZero=True, p_sName=None):
    
    if p_sName is None:
        sName = "conv.w%02i" % p_nWeightID
    else:
        sName = p_sName

    #with tf.name_scope(p_sName) as scope:
    # Initialize the convolution kernel weights
    if p_bIsWeightInitUniformScale:
        oWeightInitializer=uniform_scaling(shape=p_nKernelShape)
        w_conv = tf.get_variable(name=sName
                              ,dtype=tf.float32
                              ,initializer=oWeightInitializer
                              ,regularizer=None
                              ,trainable=True)
    else:
        #[MyMNIST]
        w_conv = tf.Variable(tf.truncated_normal(p_nKernelShape, stddev=0.1), name="conv.w%02i" % p_nWeightID, dtype = tf.float32, trainable=True)
        
    tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + p_oScope, w_conv)
    
    # Initialize the convolution kernel biases           
    if p_bIsBiasInitZero:
        b_conv = tf.Variable(initial_value=tf.zeros(shape=[p_nKernelShape[3]], dtype=tf.float32), trainable=True, name="conv.b%02i" % p_nWeightID)
    else:
        #[MyMNIST]
        b_conv = tf.Variable(initial_value=tf.constant(0.1, shape=[p_nKernelShape[3]]), trainable=True, name="conv.b%02i" % p_nWeightID, dtype = tf.float32)
    
    tf.add_to_collection(tf.GraphKeys.BIASES + "/" + p_oScope, b_conv)

    # convolution synaptic sum
    s_conv = tf.nn.conv2d(p_nInputTensor, w_conv, strides=[1, p_nConvStride, p_nConvStride, 1], padding=p_sPadding, name="conv2d_%02i" % p_nWeightID)
    y_conv = tf.nn.bias_add(s_conv, b_conv)
    
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS + "/" + p_oScope, y_conv)
        
                
    return y_conv
#------------------------------------------------------------------------------------
def _not_sum_conv2d(p_sName, p_nWeightID, p_nInputTensor, p_nKernelShape, p_nConvStride, p_sPadding="VALID", p_bIsWeightInitUniformScale=False, p_bIsBiasInitZero=True, p_WeightInitPseudoRandomSeed=2000):
    
    with tf.name_scope(p_sName) as scope:
        # Initialize the convolution kernel weights
        if p_bIsWeightInitUniformScale:
            #oWeightInitializer=mwe.uniform_scaling(shape=p_nKernelShape)
            oWeightInitializer=tf.uniform_unit_scaling_initializer(seed=p_WeightInitPseudoRandomSeed, dtype=tf.float32)
            w_conv = tf.get_variable(name="conv.w%02i" % p_nWeightID, shape=p_nKernelShape 
                                     ,dtype=tf.float32
                                     ,initializer=oWeightInitializer
                                     ,regularizer=None
                                     ,trainable=True)
        else:
            #[MyMNIST]
            w_conv = tf.Variable(tf.truncated_normal(p_nKernelShape, stddev=0.1), name="conv.w%02i" % p_nWeightID, dtype = tf.float32, trainable=True)
            
        tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + scope, w_conv)
        
        # Initialize the convolution kernel biases           
        if p_bIsBiasInitZero:
            b_conv = tf.Variable(initial_value=tf.zeros(shape=[p_nKernelShape[3]], dtype=tf.float32), trainable=True, name="conv.b%02i" % p_nWeightID)
            #oBiasInitializer=tf.constant_initializer(0.0, tf.float32)
#                 b_conv = tf.get_variable(shape=[p_nKernelShape[3]], name= "conv.b%02i" % p_nWeightID
#                                       ,dtype=tf.float32
#                                       ,initializer=oBiasInitializer
#                                       ,regularizer=None
#                                       ,trainable=True)
            
                        
            
            
            
        else:
            #[MyMNIST]
            b_conv = tf.Variable(initial_value=tf.constant(0.1, shape=[p_nKernelShape[3]]), trainable=True, name="conv.b%02i" % p_nWeightID, dtype = tf.float32)
        
        tf.add_to_collection(tf.GraphKeys.BIASES + "/" + scope, b_conv)

        # convolution synaptic sum
        s_conv = tf.nn.conv2d(p_nInputTensor, w_conv, strides=[1, p_nConvStride, p_nConvStride, 1], padding=p_sPadding, name="conv2d_%02i" % p_nWeightID)
        y_conv = tf.nn.bias_add(s_conv, b_conv)
        
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS + "/" + scope, y_conv)
        
                
    return y_conv
#------------------------------------------------------------------------------------        
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name, dtype=tf.float32)
#------------------------------------------------------------------------------------        
def softmax_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]), dtype=tf.float32)

    fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)

    return fc_h
#------------------------------------------------------------------------------------        
def conv_layer(inpt, filter_shape, stride):
    out_channels = filter_shape[3]

    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta", dtype=tf.float32)
    gamma = weight_variable([out_channels], name="gamma")
    
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)

    out = tf.nn.relu(batch_norm)

    return out    
#------------------------------------------------------------------------------------      