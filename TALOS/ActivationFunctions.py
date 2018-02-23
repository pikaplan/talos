# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        NEURAL NETWORK ACTIVATION FUNCTIONS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import tensorflow as tf
import TALOS.Constants as tcc



#------------------------------------------------------------------------------------
def ActivationFunctionDescr(p_nActivationFunctionType):
    if p_nActivationFunctionType==tcc.NAF_LINEAR:
        return "f(x)=s"            
    elif p_nActivationFunctionType==tcc.NAF_TANH:
        return "f(x)=tanh(s)"
    elif p_nActivationFunctionType==tcc.NAF_SIGMOID:
        return "f(x)=sig(s)"
    elif p_nActivationFunctionType==tcc.NAF_SOFTMAX:
        return "f(x)=softmax(s)"
    elif p_nActivationFunctionType==tcc.NAF_RELU:
        return "f(x)=relu(s)"
    elif p_nActivationFunctionType==tcc.NAF_RELU6:
        return "f(x)=relu6(s)"
    elif p_nActivationFunctionType==tcc.NAF_SOFTPLUS:
        return "f(x)=softplus(s)"
    elif p_nActivationFunctionType==tcc.NAF_SOFTSIGN:
        return "f(x)=softsign(s)"
    elif p_nActivationFunctionType==tcc.NAF_EXP_RELU:
        return "f(x)=exprelu(s)"
    elif p_nActivationFunctionType==tcc.NAF_LEEKY_RELU:
        return "f(x)=leekyrelu(s)"
    elif p_nActivationFunctionType==tcc.NAF_ELU:
        return "f(x)=elu(s)"
    elif p_nActivationFunctionType==tcc.NAF_BN_ELU:
        return "f(x)=elu(batchnorm(s))"
    elif p_nActivationFunctionType==tcc.NAF_RESTRICTED_LINEAR:
        return "f(x)=x*exp(1-|x|)"
    elif p_nActivationFunctionType==tcc.NAF_PRELU:
        return "f(x)=prelu(s)"

#------------------------------------------------------------------------------------
def ActivationFunction(p_nFunctionType, x, p_nLeekyReluAlpha=0.1, p_bIsBatchNormalization=False):
    if p_nFunctionType == tcc.NAF_LINEAR:
        y = x
    elif p_nFunctionType == tcc.NAF_TANH:
        y = tf.tanh(x)
    elif p_nFunctionType == tcc.NAF_SIGMOID:
        y = tf.nn.sigmoid(x)
    elif p_nFunctionType == tcc.NAF_SOFTMAX:
        y = tf.nn.softmax(x)
    elif p_nFunctionType == tcc.NAF_RELU:
        y = tf.nn.relu(x)
    elif p_nFunctionType == tcc.NAF_RELU6:
        y = tf.nn.relu6(x)
    elif p_nFunctionType == tcc.NAF_SOFTPLUS:
        y = tf.nn.softplus(x)
    elif p_nFunctionType == tcc.NAF_SOFTSIGN:
        y = tf.nn.softsign(x)
    elif p_nFunctionType == tcc.NAF_EXP_RELU:
        y = tf.nn.elu()
    elif p_nFunctionType == tcc.NAF_LEEKY_RELU:
        y = None        
    elif p_nFunctionType == tcc.NAF_ELU:
        y = tf.nn.elu(x)
    elif p_nFunctionType == tcc.NAF_BN_ELU:
        y=None
    elif p_nFunctionType==tcc.NAF_RESTRICTED_LINEAR:
        y = tf.multiply( x, tf.exp( tf.subtract(tf.constant(1.0, tf.float32), tf.abs(x))) )
    elif p_nFunctionType==tcc.NAF_PRELU:
        y = PReLU(x)      
    else:
        y = None
        
        
    assert y is not None
    
    return y
#------------------------------------------------------------------------------------
def ActivationFunctionNodeName(p_nFunctionType, p_nIndex):
    if p_nFunctionType == tcc.NAF_LINEAR:
        sResult = "Linear%02i" % p_nIndex
    elif p_nFunctionType == tcc.NAF_TANH:
        sResult = "TanH%02i" % p_nIndex
    elif p_nFunctionType == tcc.NAF_SIGMOID:
        sResult = "Sig%02i"
    elif p_nFunctionType == tcc.NAF_SOFTMAX:
        sResult = "SMax%02i"
    elif p_nFunctionType == tcc.NAF_RELU:
        sResult = "ReLU%02i"
    elif p_nFunctionType == tcc.NAF_RELU6:
        sResult = "ReLU6%02i"
    elif p_nFunctionType == tcc.NAF_SOFTPLUS:
        sResult = "SPlus%02i"
    elif p_nFunctionType == tcc.NAF_SOFTSIGN:
        sResult = "SSign%02i"        
    elif p_nFunctionType == tcc.NAF_EXP_RELU:
        sResult = "ExpReLU%02i"        
    elif p_nFunctionType == tcc.NAF_LEEKY_RELU:
        sResult = "LReLU%02i"
    elif p_nFunctionType == tcc.NAF_ELU:
        sResult = "ELU%02i"        
    elif p_nFunctionType == tcc.NAF_BN_ELU:
        sResult = "BN_ELU%02i"        
    elif p_nFunctionType==tcc.NAF_RESTRICTED_LINEAR:
        sResult = "BoundLinear%02i"
    elif p_nFunctionType==tcc.NAF_PRELU:
        sResult = "PReLU%02i"
    else:
        sResult = None
    return sResult
#------------------------------------------------------------------------------------
def BatchNormalization(p_tSynapticSumOrActivation, p_nFeatureDepth, p_tIsTraining, p_sScope, p_nModuleIndex):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        p_tSynapticSumOrActivation:         Tensor, 4D BHWD input maps
        p_nFeatureDepth:                    integer, depth of input maps
        p_tIsTraining:                      boolean tf.Varialbe, true indicates training phase
        p_sScope:                           string, variable scope
        p_nModuleIndex:                     module index for proper naming
    Return:
        tBatchNormLayer:                   batch-normalized maps
    """

    with tf.variable_scope(p_sScope):
        tBeta = tf.Variable(tf.constant(0.0, shape=[p_nFeatureDepth], dtype=tf.float32) , name="BNBeta%02i" % p_nModuleIndex, trainable=True, dtype=tf.float32)
        tGamma = tf.Variable(tf.constant(1.0, shape=[p_nFeatureDepth], dtype=tf.float32), name="BNGamma%02i" % p_nModuleIndex, trainable=True, dtype=tf.float32)
        batch_mean, batch_var = tf.nn.moments(p_tSynapticSumOrActivation, [0,1,2], name="BNMoments%02i" % p_nModuleIndex)
        tExpMovAvg = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = tExpMovAvg.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        tMean, tVar = tf.cond(p_tIsTraining,
                            mean_var_with_update,
                            lambda: (tExpMovAvg.average(batch_mean), tExpMovAvg.average(batch_var)))
        tBatchNormLayer = tf.nn.batch_normalization(p_tSynapticSumOrActivation, tMean, tVar, tBeta, tGamma, 1e-3)
    
    return tBatchNormLayer
#------------------------------------------------------------------------------------
def PReLU(x, p_nInitialValue=0.0):
    tAlphas = tf.get_variable('PReLU_Alpha', x.get_shape()[-1],
                          initializer = tf.constant_initializer(p_nInitialValue),
                          dtype = tf.float32)
    tPositive = tf.nn.relu(x)
    tNegative = tAlphas * (x - abs(x)) * 0.5
    
    return tPositive + tNegative
#------------------------------------------------------------------------------------











  