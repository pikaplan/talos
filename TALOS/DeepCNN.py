# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.1.0-ALPHA
#        DEEP CONVOLUTIONAL NEURAL NETWORKS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import numpy as np
import tensorflow as tf
#from tensorflow.contrib.slim import avg_pool2d
import TALOS.Constants as tcc
import TALOS.ActivationFunctions as mfunc
#from TALOS.Core import GetActiveLog
from TALOS.ActivationFunctions import ActivationFunctionDescr
#from TALOS.Params import sum_conv2d, avg_conv2d, conv_layer
from TALOS.NeuralNetworks import BaseNN, BaseNeuralModule
#from TALOS.Constants import NAF_RELU


#==================================================================================================
class NNPoolType():
    MAX = 0
    AVERAGE = 1
    GLOBAL_AVERAGE = 2
#==================================================================================================

    
    
#==================================================================================================
class ArchitectureDefKind():
    CONV_MAX_POOL_LRN = 0
    FULLY_CONNECTED = 1
    SOFTMAX = 2
    VGG_MODULE = 3
    
    
#==================================================================================================





#==================================================================================================
class ArchitectureDefItem(object):
    #------------------------------------------------------------------------------------
    def __init__(self, p_nKind, p_nModuleIndex, p_tInput):
        #........ |  Instance Attributes | ..............................................
        self.Kind   = p_nKind
        self.ModuleIndex = p_nModuleIndex
        self.Children = []
        
        self.InputSampleCount = None
        self.InputSizeX       = None
        self.InputSizeY       = None 
        self.InputFeatures     = None
        self.InputFeaturesFlat = None

        self.ConvPadding = None   
        self.ConvPaddingStr = None     
        self.ConvWindowSize = None
        self.ConvWindowSizeY = None
        self.ConvStride = None
        self.ConvInFeatures = None
        self.ConvOutFeatures = None
        
        self.ConvOutSizeX = None
        self.ConvOutSizeY = None
        self.ActivationFunction =  None

        self.IsPooling = False
        self.PoolPadding = None
        self.PoolPaddingStr = None
        self.PoolDim = None
        self.PoolStride = None
        self.PoolType = None
        self.PoolTypeStr = None
        
        self.LayerWindowSizeX = None
        self.LayerWindowSizeY = None
        self.LayerOutFeatures = None
        
                
        self.IsLRN = False
        self.IsDropOut = False 
        self.DropOutKeepProb = 1.0
        self.Extra = ""
        
        #................................................................................
        self.SetInput(p_tInput)
    #------------------------------------------------------------------------------------
    def AddChild(self, p_nKind, p_tInput):
        oChild = ArchitectureDefItem(p_nKind, len(self.Children) + 1, p_tInput)
        self.Children.append(oChild)
        return oChild
    #------------------------------------------------------------------------------------
    def SetInput(self, p_tInputTensor):
        nInputSampleCount   = None
        nInputWindowSizeX   = None
        nInputWindowSizeY   = None
        nInputFeatures      = None
        nInputFeaturesFlat  = None
        
        assert p_tInputTensor is not None, "Input tensor missing in architecture item"
        
        if p_tInputTensor.get_shape().ndims==4:
            nInputSampleCount=p_tInputTensor.get_shape()[0].value
            if nInputSampleCount is None:
                nInputSampleCount=-1

            nInputWindowSizeX=p_tInputTensor.get_shape()[1].value
            nInputWindowSizeY=p_tInputTensor.get_shape()[2].value
            nInputFeatures=p_tInputTensor.get_shape()[3].value
            nInputFeaturesFlat=nInputWindowSizeX*nInputWindowSizeY*nInputFeatures

        elif p_tInputTensor.get_shape().ndims==2:
            nInputSampleCount=p_tInputTensor.get_shape()[0].value
            if nInputSampleCount is None:
                nInputSampleCount=-1

            nInputFeatures=p_tInputTensor.get_shape()[1].value
            nInputFeaturesFlat=nInputFeatures
                
        self.InputSampleCount = nInputSampleCount
        self.InputSizeX = nInputWindowSizeX
        self.InputSizeY = nInputWindowSizeY 
        self.InputFeatures = nInputFeatures
        self.InputFeaturesFlat = nInputFeaturesFlat
    #------------------------------------------------------------------------------------
    def __paddingTypeToStr(self, p_nPaddingType):
        if p_nPaddingType == tcc.TF_ZERO_PADDING:
            sResult="zero"
        elif p_nPaddingType == tcc.TF_NO_PADDING:
            sResult="    "
        else:
            sResult = "?   "
        return sResult
    #------------------------------------------------------------------------------------
    def SetConvFilter(self, p_tConvFilter, p_nConvStride, p_nConvPadding):
        self.ConvWindowSizeX = p_tConvFilter.get_shape()[0].value
        self.ConvWindowSizeY = p_tConvFilter.get_shape()[1].value
        self.ConvInFeatures = p_tConvFilter.get_shape()[2].value
        self.ConvOutFeatures = p_tConvFilter.get_shape()[3].value
        self.ConvStride = p_nConvStride
        self.ConvPadding = p_nConvPadding
        self.ConvPaddingStr = self.__paddingTypeToStr(p_nConvPadding)
    #------------------------------------------------------------------------------------
    def SetPooling(self, p_nPoolType, p_nPoolDim, p_nPoolStride, p_nPoolPadding):
        self.IsPooling = True
        self.PoolDim = p_nPoolDim
        self.PoolStride = p_nPoolStride
        self.PoolType = p_nPoolType
        if self.PoolType == NNPoolType.MAX:
            self.PoolTypeStr = "max pool"
        elif self.PoolType == NNPoolType.AVERAGE:
            self.PoolTypeStr = "avg pool"
        elif self.PoolType == NNPoolType.GLOBAL_AVERAGE:
            self.PoolTypeStr = "glavp"
        else:
            self.PoolTypeStr = None
        self.PoolPadding = p_nPoolPadding
        self.PoolPaddingStr = self.__paddingTypeToStr(p_nPoolPadding) 
    #------------------------------------------------------------------------------------
    def SetDropout(self, p_nDropOutKeepProb):
        self.IsDropOut = True
        self.DropOutKeepProb = p_nDropOutKeepProb
    #------------------------------------------------------------------------------------
    def SetFunction(self, p_nConvTensor, p_nActivationFunction):
        nShape = p_nConvTensor.get_shape()
        
        self.ActivationFunction = p_nActivationFunction
        self.ConvOutSizeX = int(nShape[1])
        self.ConvOutSizeY = int(nShape[2])
        
        assert self.InputSampleCount == int(nShape[0])
        assert self.ConvOutFeatures == int(nShape[3])
    #------------------------------------------------------------------------------------
    def SetOutput(self, p_tLayerTensor):
        self.LayerWindowSizeX = None
        self.LayerWindowSizeY = None
        self.LayerOutFeatures = None
        
        if p_tLayerTensor.get_shape().ndims==4:
            nLayerSampleCount=p_tLayerTensor.get_shape()[0].value
            if nLayerSampleCount is None:
                nLayerSampleCount=-1
            self.LayerWindowSizeX=p_tLayerTensor.get_shape()[1].value
            self.LayerWindowSizeY=p_tLayerTensor.get_shape()[2].value
            self.LayerOutFeatures=p_tLayerTensor.get_shape()[3].value
            
            assert nLayerSampleCount==self.InputSampleCount            
        elif p_tLayerTensor.get_shape().ndims==2:
            nLayerSampleCount=p_tLayerTensor.get_shape()[0].value
            if nLayerSampleCount is None:
                nLayerSampleCount=-1
    
            self.LayerOutFeatures=p_tLayerTensor.get_shape()[1].value
            
            assert nLayerSampleCount==self.InputSampleCount
    #------------------------------------------------------------------------------------
    def WriteTo(self, p_oFile, p_nIdent=0):
        sFunc = ActivationFunctionDescr(self.ActivationFunction)
        
        if self.IsLRN:
            sExtra = "LRN"
        else:
            sExtra = "   "
            
        if p_nIdent == 0:
            sIdent = ""
        else: 
            sIdent = " " * p_nIdent * 4
        if self.Kind == ArchitectureDefKind.CONV_MAX_POOL_LRN:
            sText = "%s[%.2d]  Convolutional Module   ;in [%dx%d|%d] ;conv [%dx%d/%d|%d->%d] %s ,%s => ;[%dx%d|%d] %s" % (
                       sIdent   
                      ,self.ModuleIndex
                      
                      ,self.InputSizeX
                      ,self.InputSizeY
                      ,self.InputFeatures
                      
                      ,self.ConvWindowSizeX
                      ,self.ConvWindowSizeY
                      ,self.ConvStride
                      ,self.InputFeatures
                      ,self.ConvOutFeatures
                      ,self.ConvPaddingStr
                      
                      ,sFunc
                      ,self.ConvOutSizeX
                      ,self.ConvOutSizeY
                      ,self.ConvOutFeatures
                      ,sExtra
                   )
            if self.IsPooling:
                sText += ";%s [%dx%d/%d] => ;[%dx%d|%d]" % (
                         self.PoolTypeStr
                        ,self.PoolDim
                        ,self.PoolDim
                        ,self.PoolStride
                        ,self.LayerWindowSizeX
                        ,self.LayerWindowSizeY
                        ,self.LayerOutFeatures
                    )
                 
            print(sText, file=p_oFile)
        elif self.Kind == ArchitectureDefKind.FULLY_CONNECTED:
            if self.IsDropOut: 
                sDropOut = "do:%.2f" % self.DropOutKeepProb
            else:
                sDropOut = ""
            sText = ("%s[%.2d]  Fully Connected Module ;in [%d] ;%s, %s => ;[%d] %s" % ( 
                       sIdent   
                      ,self.ModuleIndex
                      ,self.InputFeaturesFlat
                      ,sFunc
                      ,sDropOut                      
                      ,self.LayerOutFeatures
                      ,sExtra
                   )
                 )
            print(sText, file=p_oFile)
        elif self.Kind == ArchitectureDefKind.SOFTMAX:
            sText = ("%s[%.2d]  Softmax Output          ;in [%d] ;%s => ;[%d] %s" % (
                       sIdent   
                      ,self.ModuleIndex
                      ,self.InputFeaturesFlat
                      ,sFunc
                      ,self.LayerOutFeatures
                      ,sExtra
                   )
                 )
            print(sText, file=p_oFile)
            
        elif self.Kind == ArchitectureDefKind.VGG_MODULE:
            sText = "%s[%.2d]  VGG Module             ;in [%dx%d|%d] ;;out [%dx%d|%d] %s" % (
                       sIdent   
                      ,self.ModuleIndex
                      
                      ,self.InputSizeX
                      ,self.InputSizeY
                      ,self.InputFeatures
                      
                      ,self.Children[-1].ConvOutSizeX
                      ,self.Children[-1].ConvOutSizeY
                      ,self.Children[-1].ConvOutFeatures
                      ,sExtra
                   )    
            print(sText, file=p_oFile)
                    
            for nIndex, nChild in enumerate(self.Children):
                nChild.WriteTo(p_oFile, p_nIdent + 1)
                
            if self.IsPooling:
                sText = "%s[%.2d] %s;in [%dx%d|%d] ;[%dx%d/%d] => ;[%dx%d|%d]" % (
                         " " * (p_nIdent + 1) * 4
                        ,len(self.Children) + 1
                        ,self.PoolTypeStr
                        
                        ,self.Children[-1].ConvOutSizeX
                        ,self.Children[-1].ConvOutSizeY
                        ,self.Children[-1].ConvOutFeatures                        
                        
                        ,self.PoolDim
                        ,self.PoolDim
                        ,self.PoolStride
                        ,self.LayerWindowSizeX
                        ,self.LayerWindowSizeY
                        ,self.LayerOutFeatures
                    )    
                print(sText, file=p_oFile)                            
    #------------------------------------------------------------------------------------                
#==================================================================================================



















    


#==================================================================================================
class InputCroppedColorImageAndClass(BaseNeuralModule):
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_nTargetImageDimensions, p_nInputDimensions, p_nImageBatchSize):
        super(InputCroppedColorImageAndClass, self).__init__(p_oParent, 0, p_nOutputFeatureDepth=3)
        #........ |  Instance Attributes | ..............................................
        self.TargetImageDimensions  = p_nTargetImageDimensions
        self.InputDimensions        = p_nInputDimensions
        self.WidthRatio             = self.TargetImageDimensions[0] / self.InputDimensions[0]
        self.HeightRatio            = self.TargetImageDimensions[1] / self.InputDimensions[1]
        self.ImageBatchSize = p_nImageBatchSize

        self.TrainInput=None
        self.TrainTargets=None
        self.ValInput=None
        self.ValTargets=None
        
        self.TrainOutput=None
        self.ValOutput=None
        #................................................................................
        #self.Input=tf.placeholder(dtype=tf.float32, shape=[1,self.InputDimensions[0],self.InputDimensions[1],3], name="RGBImage")
        
        #TEMP
#         tf.pad(
#             tf.placeholder(dtype=tf.float32, shape=[1,self.InputDimensions[0],self.InputDimensions[1],3], name="RGBImage")
#             ,[[1, 1,], [1, 1]]
#             ,mode="CONSTANT"
#             ,constant_values=0
#             )
#         print(self.Input.get_shape())
    #------------------------------------------------------------------------------------        
    def _cropImage(self, p_nInputTensor):
        nDiffW = self.InputDimensions[0] - self.TargetImageDimensions[0]
        nDiffH = self.InputDimensions[1] - self.TargetImageDimensions[1]
        assert (nDiffW > 0) and (nDiffH > 0), "Invalid target dimensions for cropping" 
        
        nTop = nDiffW // 2
        nHeight = nDiffH // 2
        
        return tf.image.crop_to_bounding_box(p_nInputTensor, nTop, nHeight, self.TargetImageDimensions[0], self.TargetImageDimensions[1])
    #------------------------------------------------------------------------------------
    def ModuleOnCreateTensors(self, p_bIsTrainingGraph):
        if p_bIsTrainingGraph:
            self.TrainInput=tf.placeholder(dtype=np.uint8, shape=[self.ImageBatchSize,self.InputDimensions[0],self.InputDimensions[1],3], name="TImageRGB")
            tNormalizedImage=tf.div(tf.cast(self.TrainInput, dtype=tf.float32), tf.constant(255.0, dtype=tf.float32), name="TImageRGBNormCrop")
            self.TrainOutput=self._cropImage(tNormalizedImage)
            self.Output = self.TrainOutput
            
            self.TrainTargets=tf.placeholder(dtype=np.int32, shape=[self.ImageBatchSize,1])
        else:
            self.ValInput=tf.placeholder(dtype=np.uint8, shape=[self.ImageBatchSize,self.InputDimensions[0],self.InputDimensions[1],3], name="VImageRGB")
            tNormalizedImage=tf.div(tf.cast(self.ValInput, dtype=tf.float32), tf.constant(255.0, dtype=tf.float32), name="VImageRGBNormCrop")
            self.ValOutput=self._cropImage(tNormalizedImage)          
            self.Output = self.ValOutput

            self.ValTargets=tf.placeholder(dtype=np.int32, shape=[self.ImageBatchSize,1])
    #------------------------------------------------------------------------------------
#================================================================================================== 






#==================================================================================================
class InputColorImageAndClass(BaseNeuralModule):
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_nInputDimensions, p_nImageBatchSize):
        super(InputColorImageAndClass, self).__init__(p_oParent, 0, p_nOutputFeatureDepth=3)
        #........ |  Instance Attributes | ..............................................
        self.InputDimensions = p_nInputDimensions
        self.ImageBatchSize = p_nImageBatchSize

        self.TrainInput=None
        self.TrainTargets=None
        self.ValInput=None
        self.ValTargets=None

        self.TrainOutput=None
        self.ValOutput=None
        
        # Iteration positions
        self.TrainingPos = None
        self.ValidationPos = None
        #................................................................................
    #------------------------------------------------------------------------------------
    def ModuleOnCreateTensors(self, p_bIsTrainingGraph):
        if p_bIsTrainingGraph:
            self.TrainInput=tf.placeholder(dtype=np.uint8, shape=[self.ImageBatchSize,self.InputDimensions[0],self.InputDimensions[1],3], name="TImageRGB")
            tNormalizedImage=tf.div(tf.cast(self.TrainInput, dtype=tf.float32), tf.constant(255.0, dtype=tf.float32), name="TImageRGBNorm")
            self.TrainOutput=tNormalizedImage
            self.Output = self.TrainOutput
            
            self.TrainTargets=tf.placeholder(dtype=np.int32, shape=[self.ImageBatchSize,1])
        else:
            self.ValInput=tf.placeholder(dtype=np.uint8, shape=[self.ImageBatchSize,self.InputDimensions[0],self.InputDimensions[1],3], name="VImageRGB")
            tNormalizedImage=tf.div(tf.cast(self.ValInput, dtype=tf.float32), tf.constant(255.0, dtype=tf.float32), name="VImageRGBNorm")
            self.ValOutput=tNormalizedImage
            self.Output = self.ValOutput

            self.ValTargets=tf.placeholder(dtype=np.int32, shape=[self.ImageBatchSize,1])
    #------------------------------------------------------------------------------------
#==================================================================================================        


 
 
 
 
 
 
 
#==================================================================================================
class ConvMaxPoolLRN(BaseNeuralModule):
    __verboseLevel=1
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_nModuleIndex, p_nOutputFeatureDepth, p_nConvKernelRect, p_nConvStride, p_nPoolRect, p_nPoolStride, p_bIsLRN=True, p_nActivationType=tcc.NAF_RELU, p_bIsPaddingConv=True, p_bIsPaddingPool=False):
        super(ConvMaxPoolLRN, self).__init__(p_oParent, p_nModuleIndex, p_nOutputFeatureDepth)
                
        #........ |  Instance Attributes | ..............................................
        
        self.ConvKernelRect=p_nConvKernelRect
        self.ConvStride=p_nConvStride
        
        self.PoolRect=p_nPoolRect
        self.PoolStride=p_nPoolStride
        
        self.IsPooling=True
        if p_nPoolRect is None:
            self.IsPooling=False
        else:
            self.IsPooling=p_nPoolRect[0] != 0
        
        self.IsLRN=p_bIsLRN
        self.ActivationFunctionType=p_nActivationType
        self.IsPaddingConv=p_bIsPaddingConv
        self.IsPaddingPool=p_bIsPaddingPool
        #................................................................................
    #------------------------------------------------------------------------------------
    def ModuleOnCreateTensors(self, p_bIsTrainingGraph):
        tInput, nPreviousModuleFeatures = self.Parent.GetPreviousModuleOutput(self.ModuleIndex)
        self.Input=tInput
        sComment=""

        nDefItem = ArchitectureDefItem(ArchitectureDefKind.CONV_MAX_POOL_LRN, self.ModuleIndex, self.Input)
        
        if self.IsPaddingConv:
            sConvPaddingType=tcc.TF_ZERO_PADDING 
        else:
            sConvPaddingType=tcc.TF_NO_PADDING
            
        if self.IsPaddingPool:
            sPoolPaddingType=tcc.TF_ZERO_PADDING
        else:
            sPoolPaddingType=tcc.TF_NO_PADDING

        sScopeName="ConvMod%02i" % self.ModuleIndex
        with tf.name_scope(sScopeName) as oScope:
            nKernelShape=(self.ConvKernelRect[0], self.ConvKernelRect[1], nPreviousModuleFeatures, self.OutputFeatureDepth)
            tKernels, tBiases = self.Parent.ParamVars.ConvolutionalLayerParams(self.ModuleIndex, nKernelShape, p_bIsWeightInitUniformScale=True, bIsBiasInitZero=True)
            nDefItem.SetConvFilter(tKernels, self.ConvStride, sConvPaddingType)
            
            # Convolutional layer
            nStrides=[1, self.ConvStride, self.ConvStride, 1]
            tConv = tf.nn.bias_add( tf.nn.conv2d(self.Input, tKernels, strides=nStrides, padding=sConvPaddingType, name="conv2d_%02i" % self.ModuleIndex)
                                   ,tBiases)
            
            
            # [TODO] Check Scope
            # Batch normalization is used only for training
#             if p_bIsBatchNormalization:
#                 tSynapticSum = mfunc.BatchNormalization(tConv, p_nFeatureDepth, self.IsTrainingProcessTensor(), sScopeName, p_nModuleIndex)
#                 sComment=ActivationFunctionDescr(p_nFunction), " batch norm"
#             else:
#                 ...
            tSynapticSum = tConv
            tFunc = mfunc.ActivationFunction(self.ActivationFunctionType, tSynapticSum)
            sComment = sComment + ActivationFunctionDescr(self.ActivationFunctionType)
            nDefItem.SetFunction(tFunc, self.ActivationFunctionType)
            
            tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + oScope    , tKernels)
            tf.add_to_collection(tf.GraphKeys.BIASES + "/" + oScope     , tBiases)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tSynapticSum)            
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tFunc)
            self.Parent.Layers.append(tFunc)            
            
            
            # Max pooling layer
            if self.IsPooling:
                nMaxPoolShape=(1, self.PoolRect[0], self.PoolRect[0], 1)
                nStrides=[1, self.PoolStride, self.PoolStride, 1]
                tPool = tf.nn.max_pool(tFunc, ksize=nMaxPoolShape, strides=nStrides, padding=sPoolPaddingType, name="maxpool_%02i" % self.ModuleIndex)
                
                self.Parent.Layers.append(tPool)
                tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tPool)
                nDefItem.SetPooling(NNPoolType.MAX, self.PoolRect[0], self.PoolStride, sPoolPaddingType)
            else:
                tPool=tFunc
        

            # It will add an LRN even when the max pooling layer is disabled, if the pooling stride is not zero
            if self.IsLRN:
                tOut = tf.nn.local_response_normalization(tPool, name="lrn_%02i" % self.ModuleIndex)
                sComment = sComment + ", lrn"
                tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tOut)
                nDefItem.IsLRN = True
            else:
                tOut=tPool                
            
            self.Parent.Layers.append(tOut)
    


        self.Parent.SetTensor(self.ModuleIndex, tOut)

        self.Output = tOut

        nDefItem.SetOutput(self.Output)
        self.Parent.Architecture.Add(nDefItem)

        if type(self).__verboseLevel >= 2:
            LogCNNLayerAndPool("[%02i] Convolutional Module" % self.ModuleIndex
                         ,self.Input, tFunc, self.Output 
                         ,p_tConvFilter=tKernels, p_nConvStride=self.ConvStride, p_sConvType=sConvPaddingType
                         ,p_nPoolRect=self.PoolRect, p_nPoolStride=self.PoolStride, p_sPoolType=sPoolPaddingType 
                         ,p_sComment=sComment
                        )
    #------------------------------------------------------------------------------------
#==================================================================================================






#==================================================================================================
class VGGModule(BaseNeuralModule):
    __verboseLevel=1
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_nModuleIndex, p_nOutputFeatureDepth, p_nConvLayerCount, p_nActivationType=tcc.NAF_RELU, p_bIsPaddingConv=True, p_bIsPaddingPool=False):
        super(VGGModule, self).__init__(p_oParent, p_nModuleIndex, p_nOutputFeatureDepth)
                
        #........ |  Instance Attributes | ..............................................
        self.ConvLayerCount=p_nConvLayerCount
        self.IsLRN=False
        self.ActivationFunctionType=p_nActivationType
        self.IsPaddingConv=p_bIsPaddingConv
        self.IsPaddingPool=p_bIsPaddingPool
        #................................................................................
    #------------------------------------------------------------------------------------
    def ModuleOnCreateTensors(self, p_bIsTrainingGraph):
        tInput, nPreviousModuleFeatures = self.Parent.GetPreviousModuleOutput(self.ModuleIndex)
        self.Input=tInput
        sComment=""
        
        nDefItem = ArchitectureDefItem(ArchitectureDefKind.VGG_MODULE, self.ModuleIndex, self.Input)
        
        if self.IsPaddingConv:
            sConvPaddingType=tcc.TF_ZERO_PADDING 
        else:
            sConvPaddingType=tcc.TF_NO_PADDING
            
        if self.IsPaddingPool:
            sPoolPaddingType=tcc.TF_ZERO_PADDING
        else:
            sPoolPaddingType=tcc.TF_NO_PADDING

        sScopeName="VGGMod%02i" % self.ModuleIndex
        with tf.name_scope(sScopeName) as oScope:
            nKernelShape=(3, 3, nPreviousModuleFeatures, self.OutputFeatureDepth)
            nStrides=[1, 1, 1, 1]
            
            tPreviousLayer = self.Input
            
            for nLayerIndex in range(0, self.ConvLayerCount):
                nDefSubItem = nDefItem.AddChild(ArchitectureDefKind.CONV_MAX_POOL_LRN, tPreviousLayer)
                
                # VGG 3x3 Convolutional layer
                #TODO: Better numbering of convolutional layer params
                tKernels, tBiases = self.Parent.ParamVars.ConvolutionalLayerParams(self.ModuleIndex*self.Parent.BigConvLayerCount + nLayerIndex, nKernelShape, p_bIsWeightInitUniformScale=True, bIsBiasInitZero=True)
                nDefSubItem.SetConvFilter(tKernels, 1, sConvPaddingType)                
                
                tConv = tf.nn.bias_add( tf.nn.conv2d(tPreviousLayer, tKernels, strides=nStrides, padding=sConvPaddingType, name="conv2d_%02i.%i" % (self.ModuleIndex, nLayerIndex)), tBiases)
                tSynapticSum = tConv
                tFunc = mfunc.ActivationFunction(self.ActivationFunctionType, tSynapticSum)
                nDefSubItem.SetFunction(tFunc, self.ActivationFunctionType)
                
                #print("VGG Layer %d" % nLayerIndex, tPreviousLayer, tFunc)
                tPreviousLayer = tFunc
                nKernelShape=(3, 3, self.OutputFeatureDepth, self.OutputFeatureDepth)
                
                sComment = sComment + ActivationFunctionDescr(self.ActivationFunctionType)
                
                tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + oScope    , tKernels)
                tf.add_to_collection(tf.GraphKeys.BIASES + "/" + oScope     , tBiases)
                tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tSynapticSum)            
                tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tFunc)
                self.Parent.Layers.append(tFunc)
                            
                nDefSubItem.SetOutput(tPreviousLayer)
            
            # Max pooling layer
            nMaxPoolShape=(1, 2, 2, 1)
            nStrides=[1, 2, 2, 1]
            tPool = tf.nn.max_pool(tFunc, ksize=nMaxPoolShape, strides=nStrides, padding=sPoolPaddingType, name="maxpool_%02i" % self.ModuleIndex)
            nDefItem.SetPooling(NNPoolType.MAX, 2, 2, sPoolPaddingType)
            
            self.Parent.Layers.append(tPool)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tPool)

            # It will add an LRN even when the max pooling layer is disabled, if the pooling stride is not zero
            if self.IsLRN:
                tOut = tf.nn.local_response_normalization(tPool, name="lrn_%02i" % self.ModuleIndex)
                sComment = sComment + ", lrn"
                tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tOut)
                nDefItem.IsLRN = True
            else:
                tOut=tPool                
            
            self.Parent.Layers.append(tOut)
    


        self.Parent.SetTensor(self.ModuleIndex, tOut)

        self.Output = tOut
        nDefItem.SetOutput(self.Output)
        self.Parent.Architecture.Add(nDefItem)


        if type(self).__verboseLevel >= 2:
            LogCNNLayerAndPool("[%02i] VGG Module" % self.ModuleIndex
                         ,self.Input, tFunc, self.Output 
                         ,p_tConvFilter=tKernels, p_nConvStride=1, p_sConvType=sConvPaddingType
                         ,p_nPoolRect=(2,2), p_nPoolStride=2, p_sPoolType=sPoolPaddingType 
                         ,p_sComment=sComment
                        )
    #------------------------------------------------------------------------------------    
#==================================================================================================





#==================================================================================================
class BasePoolModule(BaseNeuralModule):
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_nModuleIndex, p_nPoolRect, p_nPoolStride, p_bIsPaddingPool, p_bIsLRN):
        super(BasePoolModule, self).__init__(p_oParent, p_nModuleIndex, None)
                
        #........ |  Instance Attributes | ..............................................
        self.PoolRect = p_nPoolRect
        self.PoolStride = p_nPoolStride
        self.IsPaddingPool = p_bIsPaddingPool
        self.IsLRN = p_bIsLRN
        #................................................................................
    #------------------------------------------------------------------------------------
#==================================================================================================        
        
        
        
        
        
        
        
        
        
        
#==================================================================================================            
class MaxPool(BasePoolModule):
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_nModuleIndex, p_nPoolRect=[3,3], p_nPoolStride=2, p_bIsPaddingPool=False, p_bIsLRN=True):
        super(MaxPool, self).__init__(p_oParent, p_nModuleIndex, p_nPoolRect, p_nPoolStride, p_bIsPaddingPool=p_bIsPaddingPool, p_bIsLRN=p_bIsLRN)
                
        #........ |  Instance Attributes | ..............................................
        #................................................................................
    #------------------------------------------------------------------------------------
    def ModuleOnCreateTensors(self, p_bIsTrainingGraph):
        tInput, _ = self.Parent.GetPreviousModuleOutput(self.ModuleIndex)
        self.Input=tInput

        if self.IsPaddingPool:
            sPoolPaddingType=tcc.TF_ZERO_PADDING
        else:
            sPoolPaddingType=tcc.TF_NO_PADDING
        
        nMaxPoolShape=(1, self.PoolRect[0], self.PoolRect[1], 1)
        nStrides=[1, self.PoolStride, self.PoolStride, 1]
        tOut = tf.nn.max_pool(tInput, ksize=nMaxPoolShape, strides=nStrides, padding=sPoolPaddingType, name="maxpool_%02i" % self.ModuleIndex)

        self.Parent.SetTensor(self.ModuleIndex, tOut)

        self.Output = tOut

        print("Max Pool")
    #------------------------------------------------------------------------------------            
#==================================================================================================







#==================================================================================================    
class GlobalAvgPool(BasePoolModule):
    __verboseLevel=1   
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_nModuleIndex, p_bIsLRN=True):
        super(GlobalAvgPool, self).__init__(p_oParent, p_nModuleIndex, p_nPoolRect=None, p_nPoolStride=None, p_bIsPaddingPool=False, p_bIsLRN=False)
        #........ |  Instance Attributes | ..............................................
        #................................................................................
    #------------------------------------------------------------------------------------
    def ModuleOnCreateTensors(self, p_bIsTrainingGraph):
        tInput, _ = self.Parent.GetPreviousModuleOutput(self.ModuleIndex)
        self.Input=tInput

            

        sComment=""
        with tf.name_scope("GlobAvgPool"):
            # Global Average Pooling Layer
            #Example: _pool2 = tf.nn.avg_pool(_conv2, ksize=[1, 14, 14, 1], strides=[1, 14, 14, 1], padding='SAME')
            tOut=tf.reduce_mean(tInput, [1,2])

            
            

        self.Parent.SetTensor(self.ModuleIndex, tOut)

        if type(self).__verboseLevel >= 1:
            LogCNNLayer("[%02i] Global Average Pooling Layer" % self.ModuleIndex, tOut, tInput
                         ,p_sComment=sComment
                        )        
        #------------------------------------------------------------------------------------
#==================================================================================================


            
    
    
    

#==================================================================================================
class InceptionModule(BaseNeuralModule):
    __verboseLevel=1
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_nModuleIndex, p_nIntermediateFeatureDepth, p_nActivationType=tcc.NAF_RELU):
        
        #[[None, 65], [96, 128], [16, 32], [None,32]]
        nOutputFeatureDepth=0
        for nFeatureDepth in p_nIntermediateFeatureDepth:
            nOutputFeatureDepth+=nFeatureDepth[1]
        super(InceptionModule, self).__init__(p_oParent, p_nModuleIndex, nOutputFeatureDepth)
                
        #........ |  Instance Attributes | ..............................................
        self.ActivationFunctionType=p_nActivationType
        self.IntermediateFeatureDepth=p_nIntermediateFeatureDepth
        self.IsLRN=False
        #................................................................................
    #------------------------------------------------------------------------------------
    def __getKernelShape(self, p_nLateralPathNumber, p_bIsOutput, p_nInputFeatures):
        nFeaturePair=self.IntermediateFeatureDepth[p_nLateralPathNumber-1]
        if p_bIsOutput:
            if p_nLateralPathNumber==2:
                nKernelSize=3
            if p_nLateralPathNumber==3:
                nKernelSize=5
            else:
                nKernelSize=1
                
            if nFeaturePair[0] is None:
                # "1x1" or "pool proj" 
                nKernelShape=(nKernelSize, nKernelSize, p_nInputFeatures,  nFeaturePair[1])
            else:
                nKernelShape=(nKernelSize, nKernelSize, nFeaturePair[0],  nFeaturePair[1])
        else:
            nKernelShape=(1, 1, p_nInputFeatures, nFeaturePair[0])
            
        return nKernelShape
    #------------------------------------------------------------------------------------
    def ModuleOnCreateTensors(self, p_bIsTrainingGraph):
        tInput, nPreviousModuleFeatures = self.Parent.GetPreviousModuleOutput(self.ModuleIndex)
        self.Input=tInput
        sComment=""
        



        sScopeName="InceptionMod%02i" % self.ModuleIndex
        with tf.name_scope(sScopeName) as oScope:
            nStrides=[1, 1, 1, 1]

         
                
                

        #tcc.TF_NO_PADDING
        #tcc.TF_ZERO_PADDING            
            
         
            
            # .................................................
            # Lateral path 1/4 - [1x1~1]
            nLayerIndex=1
            nKernelShape = self.__getKernelShape(1, True, nPreviousModuleFeatures)
            tKernels, tBiases = self.Parent.ParamVars.ConvolutionalLayerParams(self.ModuleIndex*6 + nLayerIndex, nKernelShape, p_bIsWeightInitUniformScale=True, bIsBiasInitZero=True)
            tConv = tf.nn.bias_add( tf.nn.conv2d(self.Input, tKernels, strides=nStrides, padding=tcc.TF_NO_PADDING, name="c1x1_%02i.%i" % (self.ModuleIndex, nLayerIndex)), tBiases)
            tSynapticSum = tConv
            tFunc = mfunc.ActivationFunction(self.ActivationFunctionType, tSynapticSum)
            print("1x1", self.Input, tFunc, end=" ")
            tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + oScope    , tKernels)
            tf.add_to_collection(tf.GraphKeys.BIASES + "/" + oScope     , tBiases)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tSynapticSum)            
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tFunc)
            self.Parent.Layers.append(tFunc)   
            tLateralFunc1=tFunc
            # .................................................            
            # Lateral path 2/4 - [1x1~1] followed by [3x3~1]
            nLayerIndex=2
            nKernelShape = self.__getKernelShape(2, False, nPreviousModuleFeatures)
            tKernels, tBiases = self.Parent.ParamVars.ConvolutionalLayerParams(self.ModuleIndex*6 + nLayerIndex, nKernelShape, p_bIsWeightInitUniformScale=True, bIsBiasInitZero=True)
            tConv = tf.nn.bias_add( tf.nn.conv2d(self.Input, tKernels, strides=nStrides, padding=tcc.TF_NO_PADDING, name="c1x1_%02i.%i" % (self.ModuleIndex, nLayerIndex)), tBiases)
            tSynapticSum = tConv
            tFunc = mfunc.ActivationFunction(self.ActivationFunctionType, tSynapticSum)
            print("red3x3", self.Input, tFunc, end=" ")
            tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + oScope    , tKernels)
            tf.add_to_collection(tf.GraphKeys.BIASES + "/" + oScope     , tBiases)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tSynapticSum)            
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tFunc)
            self.Parent.Layers.append(tFunc)   
            tPreviousLayer=tFunc
                    
            nLayerIndex=3
            nKernelShape = self.__getKernelShape(2, True, nPreviousModuleFeatures)
            tKernels, tBiases = self.Parent.ParamVars.ConvolutionalLayerParams(self.ModuleIndex*6 + nLayerIndex, nKernelShape, p_bIsWeightInitUniformScale=True, bIsBiasInitZero=True)
            tConv = tf.nn.bias_add( tf.nn.conv2d(tPreviousLayer, tKernels, strides=nStrides, padding=tcc.TF_ZERO_PADDING, name="c3x3_%02i.%i" % (self.ModuleIndex, nLayerIndex)), tBiases)
            tSynapticSum = tConv
            tFunc = mfunc.ActivationFunction(self.ActivationFunctionType, tSynapticSum)
            print("3x3", self.Input, tFunc, end=" ")
            tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + oScope    , tKernels)
            tf.add_to_collection(tf.GraphKeys.BIASES + "/" + oScope     , tBiases)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tSynapticSum)            
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tFunc)
            self.Parent.Layers.append(tFunc)   
            tLateralFunc2=tFunc
            # .................................................   
            # Lateral path 3/4 - [1x1~1] followed by [5x5~1]
            nLayerIndex=4
            nKernelShape = self.__getKernelShape(3, False, nPreviousModuleFeatures)
            tKernels, tBiases = self.Parent.ParamVars.ConvolutionalLayerParams(self.ModuleIndex*6 + nLayerIndex, nKernelShape, p_bIsWeightInitUniformScale=True, bIsBiasInitZero=True)
            tConv = tf.nn.bias_add( tf.nn.conv2d(self.Input, tKernels, strides=nStrides, padding=tcc.TF_NO_PADDING, name="c1x1_%02i.%i" % (self.ModuleIndex, nLayerIndex)), tBiases)
            tSynapticSum = tConv
            tFunc = mfunc.ActivationFunction(self.ActivationFunctionType, tSynapticSum)
            print("red5x5", self.Input, tFunc,end=" ")
            tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + oScope    , tKernels)
            tf.add_to_collection(tf.GraphKeys.BIASES + "/" + oScope     , tBiases)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tSynapticSum)            
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tFunc)
            self.Parent.Layers.append(tFunc)   
            tPreviousLayer=tFunc
                        
            nLayerIndex=5
            nKernelShape = self.__getKernelShape(3, True, nPreviousModuleFeatures)
            tKernels, tBiases = self.Parent.ParamVars.ConvolutionalLayerParams(self.ModuleIndex*6 + nLayerIndex, nKernelShape, p_bIsWeightInitUniformScale=True, bIsBiasInitZero=True)
            tConv = tf.nn.bias_add( tf.nn.conv2d(tPreviousLayer, tKernels, strides=nStrides, padding=tcc.TF_ZERO_PADDING, name="c5x5_%02i.%i" % (self.ModuleIndex, nLayerIndex)), tBiases)
            tSynapticSum = tConv
            tFunc = mfunc.ActivationFunction(self.ActivationFunctionType, tSynapticSum)
            print("5x5", self.Input, tFunc,end=" ")
            tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + oScope    , tKernels)
            tf.add_to_collection(tf.GraphKeys.BIASES + "/" + oScope     , tBiases)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tSynapticSum)            
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tFunc)
            self.Parent.Layers.append(tFunc)   
            tLateralFunc3=tFunc
            # ................................................. 
            # Lateral path 4/4 - MaxPool[3x3] followed by [1x1~1]
            nMaxPoolShape=(1, 3, 3, 1)
            nStrides=[1, 1, 1, 1]
            tPool = tf.nn.max_pool(self.Input, ksize=nMaxPoolShape, strides=nStrides, padding=tcc.TF_ZERO_PADDING, name="maxpool_%02i" % self.ModuleIndex)
            self.Parent.Layers.append(tPool)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tPool)
            tPreviousLayer=tPool
           
            nLayerIndex=6
            nKernelShape = self.__getKernelShape(4, True, nPreviousModuleFeatures)            
            tKernels, tBiases = self.Parent.ParamVars.ConvolutionalLayerParams(self.ModuleIndex*6 + nLayerIndex, nKernelShape, p_bIsWeightInitUniformScale=True, bIsBiasInitZero=True)
            tConv = tf.nn.bias_add( tf.nn.conv2d(tPreviousLayer, tKernels, strides=nStrides, padding=tcc.TF_NO_PADDING, name="c1x1_%02i.%i" % (self.ModuleIndex, nLayerIndex)), tBiases)
            tSynapticSum = tConv
            tFunc = mfunc.ActivationFunction(self.ActivationFunctionType, tSynapticSum)
            print("pp1x1", self.Input, tFunc,end=" ")                  
            tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + oScope    , tKernels)
            tf.add_to_collection(tf.GraphKeys.BIASES + "/" + oScope     , tBiases)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tSynapticSum)            
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tFunc)
            self.Parent.Layers.append(tFunc)                                                   
            tLateralFunc4=tFunc   
            # .................................................
            tConcatenation=tf.concat([tLateralFunc1, tLateralFunc2, tLateralFunc3, tLateralFunc4], 3, name="LGNConcat")   

            #sComment = sComment + ActivationFunctionDescr(self.ActivationFunctionType)
  
            # It will add an LRN even when the max pooling layer is disabled, if the pooling stride is not zero
            if self.IsLRN:
                tOut = tf.nn.local_response_normalization(tConcatenation, name="lrn_%02i" % self.ModuleIndex)
                sComment = sComment + ", lrn"
                tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tOut)
            else:
                tOut=tConcatenation                
            
            self.Parent.Layers.append(tOut)

        self.Parent.SetTensor(self.ModuleIndex, tOut)
        self.Output = tOut
        print(" ")

        #TEMP: Invalid display
        if type(self).__verboseLevel >= 1:
            LogCNNLayerAndPool("[%02i] Inception Module" % self.ModuleIndex
                         ,self.Input, tFunc, self.Output 
                         ,p_tConvFilter=tLateralFunc1, p_nConvStride=1, p_sConvType=tcc.TF_NO_PADDING
                         ,p_nPoolRect=None, p_nPoolStride=2, p_sPoolType=tcc.TF_ZERO_PADDING 
                         ,p_sComment=sComment
                        )
    #------------------------------------------------------------------------------------    
#==================================================================================================













#==================================================================================================
class FullyConnectedWithDropout(BaseNeuralModule):
    __verboseLevel=1
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_nModuleIndex, p_nOutputFeatureDepth, p_nDropoutKeepProbability, p_nActivationType=tcc.NAF_TANH):
        super(FullyConnectedWithDropout, self).__init__(p_oParent, p_nModuleIndex, p_nOutputFeatureDepth)
        #........ |  Instance Attributes | ..............................................
        self.DropoutKeepProbability=p_nDropoutKeepProbability
        self.ActivationFunctionType=p_nActivationType
        #................................................................................
    #------------------------------------------------------------------------------------
    def ModuleOnCreateTensors(self, p_bIsTrainingGraph):
        tInput, nPreviousModuleFeatures = self.Parent.GetPreviousModuleOutput(self.ModuleIndex, p_bToFullyConnected=True)
        self.Input=tInput
        sComment=""
        
        nDefItem = ArchitectureDefItem(ArchitectureDefKind.FULLY_CONNECTED, self.ModuleIndex, self.Input)
        
        # Declared here to put dropout enable/disable flags outside of the scope
        if self.DropoutKeepProbability is not None:
            tIsTraining=self.Parent.GetIsTrainingFlag(p_bIsTrainingGraph)
            
        sScopeName="FCModule%02i" % self.ModuleIndex
        with tf.name_scope(sScopeName) as oScope:
            tInput = tf.reshape(self.Input, [-1, nPreviousModuleFeatures])
            nWeightShape=[nPreviousModuleFeatures, self.OutputFeatureDepth]
            tWeights, tBiases = self.Parent.ParamVars.FullyConnectedParams(self.ModuleIndex, nWeightShape, p_bAreZeroCenteredMeanWeights=True, p_bIsBiasInitZero=True)
            
            # Weighted Sum
            tSum = tf.nn.bias_add(tf.matmul(tInput, tWeights), tBiases)
            # Activation
            tFunc = mfunc.ActivationFunction(self.ActivationFunctionType, tSum)
            nDefItem.ActivationFunction = self.ActivationFunctionType
            
            tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + oScope     , tWeights)
            tf.add_to_collection(tf.GraphKeys.BIASES + "/" + oScope      , tBiases)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS  , tFunc)
            self.Parent.Layers.append(tFunc)

             
            sComment = sComment + ActivationFunctionDescr(self.ActivationFunctionType)


            if not p_bIsTrainingGraph:
                tOut = tFunc
            else:
                # Dropout is conditionally used, only for training
                if self.DropoutKeepProbability is None:
                    tOut = tFunc
                elif self.DropoutKeepProbability == 1.0:
                    tOut = tFunc
                else:
                    sComment = sComment + ", dropout:%.2f" % self.DropoutKeepProbability
                    tOut = tf.cond(tIsTraining
                                    , lambda: tf.nn.dropout(tFunc, tf.constant(self.DropoutKeepProbability, dtype=tf.float32), name="dropout_%02i" % self.ModuleIndex)
                                    , lambda: tFunc
                                    )
                    nDefItem.SetDropout(self.DropoutKeepProbability)

        self.Parent.SetTensor(self.ModuleIndex, tFunc) #TEMP: [2017-12-05] Check this before release 
        self.Output=tOut

        nDefItem.SetOutput(self.Output)
        self.Parent.Architecture.Add(nDefItem)
        
        if type(self).__verboseLevel >= 2:
            LogCNNLayer("[%02i] Fully Connected Module" % self.ModuleIndex, tFunc, tInput
                         ,p_sComment=sComment
                        )        
        #TODO: Better Output
        return tOut
        #------------------------------------------------------------------------------------
#==================================================================================================






#==================================================================================================
class LinearWithDropout(FullyConnectedWithDropout):
    __verboseLevel=1
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_nModuleIndex, p_nOutputFeatureDepth, p_nDropoutKeepProbability, p_bIsTraining):
        super(LinearWithDropout, self).__init__(p_oParent, p_nModuleIndex, p_nOutputFeatureDepth, p_nDropoutKeepProbability, p_bIsTraining, p_nActivationType=tcc.NAF_LINEAR)
        #........ |  Instance Attributes | ..............................................
        #................................................................................
#==================================================================================================        
        






#==================================================================================================
class SoftMax(BaseNeuralModule):
    __verboseLevel=1
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_nModuleIndex, p_nTargetClassCount):
        super(SoftMax, self).__init__(p_oParent, p_nModuleIndex, p_nOutputFeatureDepth=p_nTargetClassCount)
        #........ |  Instance Attributes | ..............................................
        self.TargetClassCount=p_nTargetClassCount
        assert self.TargetClassCount is not None, "Class count is not provided"
        #................................................................................
    #------------------------------------------------------------------------------------
    def ModuleOnCreateTensors(self, p_bIsTrainingGraph):
        #SoftMax(self, p_nModuleIndex, p_nTargets, p_bIsTrainingFunction, p_nInputTensor=None):
        tInput, nPreviousModuleFeatures = self.Parent.GetPreviousModuleOutput(self.ModuleIndex, p_bToFullyConnected=True)
        self.Input=tInput
        
        nDefItem = ArchitectureDefItem(ArchitectureDefKind.SOFTMAX, self.ModuleIndex, self.Input)
        

        sScopeName="SoftMax%02i" % self.ModuleIndex
        with tf.name_scope(sScopeName) as oScope:
            nWeightShape=[nPreviousModuleFeatures, self.TargetClassCount]
            tWeights, tBiases = self.Parent.ParamVars.FullyConnectedParams(self.ModuleIndex, nWeightShape, p_bAreZeroCenteredMeanWeights=True, p_bIsBiasInitZero=True)
            
            tSum = tf.nn.bias_add(tf.matmul(self.Input, tWeights), tBiases)
            
            # For using categorical cross entrory training loss soft max operation must not be applied before feeding output to softmax_cross_entropy_with_logits tensor.
            if p_bIsTrainingGraph:
                self.Parent.TrainLossInput=tSum
            else:
                self.Parent.ValLossInput=tSum

            # The softmax is the output of the neural net for recalling
            tFunc = tf.nn.softmax(tSum, name="softmax_%02i" % self.ModuleIndex)
            nDefItem.ActivationFunction = tcc.NAF_SOFTMAX
            
            tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + oScope, tWeights)
            tf.add_to_collection(tf.GraphKeys.BIASES + "/" + oScope, tBiases)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tSum)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tFunc)
            self.Parent.Layers.append(tSum)
            self.Parent.Layers.append(tFunc)
        
        tOut=tFunc
        
        self.Parent.SetTensor(self.ModuleIndex, tOut)
        self.Output=tOut
        
        nDefItem.SetOutput(self.Output)
        self.Parent.Architecture.Add(nDefItem)
        
        if type(self).__verboseLevel >= 2:
            LogCNNLayer("[%02i] Softmax" % self.ModuleIndex, tFunc, tInput
                         ,p_sComment="f(x)=softmax(s)"
                        )   
                    
        return self.Output          
#==================================================================================================















    
#==================================================================================================
class DCNN(BaseNN):
    __verboseLevel=1

    #------------------------------------------------------------------------------------
    def __init__(self, p_sExperimentName, p_tInputQueue, p_oExperiment=None, p_oSettings=None):
        #........ |  Instance Attributes | ..............................................
        #................................................................................
        super(DCNN, self).__init__(p_sExperimentName, p_tInputQueue, p_oExperiment, p_oSettings)
    #------------------------------------------------------------------------------------
    def ConstructDeepCNN(self, p_nInputLayer, p_nFeatures, p_nKernelRects, p_nConvStrides, p_nPoolDims, p_nPoolStrides):
        tPrevious = p_nInputLayer
        for nIndex, oKernelRect in enumerate(p_nKernelRects):
            tPrevious = self.ConvolutionalMaxPoolLRN(nIndex + 1, p_nFeatures[nIndex]
                                                     , p_nKernelRect=oKernelRect, p_nConvStride=p_nConvStrides[nIndex]
                                                     , p_nPoolDim=p_nPoolDims[nIndex], p_nPoolStride=p_nPoolStrides[nIndex]
                                                     , p_bIsSamePool=True, p_nInputTensor=tPrevious)
        
        nFullyConnectedMax = len(p_nFeatures) - len(p_nKernelRects) - 1
        for nAddIndex in range(0, nFullyConnectedMax):
            tPrevious=self.FullyConnectedDropout(nIndex + nAddIndex + 2, p_nFeatures[nIndex + nAddIndex + 1], p_nDropoutKeepProbability=0.5, p_nFunction=tcc.NAF_TANH, p_nInputTensor=tPrevious )
        
        nIndex = len(p_nKernelRects) + 2
        tOutputLayer = self.SoftMax(nIndex + 1, p_nFeatures[nIndex], p_nInputTensor=tPrevious)

        return tOutputLayer    
    #------------------------------------------------------------------------------------
    def ConvolutionalMaxPoolLRN(self, p_nModuleIndex, p_nFeatureDepth, p_nKernelRect=[5,5], p_nConvStride=1, p_nPoolDim=2, p_nPoolStride=2, p_nKernelSize=None, p_nPoolSize=None, p_bIsSameConv=True, p_bIsSamePool=False, p_bIsBatchNormalization=False, p_bIsUsingLRN=True, p_nFunction=tcc.NAF_RELU, p_nInputTensor=None ):
        if p_nKernelSize is not None:
            p_nKernelRect = [ p_nKernelSize[0], p_nKernelSize[1] ]
            p_nConvStride = p_nKernelSize[2]
            
        if p_nPoolSize is not None:
            p_nPoolDim      = p_nPoolSize[0]
            p_nPoolStride   = p_nPoolSize[1]
            
        tPreviousTensor, nPreviousModuleFeatures = self.GetPreviousModuleOf(p_nModuleIndex, p_nInputTensor)
        
        nKernelShape=(p_nKernelRect[0], p_nKernelRect[1], nPreviousModuleFeatures, p_nFeatureDepth)
        nMaxPoolShape=(1, p_nPoolDim, p_nPoolDim, 1)
        
        if p_bIsSameConv:
            sConvPaddingType=tcc.TF_ZERO_PADDING 
        else:
            sConvPaddingType=tcc.TF_NO_PADDING
            
        if p_bIsSamePool:
            sPoolPaddingType=tcc.TF_ZERO_PADDING
        else:
            sPoolPaddingType=tcc.TF_NO_PADDING

        sScopeName="ConvMod%02i" % p_nModuleIndex
        with tf.name_scope(sScopeName) as oScope:
            tKernels, tBiases = self.ParamVars.ConvolutionalLayerParams(p_nModuleIndex, nKernelShape, p_bIsWeightInitUniformScale=True, bIsBiasInitZero=True)
            
            # Convolution
            tConv = tf.nn.bias_add( tf.nn.conv2d(p_nInputTensor, tKernels, strides=[1, p_nConvStride, p_nConvStride, 1]
                                                 ,padding=sConvPaddingType, name="conv2d_%02i" % p_nModuleIndex)
                                   ,tBiases)
            
            
            # Batch normalization is used only for training
            if p_bIsBatchNormalization:
                tSynapticSum = mfunc.BatchNormalization(tConv, p_nFeatureDepth, self.IsTrainingProcessTensor(), sScopeName, p_nModuleIndex)
                sComment=ActivationFunctionDescr(p_nFunction), " batch norm"
            else:
                tSynapticSum=tConv
                sComment=ActivationFunctionDescr(p_nFunction)
            
            # Activation
            tFunc = mfunc.ActivationFunction(p_nFunction, tSynapticSum)
            
            tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + oScope    , tKernels)
            tf.add_to_collection(tf.GraphKeys.BIASES + "/" + oScope     , tBiases)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tSynapticSum)            
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tFunc)
            self.Layers.append(tFunc)            
            
            # Disables the max pooling layer when pool dim is zero
            if p_nPoolDim > 0:
                tPool=tf.nn.max_pool(tFunc, ksize=nMaxPoolShape, strides=[1, p_nPoolStride, p_nPoolStride, 1], padding=sPoolPaddingType, name="maxpool_%02i" % p_nModuleIndex)
                
                self.Layers.append(tPool)
                tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tPool)
            else:
                tPool=tFunc
        

            # It will add an LRN even when the max pooling layer is disabled, if the pooling stride is not zero
            if (p_nPoolStride > 0) and p_bIsUsingLRN:
                tOut=tf.nn.local_response_normalization(tPool, name="lrn_%02i" % p_nModuleIndex)
                sComment=sComment +", lrn"
                tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tOut)
            else:
                tOut=tPool                
            
            self.Layers.append(tOut)
    


        self.SetModule(p_nModuleIndex, tOut)

        if type(self).__verboseLevel >= 1:
            LogCNNLayerAndPool("[%02i] Convolutional Module" % p_nModuleIndex
                         ,tPreviousTensor, tFunc, tOut 
                         ,p_tConvFilter=tKernels, p_nConvStride=p_nConvStride, p_sConvType=sConvPaddingType
                         ,p_nPoolDim=p_nPoolDim, p_nPoolStride=p_nPoolStride, p_sPoolType=sPoolPaddingType 
                         ,p_sComment=sComment
                        )
                        
        return tOut

    #------------------------------------------------------------------------------------
    def GlobalAveragePoolingDropOut(self, p_nModuleIndex, p_bIsTrainingFunction, p_nDropoutKeepProbability=0.5, p_nInputTensor=None):
        tPreviousTensor, nPreviousModuleFeatures = self.GetPreviousModuleOf(p_nModuleIndex, p_nInputTensor) 
        
        # Declared here to put flags outside of the scope
        if p_nDropoutKeepProbability != 0:
            tIsTraining=self.IsTrainingFlag(p_bIsTrainingFunction)
            
        tInput=tPreviousTensor
        #nGlobalDim = tInput.get_shape()[0]

        sComment=""
        with tf.name_scope("GlobAvgPool"):
            # Global Average Pooling Layer
            #Example: _pool2 = tf.nn.avg_pool(_conv2, ksize=[1, 14, 14, 1], strides=[1, 14, 14, 1], padding='SAME')
            tPool=tf.reduce_mean(tInput, [1,2])

            self.Layers.append(tPool)
            
            # Dropout is conditionally used, only for training
            if p_nDropoutKeepProbability is None:
                y_do = tPool
            else:
                sComment = " dropout:%.2f" % p_nDropoutKeepProbability
                y_do = tf.cond(tIsTraining
                                , lambda: tf.nn.dropout(tPool, tf.constant(p_nDropoutKeepProbability, dtype=tf.float32), name="dropout_%02i" % p_nModuleIndex)
                                , lambda: tPool
                                )
                
                
            # Flatten for use as softmax input
            tOutput = tf.reshape(y_do, [-1, nPreviousModuleFeatures])
            
        self.SetModule(p_nModuleIndex, tPool)
        
        
        

        if type(self).__verboseLevel >= 1:
            LogCNNLayer("[%02i] Global Average Pooling Layer" % p_nModuleIndex, tOutput, tInput
                         ,p_sComment=sComment
                        )        
    
        return tOutput
    
            
        
        
    #------------------------------------------------------------------------------------
    def FullyConnectedDropout(self, p_nModuleIndex, p_nNeurons, p_bIsTrainingFunction, p_nDropoutKeepProbability=0.5, p_nFunction=tcc.NAF_TANH, p_nInputTensor=None):
        
        tPreviousTensor, nPreviousModuleFeatures = self.GetPreviousModuleOf(p_nModuleIndex, p_nInputTensor, p_bToFullyConnected=True) 
        
        nWeightShape=[nPreviousModuleFeatures, p_nNeurons]
        
        # Declared here to put flags outside of the scope
        if p_nDropoutKeepProbability is not None:
            tIsTraining=self.IsTrainingFlag(p_bIsTrainingFunction)
            
        sScopeName="FCModule%02i" % p_nModuleIndex
        with tf.name_scope(sScopeName) as oScope:
            tInput = tf.reshape(tPreviousTensor, [-1, nPreviousModuleFeatures])
            tWeights, tBiases = self.ParamVars.FullyConnectedParams(p_nModuleIndex, nWeightShape, p_bAreZeroCenteredMeanWeights=True, p_bIsBiasInitZero=True)
            
            # Weighted Sum
            tSum = tf.nn.bias_add(tf.matmul(tInput, tWeights), tBiases)
            # Activation
            tFunc = mfunc.ActivationFunction(p_nFunction, tSum)
            
            tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + oScope     , tWeights)
            tf.add_to_collection(tf.GraphKeys.BIASES + "/" + oScope      , tBiases)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS  , tFunc)
            self.Layers.append(tFunc)

             
            sComment=ActivationFunctionDescr(p_nFunction)


            # Dropout is conditionally used, only for training
            if p_nDropoutKeepProbability is None:
                y_do = tFunc
            elif p_nDropoutKeepProbability == 1.0:
                y_do = tFunc
            else:
                sComment = sComment + ", dropout:%.2f" % p_nDropoutKeepProbability
                y_do = tf.cond(tIsTraining
                                , lambda: tf.nn.dropout(tFunc, tf.constant(p_nDropoutKeepProbability, dtype=tf.float32), name="dropout_%02i" % p_nModuleIndex)
                                , lambda: tFunc
                                )
            
        self.SetModule(p_nModuleIndex, tFunc)
        
        
        
        
        if type(self).__verboseLevel >= 1:
            LogCNNLayer("[%02i] Fully Connected Module" % p_nModuleIndex, tFunc, tInput
                         ,p_sComment=sComment
                        )        
    
        return y_do
    
    #------------------------------------------------------------------------------------
    def SoftMax(self, p_nModuleIndex, p_nTargets, p_bIsTrainingFunction, p_nInputTensor=None):

        tPreviousTensor, nPreviousModuleFeatures = self.GetPreviousModuleOf(p_nModuleIndex, p_nInputTensor, p_bToFullyConnected=True)
        
        nWeightShape=[nPreviousModuleFeatures, p_nTargets]
        
        sScopeName="SoftMax%02i" % p_nModuleIndex
        with tf.name_scope(sScopeName) as oScope:
            tWeights, tBiases = self.ParamVars.FullyConnectedParams(p_nModuleIndex, nWeightShape, p_bAreZeroCenteredMeanWeights=True, p_bIsBiasInitZero=True)
            
            tSum = tf.nn.bias_add(tf.matmul(tPreviousTensor, tWeights), tBiases)
            
            # For training with categorical cross entrory soft max operation must not be applied, before feeding the softmax_cross_entropy_with_logits tensor
            
            if p_bIsTrainingFunction:
                self.TrainLossInput=tSum
            else:
                self.ValLossInput=tSum

            # The softmax is the output of the neural net for recalling
            tFunc = tf.nn.softmax(tSum, name="softmax_%02i" % p_nModuleIndex)
            
            tf.add_to_collection(tf.GraphKeys.WEIGHTS + '/' + oScope, tWeights)
            tf.add_to_collection(tf.GraphKeys.BIASES + "/" + oScope, tBiases)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tSum)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tFunc)
            self.Layers.append(tSum)
            self.Layers.append(tFunc)
        
        self.SetModule(p_nModuleIndex, tFunc)
        
        if type(self).__verboseLevel >= 1:
            LogCNNLayer("[%02i] Softmax" % p_nModuleIndex, tFunc, tPreviousTensor
                         ,p_sComment="f(x)=softmax(s)"
                        )   
                    
        return tFunc          
    #------------------------------------------------------------------------------------

#==================================================================================================















#------------------------------------------------------------------------------------
def LogCNNLayer(p_sTitle, p_tLayerTensor, p_tInputTensor=None, p_tConvFilter=None, p_nConvStride=1, p_sConvType="", p_nPoolDim=None, p_nPoolStride=1, p_sPoolType="", p_sComment="", p_nIdentLevel=1):
    TF_ZERO_PADDING = "SAME"
    TF_NO_PADDING   = "VALID"
        
    
    """ Logs the structure of a tensorflow convolutional layer with or without pooling """
    sText = "  " * p_nIdentLevel + "|__  " + "{:<50} ".format(p_sTitle)
        
    if p_tLayerTensor is not None: 
        if p_sConvType == TF_ZERO_PADDING:
            p_sConvType="zero"
        elif p_sConvType == TF_NO_PADDING:
            p_sConvType="no pad"
        
        if p_sPoolType == TF_ZERO_PADDING:
            p_sPoolType="zero"        
        elif p_sPoolType == TF_NO_PADDING:
            p_sPoolType="no pad"
    
        
        
        if p_tInputTensor is not None:
            if p_tInputTensor.get_shape().ndims==4:
                nInputSampleCount=p_tInputTensor.get_shape()[0].value
                if nInputSampleCount is None:
                    nInputSampleCount=-1
    
                nInputWindowSizeX=p_tInputTensor.get_shape()[1].value
                nInputWindowSizeY=p_tInputTensor.get_shape()[2].value
                nInputFeatures=p_tInputTensor.get_shape()[3].value
                sText += "in [%ix%i|%i]\t" % (nInputWindowSizeX, nInputWindowSizeY, nInputFeatures)
            elif p_tInputTensor.get_shape().ndims==2:
                nInputSampleCount=p_tInputTensor.get_shape()[0].value
                if nInputSampleCount is None:
                    nInputSampleCount=-1
    
                nInputFeatures=p_tInputTensor.get_shape()[1].value
                sText += "in [%i]\t" % (nInputFeatures)
        
        if p_tConvFilter is not None:
            nConvWindowSizeX=p_tConvFilter.get_shape()[0].value
            nConvWindowSizeY=p_tConvFilter.get_shape()[1].value
            nConvInFeatures=p_tConvFilter.get_shape()[2].value
            nConvOutFeatures=p_tConvFilter.get_shape()[3].value
            sText += "conv %s [%ix%i~%i|%i->%i]\t" % (p_sConvType, nConvWindowSizeX, nConvWindowSizeY, p_nConvStride, nConvInFeatures, nConvOutFeatures)
            
            
        if (p_nPoolDim is not None) and (p_nPoolDim != 0):
            sText += "pool %s [%ix%i~%i]\t" % (p_sPoolType, p_nPoolDim, p_nPoolDim, p_nPoolStride)
        
        if p_tLayerTensor.get_shape().ndims==4:
            nLayerSampleCount=p_tLayerTensor.get_shape()[0].value
            if nLayerSampleCount is None:
                nLayerSampleCount=-1
            nLayerWindowSizeX=p_tLayerTensor.get_shape()[1].value
            nLayerWindowSizeY=p_tLayerTensor.get_shape()[2].value
            nLayerFeatures=p_tLayerTensor.get_shape()[3].value
            
            if p_tInputTensor is not None:
                assert nLayerSampleCount==nInputSampleCount            
                sText += "=> [%ix%i|%i]\t" % (nLayerWindowSizeX, nLayerWindowSizeY, nLayerFeatures)
            else:
                sText += "[%ix%i|%i] samples:%i" % (nLayerWindowSizeX, nLayerWindowSizeY, nLayerFeatures, nLayerSampleCount)
        elif p_tLayerTensor.get_shape().ndims==2:
            nLayerSampleCount=p_tLayerTensor.get_shape()[0].value
            if nLayerSampleCount is None:
                nLayerSampleCount=-1
    
            nLayerFeatures=p_tLayerTensor.get_shape()[1].value
                    
            if p_tInputTensor is not None:
                assert nLayerSampleCount==nInputSampleCount, "Sample count in layer not equal, sample count in inputs"
                sText += "=> [%i]\t" % (nLayerFeatures)
            else:
                sText += "[%i] samples:%i" % (nLayerFeatures, nLayerSampleCount)
        
    
    sText += p_sComment
    
    print(sText)
#     #TODO: Log this
#     oActiveLog = GetActiveLog()
#     if oActiveLog is not None:
#         oActiveLog.WriteLine(sText)
#     else:
#         print(sText)
#------------------------------------------------------------------------------------
def LogCNNLayerAndPool(  p_sTitle, p_tInputTensor, p_tPoolInputTensor, p_tOutputTensor
                       , p_tConvFilter=None, p_nConvStride=1, p_sConvType=""
                       , p_nPoolRect=None, p_nPoolDim=None, p_nPoolStride=1, p_sPoolType=""
                       , p_sComment="", p_nIdentLevel=1):  
    
    TF_ZERO_PADDING = "SAME"
    TF_NO_PADDING   = "VALID"
        
    
    """ Logs the structure of a tensorflow convolutional layer with or without pooling """
    sText = "  " * p_nIdentLevel + "|__  " + "{:<50} ".format(p_sTitle)
        
    if p_tOutputTensor is not None: 
        if p_sConvType == TF_ZERO_PADDING:
            p_sConvType="zero"
        elif p_sConvType == TF_NO_PADDING:
            p_sConvType="no pad"
        
        if p_sPoolType == TF_ZERO_PADDING:
            p_sPoolType="zero"        
        elif p_sPoolType == TF_NO_PADDING:
            p_sPoolType="no pad"
    
        
        
        if p_tInputTensor is not None:
            if p_tInputTensor.get_shape().ndims==4:
                nInputSampleCount=p_tInputTensor.get_shape()[0].value
                if nInputSampleCount is None:
                    nInputSampleCount=-1
    
                nInputWindowSizeX=p_tInputTensor.get_shape()[1].value
                nInputWindowSizeY=p_tInputTensor.get_shape()[2].value
                nInputFeatures=p_tInputTensor.get_shape()[3].value
                sText += "in [%ix%i|%i] " % (nInputWindowSizeX, nInputWindowSizeY, nInputFeatures)
            elif p_tInputTensor.get_shape().ndims==2:
                nInputSampleCount=p_tInputTensor.get_shape()[0].value
                if nInputSampleCount is None:
                    nInputSampleCount=-1
    
                nInputFeatures=p_tInputTensor.get_shape()[1].value
                sText += "in [%i] " % (nInputFeatures)

        
        if p_tConvFilter is not None:
            nConvWindowSizeX=p_tConvFilter.get_shape()[0].value
            nConvWindowSizeY=p_tConvFilter.get_shape()[1].value
            nConvInFeatures=p_tConvFilter.get_shape()[2].value
            nConvOutFeatures=p_tConvFilter.get_shape()[3].value
            sText += "| conv %s [%ix%i~%i|%i->%i] " % (p_sConvType, nConvWindowSizeX, nConvWindowSizeY, p_nConvStride, nConvInFeatures, nConvOutFeatures)
        
        if (p_nPoolDim is None) and (p_nPoolRect is not None):
            p_nPoolDim = p_nPoolRect[0] #Supports only squared pools
        
        
        if (p_nPoolDim is not None) and (p_nPoolDim != 0):
            if p_tPoolInputTensor.get_shape().ndims==4:
                nInputWindowSizeX=p_tPoolInputTensor.get_shape()[1].value
                nInputWindowSizeY=p_tPoolInputTensor.get_shape()[2].value   
                sText += "=> [%ix%i|%i] | pool %s [%ix%i~%i] " % (nInputWindowSizeX, nInputWindowSizeY, nConvOutFeatures, p_sPoolType, p_nPoolDim, p_nPoolDim, p_nPoolStride)             
            else:
                sText += "pool %s [%ix%i~%i] " % (p_sPoolType, p_nPoolDim, p_nPoolDim, p_nPoolStride)
        
        if p_tOutputTensor.get_shape().ndims==4:
            nLayerSampleCount=p_tOutputTensor.get_shape()[0].value
            if nLayerSampleCount is None:
                nLayerSampleCount=-1
            nLayerWindowSizeX=p_tOutputTensor.get_shape()[1].value
            nLayerWindowSizeY=p_tOutputTensor.get_shape()[2].value
            nLayerFeatures=p_tOutputTensor.get_shape()[3].value
            
            if p_tInputTensor is not None:
                assert nLayerSampleCount==nInputSampleCount            
                sText += "=> [%ix%i|%i] " % (nLayerWindowSizeX, nLayerWindowSizeY, nLayerFeatures)
            else:
                sText += "[%ix%i|%i] samples:%i " % (nLayerWindowSizeX, nLayerWindowSizeY, nLayerFeatures, nLayerSampleCount)
        elif p_tOutputTensor.get_shape().ndims==2:
            nLayerSampleCount=p_tOutputTensor.get_shape()[0].value
            if nLayerSampleCount is None:
                nLayerSampleCount=-1
    
            nLayerFeatures=p_tOutputTensor.get_shape()[1].value
                    
            if p_tInputTensor is not None:
                assert nLayerSampleCount==nInputSampleCount, "Sample count in layer not equal, sample count in inputs"
                sText += "=> [%i] " % (nLayerFeatures)
            else:
                sText += "[%i] samples:%i " % (nLayerFeatures, nLayerSampleCount)
        
    
    sText += p_sComment
    
    
    #TODO: Log this
#     oActiveLog = GetActiveLog()
#     if oActiveLog is not None:
#         oActiveLog.WriteLine(sText)
#     else:
    print(sText)        
#------------------------------------------------------------------------------------
def GetValue(p_dDict, p_sKey):
    if p_sKey in p_dDict:
        oValue = p_dDict[p_sKey]
        print("  | %s:" % p_sKey.ljust(32), oValue )
        return oValue
    else:
        return None