# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        NEURAL NETWORK HYPERPARAMETERS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import numpy as np
import json
import TALOS.Constants as tcc
from TALOS.Utils import GetValue, GetValueAsBool, MODEL_DEFINITION_FOLDER
from TALOS.Core import JSONConfig, ERLString
from TALOS.FileSystem import Storage, BaseFolders

     

#==================================================================================================
class NNMiniBatches():
    __verboseLevel=1
    #------------------------------------------------------------------------------------
    def __init__(self): 
        #........ |  Instance Attributes | ..............................................
        self.TrainingSamples=None
        self.ValidationSamples=None
        self.TestSamples=None
        
        self.TrainingSamplesPerBatch=None
        self.ValidationSamplesPerBatch=None
        self.TestSamplesPerBatch=None
        
        self.TrainingBatches=None
        self.ValidationBatches=None
        self.TestBatches=None
        
        self.TrainingSamplesPerFrag=None
        self.ValidationSamplesPerFrag=None
        self.TestSamplesPerFrag=None
        #................................................................................
    
    #------------------------------------------------------------------------------------
    def SetSamplesPerBatch(self, p_nTrainingSamplesPerBatch=None, p_nValidationSamplesPerBatch=None, p_nTestSamplesPerBatch=None):
        
        if p_nTrainingSamplesPerBatch is not None:
            self.TrainingSamplesPerBatch=p_nTrainingSamplesPerBatch
            
        if p_nValidationSamplesPerBatch is not None:
            self.ValidationSamplesPerBatch=p_nValidationSamplesPerBatch
            
        if p_nTestSamplesPerBatch is not None:
            self.TestSamplesPerBatch=p_nTestSamplesPerBatch
    #------------------------------------------------------------------------------------
    def SetSamplesPerBatchFrom(self, p_oSourceMinibatches):
        self.TrainingSamplesPerBatch=p_oSourceMinibatches.TrainingSamplesPerBatch
        self.ValidationSamplesPerBatch=p_oSourceMinibatches.ValidationSamplesPerBatch
        self.TestSamplesPerBatch=p_oSourceMinibatches.TestSamplesPerBatch
    #------------------------------------------------------------------------------------
    def SetDataCounts(self, p_oTrainingDataCount=None, p_oValidationDataCount=None, p_oTestDataCount=None):
        
        # If training data are supplied
        if p_oTrainingDataCount is not None:
            self.TrainingSamples = p_oTrainingDataCount
            self.TrainingBatches=int(np.ceil(self.TrainingSamples / self.TrainingSamplesPerBatch))
        else:
            self.TrainingSamples = 0
            self.TrainingBatches = 0
        
        # If validation data are supplied
        if p_oValidationDataCount is not None:
            self.ValidationSamples = p_oValidationDataCount
            self.ValidationBatches=int(np.ceil(self.ValidationSamples / self.ValidationSamplesPerBatch))
        else:
            self.ValidationSamples = 0
            self.ValidationBatches = 0
        
        # If test data are supplied
        if p_oTestDataCount is not None:
            self.TestSamples = p_oTestDataCount
            self.TestBatches=int(np.ceil(self.TestSamples / self.TestSamplesPerBatch))
        else:
            self.TestSamples = 0
            self.TestBatches = 0
        
            
        if NNMiniBatches.__verboseLevel==1:
            print("  |__  SampleCount={Training:%i, Validation:%i, Testing:%i} " % 
                  ( 
                    self.TrainingSamples, self.ValidationSamples, self.TestSamples
                   )
                  )
            print("  |__  Minibatches | BatchCount={Training:%i, Validation:%i, Testing:%i} SamplesPerBatch={Training:%i, Validation:%i, Testing:%i} " % 
                  ( self.TrainingBatches, self.ValidationBatches, self.TestBatches
                   ,self.TrainingSamplesPerBatch, self.ValidationSamplesPerBatch, self.TestSamplesPerBatch
                   )
                  )
    #------------------------------------------------------------------------------------
    def SetFragments(self, p_oMaxFragments=1):
        nFragmentsCount=p_oMaxFragments-1
        
        if nFragmentsCount != 0:
            self.TrainingSamplesPerFrag      = int(self.TrainingBatches / nFragmentsCount) * self.TrainingSamplesPerBatch
            self.ValidationSamplesPerFrag    = int(self.ValidationBatches / nFragmentsCount) * self.ValidationSamplesPerBatch
            self.TestSamplesPerFrag          = int(self.TestBatches / nFragmentsCount) * self.TestSamplesPerBatch
        else:
            self.TrainingSamplesPerFrag = self.TrainingSamples
            self.ValidationSamplesPerFrag = self.ValidationSamples
            self.TestSamplesPerFrag = self.TestSamples
    
        if NNMiniBatches.__verboseLevel==1:
            print("  |__  SamplesPerFrag={Training:%i, Validation:%i, Testing:%i} " % 
                  ( self.TrainingSamplesPerFrag, self.ValidationSamplesPerFrag, self.TestSamplesPerFrag
                   )
                  )
                    
    #------------------------------------------------------------------------------------
    
#==================================================================================================
        














             
#==================================================================================================
class NNBestModelParams():
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent):
        #........ |  Instance Attributes | ..............................................
        self.Parent = p_oParent
        
        # Maximum distance of epochs to the best epoch, to stop when the model stops to improve.
        self.StopAfterEpochs                = 10
        self.MaxSavedModels                 = 8
        self.AccuracyLowLimitToSave         = 0.95  #for MNIST
        self.AccuracyHighLimitToAutoSave    = 0.975 #for MNIST
        #................................................................................
    #------------------------------------------------------------------------------------
#==================================================================================================







#==================================================================================================
class NNBasicStopParams():
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent):
        #........ |  Instance Attributes | ..............................................
        self.Parent = p_oParent
        

        # Periods of recovery after a serious stop error condition in unstable optimizations  
        self.RecoveryPeriod = 3
        #................................................................................
    #------------------------------------------------------------------------------------



#==================================================================================================









#==================================================================================================
class NNMarginParams():
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent):
        #........ |  Instance Attributes | ..............................................
        self.Parent = p_oParent
        
        self.IsActive = False
        self.MarginFullyClassified = 1.0
        self.MarginFlags = 6
        self.MarginRatioPositive = 0.7
        self.MarginRatioPositive = 0.9  
        #................................................................................
    #------------------------------------------------------------------------------------
#==================================================================================================






              
              
        
        
        
              
              
              
              
              
#==================================================================================================              
class NNLearnParams():
    #------------------------------------------------------------------------------------
    def __init__(self, p_nTrainingMethod=tcc.TRN_MOMENTUM, p_bIsSecondPassTraining=False):
        #........ |  Instance Attributes | ..............................................
        
        
        self.BestModel = NNBestModelParams(self)
        self.Margin = NNMarginParams(self)
        
        
        
        # If the network is continuing training over a saved state or fine-tuning
        self.IsSecondPassTraining=p_bIsSecondPassTraining
        
        self.MiniBatches=NNMiniBatches()
        self.PseudoRandomSeed=2017        
        self.IsUsingFragments=False
        self.IsShuffling=True
            
        # Training method type
        self.TrainingMethodType=None
        # Learning rate       
        self.LearningRate=None
        # Momentum
        self.Momentum=None
        # ADAM / Epsilon hyperparameter
        self.Epsilon=1e-8
        
        
        self.IsNesterovMomentum=False
        self.Beta1=0.9
        self.Beta2=0.999
        self.DecayEpochs=1
        self.DecayRate=0.95
        self.MinimumLearningRate=0.00006

        
        self.IsMargin = False
        
        
        # Number of epochs to check for filter. None, filtering is disabled 
        self.FilterCheckEpochs = None
        # Type of margin. Values 0 (SVMBased) 1 (SalesBased)
        self.MarginType = 0
        # Low limit of margin values, above which samples are excluded
        self.MarginLowLimit = None
        # High limit of margin absolute value, below which samples are excluded
        self.MarginAbsHighLimit = None
        self.WarmupEpochs = None
        self.WarmupLRMultiplier = None

        
        
        self.MarginRatioPositive = 1.3
        self.MarginRatioNegative = 1.8
        
        
        # Common stop conditions
        #
        self.MaxTrainingEpochs=100

                
        self.MinTrainingEpochs=30
        self.MinTrainingError=0.001
        self.MaxTrainingAccuracy=1.0
    
        # Adaptive
        self.IsThrottledLearning=True
        self.AdaptSlowKickIn=True
        self.AdaptBoostFirstEpochs=True
        self.AdaptStartAtMiddle=False
        self.AdaptDeltaValErr=0.005
        self.AdaptEpochDecayHalfEpochs=10
        self.AdaptMaxLearningRate=0.1
        self.LearningRateMultiplier=1.6
        self.AdaptResetUnchangedBestEpoch=None #LEGACY value 5
        self.MomentumDivider=1.6  #Experimental
   
        # Limits for auto-saving of state during trainings
        self.AccuracyHighLimit=0.975 #TEMP: Moved
        self.AccuracyLowLimit=0.95   #TEMP: Moved
        self.NumEpochsToKeep=6       #TEMP: Moved

        # Training quality function parameters
        self.QualityPenaltyOverfitAccuracyPCOver=0.5
        self.QualityPenaltyOverfit=0.02


        
        # Stopping sub mechanisms enable/disable
        self.IsStoppingOnStd=True
        self.IsStoppingOnDeterioration=True
        self.IsAccumulativeStopConditions=True


        # Maximum distance of epochs to the best epoch, to stop when the model stops to improve.
        self.StopLimitUnchangedBestEpoch = 10 #TEMP: Moved
        
        # // Stop conditions for accumulating failures \\
        self.StopConditionInc=[13,20,20]   # For 100% completion this equals to having MaxFailures=[5,5,5]
        self.StopConditionDecDiv=[4,2,4]
        self.StopConditionLimits=[0.0001, 0.0001, 0]
        
        # Limit overfit rate to stop the training
        self.StopLimitOverfit=10
        
        # The maximum acceptable increase in validation error, over which the training stops
        self.StopBigIncreaseValErr=0.04
        
        # The acceptable decrease in validation accuracy, larger difference will stop the training 
        self.StopBigDecreaseValAcc=-0.08        

        # Very small standard deviation for validation error changes.  
        self.StopMinValidationErrorStd=0.001
    
        # Very small standard devivations for error and accuracy changes.
        self.StopMinTrainingErrorStd     =0.0001 #0.0009 
        self.StopMinTrainingAccuracyStd  =0.0005 #0.0007
        
        # Dropout for fully connected layers
        self.DropOutKeepProbability=0.5
        
        #................................................................................
        self.SetTrainingMethodType(p_nTrainingMethod)
    #------------------------------------------------------------------------------------
    def SetTrainingMethodType(self, p_oValue):
        self.TrainingMethodType = p_oValue
        
        if self.TrainingMethodType==tcc.TRN_ADAM:
            self.LearningRate=tcc.DEFAULT_LR_TRN_ADAM
        elif self.TrainingMethodType==tcc.TRN_SGD:
            self.LearningRate=tcc.DEFAULT_LR_TRN_SGD
        elif self.TrainingMethodType==tcc.TRN_MOMENTUM:
            self.LearningRate=tcc.DEFAULT_LR_TRN_MOMENTUM
            self.Momentum=tcc.DEFAULT_MO_TRN_MOMENTUM
            
        # Updates the training method name
        self.__getTrainingMethodName()
    #------------------------------------------------------------------------------------       
    def __setTrainingMethodName(self, p_sTrainingMethodName):
        if p_sTrainingMethodName is not None:
            self.TrainingMethodName = p_sTrainingMethodName
            
            self.TrainingMethodType = None
            if self.TrainingMethodName == "ADAM":
                self.TrainingMethodType = tcc.TRN_ADAM
            elif self.TrainingMethodName == "SGD":
                self.TrainingMethodType = tcc.TRN_SGD
            elif self.TrainingMethodName == "MOMENTUM":
                self.TrainingMethodType = tcc.TRN_MOMENTUM
    #------------------------------------------------------------------------------------
    def __getTrainingMethodName(self):
        sName = None
        if self.TrainingMethodType==tcc.TRN_ADAM:
            sName = "ADAM"
        elif self.TrainingMethodType==tcc.TRN_SGD:
            sName = "SGD"
        elif self.TrainingMethodType==tcc.TRN_MOMENTUM:
            sName = "MOMENTUM"
             
        self.TrainingMethodName = sName
        return self.TrainingMethodName
    #------------------------------------------------------------------------------------
    def GetConfig(self, p_oConfig):
        if p_oConfig.Has("Learn/TrainingMethodType")    : self.TrainingMethodType   = p_oConfig.Get("Learn/TrainingMethodType")
        sTrainingMethodName = None
        if p_oConfig.Has("Learn/TrainingMethod")        : sTrainingMethodName       = p_oConfig.Get("Learn/TrainingMethod")
        self.__setTrainingMethodName(sTrainingMethodName)
        if p_oConfig.Has("Learn/MaxTrainingEpochs")     : self.MaxTrainingEpochs    = p_oConfig.Get("Learn/MaxTrainingEpochs")
        if p_oConfig.Has("Learn/LearningRate")          : self.LearningRate         = p_oConfig.Get("Learn/LearningRate")
        if p_oConfig.Has("Learn/ADAM.Epsilon")          : self.Epsilon              = p_oConfig.Get("Learn/ADAM.Epsilon")
        if p_oConfig.Has("Learn/Momentum")              : self.Momentum             = p_oConfig.Get("Learn/Momentum")
        if p_oConfig.Has("Learn/FilterCheckEpochs")     : self.FilterCheckEpochs    = p_oConfig.Get("Learn/FilterCheckEpochs")
        if p_oConfig.Has("Learn/Margin.Type")           : self.MarginType           = p_oConfig.Get("Learn/Margin.Type")
        if p_oConfig.Has("Learn/Margin.LowLimit")       : self.MarginLowLimit       = p_oConfig.Get("Learn/Margin.LowLimit")
        if p_oConfig.Has("Learn/Margin.AbsHighLimit")   : self.MarginAbsHighLimit   = p_oConfig.Get("Learn/Margin.AbsHighLimit")        
        if p_oConfig.Has("Learn/Warmup.Epochs")         : self.WarmupEpochs         = p_oConfig.Get("Learn/Warmup.Epochs")
        if p_oConfig.Has("Learn/Warmup.LRMultiplier")   : self.WarmupLRMultiplier   = p_oConfig.Get("Learn/Warmup.LRMultiplier")
    #------------------------------------------------------------------------------------                
    def SetConfig(self, p_oConfig):
        #p_oConfig.Set("Learn/TrainingMethodType"    , self.TrainingMethodType)
        p_oConfig.Set("Learn/TrainingMethod"        , self.__getTrainingMethodName())
        p_oConfig.Set("Learn/MaxTrainingEpochs"     , self.MaxTrainingEpochs)
        p_oConfig.Set("Learn/LearningRate"          , self.LearningRate)
        p_oConfig.Set("Learn/ADAM.Epsilon"          , self.Epsilon)
        p_oConfig.Set("Learn/Momentum"              , self.Momentum)
        p_oConfig.Set("Learn/FilterCheckEpochs"     , self.FilterCheckEpochs)
        p_oConfig.Set("Learn/Margin.Type"           , self.MarginType)
        p_oConfig.Set("Learn/Margin.LowLimit"       , self.MarginLowLimit)
        p_oConfig.Set("Learn/Margin.AbsHighLimit"   , self.MarginAbsHighLimit)
        p_oConfig.Set("Learn/Warmup.Epochs"         , self.WarmupEpochs)
        p_oConfig.Set("Learn/Warmup.LRMultiplier"   , self.WarmupLRMultiplier)
    #------------------------------------------------------------------------------------
    def Print(self, p_oLog = None):
        if p_oLog is None: return

        p_oLog.Print("Learn/TrainingMethod"             , self.TrainingMethodName)    
        p_oLog.Print("Learn/TrainingMethodType"         , self.TrainingMethodType)
        p_oLog.Print("Learn/MaxTrainingEpochs"          , self.MaxTrainingEpochs)
        p_oLog.Print("Learn/LearningRate"               , self.LearningRate)
        p_oLog.Print("Learn/ADAM.Epsilon"               , self.Epsilon)
        p_oLog.Print("Learn/ADAM.Momentum"              , self.Momentum)
        p_oLog.Print("Learn/ADAM.FilterCheckEpochs"     , self.FilterCheckEpochs)
        p_oLog.Print("Learn/Margin.Type"                , self.MarginType)
        p_oLog.Print("Learn/Margin.LowLimit"            , self.MarginLowLimit)
        p_oLog.Print("Learn/Margin.AbsHighLimit"        , self.MarginAbsHighLimit)
        p_oLog.Print("Learn/Warmup.Epochs"              , self.WarmupEpochs)
        p_oLog.Print("Learn/Warmup.LRMultiplier"        , self.WarmupLRMultiplier)
    #------------------------------------------------------------------------------------
    def CopyMinibatchParamsFromInputQueue(self, p_oInputQueue):
        nCountTrainingBatches       = (p_oInputQueue.TrainingSamplesCount - p_oInputQueue.ValidationSampleCount) / p_oInputQueue.MiniBatchSize
        assert nCountTrainingBatches == int(nCountTrainingBatches)
        nCountTrainingBatches = int(nCountTrainingBatches)
        
        nCountValidationBatches     = p_oInputQueue.ValidationSampleCount /  p_oInputQueue.MiniBatchSize
        assert nCountValidationBatches == int(nCountValidationBatches)
        nCountValidationBatches = int(nCountValidationBatches)
           
        nTrainMiniBatchSize         = p_oInputQueue.MiniBatchSize
        nValidationMiniBatchSize    = p_oInputQueue.MiniBatchSize

        self.MiniBatches.TrainingSamples   = p_oInputQueue.TrainingSamplesCount
        self.MiniBatches.ValidationSamples = p_oInputQueue.ValidationSampleCount
        
        self.MiniBatches.TrainingSamplesPerBatch=nTrainMiniBatchSize
        self.MiniBatches.ValidationSamplesPerBatch=nValidationMiniBatchSize
        
        self.MiniBatches.TrainingBatches=nCountTrainingBatches
        self.MiniBatches.ValidationBatches=nCountValidationBatches
                        
        #self.MiniBatches.TrainingSamplesPerFrag=None
        #self.MiniBatches.ValidationSamplesPerFrag=None
    #------------------------------------------------------------------------------------
    def SetAdaptiveLearningParams(self, p_nMomentum, p_nStartLearningRate, p_nMaxLearningRate=None, p_nMaxDesiredAdaptations=None):
        
        self.Momentum=p_nMomentum
        self.LearningRate=p_nStartLearningRate
        
        if (p_nMaxLearningRate is None) and (p_nMaxDesiredAdaptations is not None):
            self.AdaptMaxLearningRate = self.LearningRate*((self.LearningRateMultiplier**p_nMaxDesiredAdaptations) +  1) 
        elif (p_nMaxLearningRate is not None):
            self.AdaptMaxLearningRate = p_nMaxLearningRate
    #------------------------------------------------------------------------------------
    def ImportFromDict(self, p_dLearnParams):
        self.StopLimitUnchangedBestEpoch=GetValue(p_dLearnParams,"StopLimitUnchangedBestEpoch")        
        self.Momentum=GetValue(p_dLearnParams,"Momentum")
        self.LearningRate=GetValue(p_dLearnParams,"LearningRate")
        self.AdaptMaxLearningRate=GetValue(p_dLearnParams,"MaxLearningRate")
        self.AdaptDeltaValErr=GetValue(p_dLearnParams,"AdaptOnSmallDeltaValErr")
        self.AdaptBoostFirstEpochs=GetValueAsBool(p_dLearnParams,"AdaptBoostFirstEpochs") 
        self.AdaptSlowKickIn=GetValueAsBool(p_dLearnParams, "AdaptSlowKickIn")
        self.AdaptStartAtMiddle=GetValueAsBool(p_dLearnParams,"AdaptStartAtMiddle")
        self.DropOutKeepProbability=GetValue(p_dLearnParams,"DropOutKeepProbability")
    #------------------------------------------------------------------------------------
    def CopyFrom(self, p_oSourceParams):
        #TEMP        
        if p_oSourceParams.StopLimitUnchangedBestEpoch is not None:
            self.StopLimitUnchangedBestEpoch = p_oSourceParams.StopLimitUnchangedBestEpoch
            
        if p_oSourceParams.AdaptMaxLearningRate is not None:      
            self.SetAdaptiveLearningParams(
                                 p_oSourceParams.Momentum
                                ,p_oSourceParams.LearningRate
                                ,p_nMaxLearningRate=p_oSourceParams.AdaptMaxLearningRate
                                )

        if p_oSourceParams.AdaptDeltaValErr is not None:    
            self.AdaptDeltaValErr           = p_oSourceParams.AdaptDeltaValErr
            
        if p_oSourceParams.AdaptBoostFirstEpochs is not None:
            self.AdaptBoostFirstEpochs      = p_oSourceParams.AdaptBoostFirstEpochs
        
        if p_oSourceParams.AdaptSlowKickIn is not None:
            self.AdaptSlowKickIn            = p_oSourceParams.AdaptSlowKickIn
        
        if p_oSourceParams.AdaptStartAtMiddle is not None:
            self.AdaptStartAtMiddle         = p_oSourceParams.AdaptStartAtMiddle
            
        if p_oSourceParams.DropOutKeepProbability is not None:
            self.DropOutKeepProbability = p_oSourceParams.DropOutKeepProbability     
    #------------------------------------------------------------------------------------
    def Dump(self):
        print("."*128)
        print(self.StopLimitUnchangedBestEpoch)        
        print("%.6f" % self.Momentum)
        print("%.6f" % self.LearningRate)
        print("%.6f" % self.AdaptMaxLearningRate)
        print("AdaptBoostFirstEpochs:", self.AdaptBoostFirstEpochs)
        print("AdaptSlowKickIn:", self.AdaptSlowKickIn)   
        print("AdaptStartAtMiddle", self.AdaptStartAtMiddle)
        print("."*128)
    #------------------------------------------------------------------------------------
        
            
#==================================================================================================








  
    
    
    
    
    
    
    
    


#==================================================================================================
class NNLearnConfig(JSONConfig):
    
    #------------------------------------------------------------------------------------    
    @classmethod
    def GetConfig(cls, p_sFolder):
        oResult = None
        
        sFolder = Storage.JoinPath(p_sFolder, "config")
        if Storage.IsExistingPath(sFolder):
            sFileList = Storage.GetFilesSorted(sFolder)
            sFileName = None
            for sItem in sFileList:
                if sItem.startswith("learn-config-used"):
                    sFileName = Storage.JoinPath(sFolder, sItem)
                    break
            
            if sFileName is not None:
                oResult = NNLearnConfig()
                oResult.LoadFromFile(sFileName)
                oResult.ParseUID()
        return oResult    
    #------------------------------------------------------------------------------------





        
    
    #------------------------------------------------------------------------------------
    def __init__(self, p_sArchitecture=None, p_sDataSetName=None, p_nBatchSize=40, p_nFoldNumber=10, p_nDataSetVariationStartPos=None, p_sERLString=None):
        super(NNLearnConfig, self).__init__(None)
        #........ |  Instance Attributes | ..............................................
        # // Composites \\
        if p_sERLString is not None:
            self.ERL = ERLString(p_sERLString)
        else:
            self.ERL = None
            
        #TODO: Test ERL Functionality
        nExperimentUID = None
        if self.ERL is not None:
            p_sArchitecture = self.ERL.Architecture
            p_sDataSetName = self.ERL.DataSetName
            p_nFoldNumber = self.ERL.FoldNumber
            nExperimentUID = self.ERL.ExperimentUID
            
        # // Control Variables \\
        self.SavedExperimentUID = nExperimentUID
                
        # // Properties \\
        self.Architecture       = p_sArchitecture
        self.DataSetName        = p_sDataSetName
        if p_nDataSetVariationStartPos is not None:
            self.DataSetVariation = int(self.DataSetName[p_nDataSetVariationStartPos:])
        else:
            self.DataSetVariation = None
        self.FoldNumber         = p_nFoldNumber 
        self.BatchSize          = p_nBatchSize
       
        self.IsTraining         = True
        self.IsEvaluating       = True
        self.IsDeterminingBest  = True
        
        self.InitExperimentCode = None
        self.InitExperimentFoldNumber = None
        self.InitExperimentUID = None
        self.Learn = NNLearnParams()

        #................................................................................
    #------------------------------------------------------------------------------------
    def LoadFromERL(self, p_sERLString):
        self.ERL = ERLString(p_sERLString)

        self.FileName = None
        if self.ERL.IsFull:
            self.FileName = Storage.JoinPaths([  BaseFolders.EXPERIMENTS_RUN
                                            ,"%s-%s" % (self.ERL.DataSetName, self.ERL.Architecture)
                                            ,"fold%.2d" % self.ERL.FoldNumber
                                            ,"%s" % self.ERL.ExperimentUID
                                            ,"config" 
                                            ,"learn-config-used-%s.cfg" % self.ERL.ExperimentUID
                                           ])            
        assert self.FileName is not None, "ERL is not valid"
        
        self.LoadFromFile()
    #------------------------------------------------------------------------------------
    def ParseUID(self):
        if len(self.FileName) >= 12:
            _, sName, _ = Storage.SplitFileName(self.FileName)
            self.SavedExperimentUID = sName[-12:]
    #------------------------------------------------------------------------------------
    def ArchiveFileName(self, p_nCounter):
        
        _, sName, sExt = Storage.SplitFileName(self.FileName)
        
        sPrefix = None
        if (self.IsTraining == True):
            # Training run
            sPrefix = "r"
        elif (self.IsTraining == False) and ((self.IsEvaluating == True) or (self.IsDeterminingBest == True)):
            # Evaluation run
            sPrefix = "e"
            
        assert sPrefix is not None
        # Prefix is valid for display up to 1000 experiments   
        assert p_nCounter < 1000
        
        sResult = sPrefix + "%.3d-" % p_nCounter + sName + sExt
        
        return sResult
    #------------------------------------------------------------------------------------
    def AssignFrom(self, p_oSource):
        if p_oSource is None: 
            return
        
        self.Architecture               = p_oSource.Architecture
        self.DataSetName                = p_oSource.DataSetName
        self.DataSetVariation           = p_oSource.DataSetVariation
        self.FoldNumber                 = p_oSource.FoldNumber 
        self.BatchSize                  = p_oSource.BatchSize
       
        self.IsTraining                 = p_oSource.IsTraining
        self.IsEvaluating               = p_oSource.IsEvaluating
        self.IsDeterminingBest          = p_oSource.IsDeterminingBest
        
        self.InitExperimentCode         = p_oSource.InitExperimentCode
        self.InitExperimentFoldNumber   = p_oSource.InitExperimentFoldNumber
        self.InitExperimentUID          = p_oSource.InitExperimentUID
        
        #TODO: Assign learn params
    #------------------------------------------------------------------------------------
    def DoBeforeSave(self):
        self.Set("GUID"                     , self.GUID)
        self.Set("AncestorGUID"             , self.AncestorGUID)
        self.Set("Architecture"             , self.Architecture)
        self.Set("Data/DataSetName"         , self.DataSetName)
        self.Set("Data/FoldNumber"          , self.FoldNumber)
        self.Set("Data/DataSetVarion"       , self.DataSetVariation)
        self.Set("BatchSize"                , self.BatchSize)
        self.Set("Flags"                    , [self.IsTraining, self.IsEvaluating, self.IsDeterminingBest])
        self.Set("InitExperimentCode"       , self.InitExperimentCode)
        self.Learn.SetConfig(self)
    #------------------------------------------------------------------------------------
    def DoAfterLoad(self):
        if self.Has("GUID"):
            sGUID = self.Get("GUID")
            if sGUID is not None:
                self.GUID = sGUID 
        self.AncestorGUID           = self.Get("AncestorGUID")
        
        self.Architecture           = self.Get("Architecture")
        self.DataSetName            = self.Get("Data/DataSetName")
        self.FoldNumber             = self.Get("Data/FoldNumber")
        self.DataSetVariation       = self.Get("Data/DataSetVarion")
        self.BatchSize              = self.Get("BatchSize")
        bFlags = self.Get("Flags")
        self.IsTraining             = bFlags[0]
        self.IsEvaluating           = bFlags[1]
        self.IsDeterminingBest      = bFlags[2]
        self.InitExperimentCode     = self.Get("InitExperimentCode")
        if self.InitExperimentCode is not None:
            if self.InitExperimentCode == "=":
                self.InitExperimentFoldNumber = 1
                self.InitExperimentUID        = None
            else:
                sParts = self.InitExperimentCode.split("/")
                self.InitExperimentFoldNumber = int(sParts[0])
                self.InitExperimentUID        = sParts[1]
        else:
            self.InitExperimentFoldNumber = None
            self.InitExperimentUID = None
        
        self.Learn.GetConfig(self)
    #------------------------------------------------------------------------------------
    def DoPrint(self):
        self.Log.Print("Architecture", self.Architecture)
        self.Log.Print("DataSetName", self.DataSetName)
        self.Log.Print("DataSetVariation", self.DataSetVariation)
        self.Log.Print("FoldNumber", self.FoldNumber)
        self.Log.Print("BatchSize", self.BatchSize)
        self.Log.Print("IsTraining", self.IsTraining)
        self.Log.Print("IsEvaluating", self.IsEvaluating)
        self.Log.Print("IsDeterminingBest", self.IsDeterminingBest)

        self.Log.Print("InitExperimentCode", self.InitExperimentCode)        
        self.Log.Print("InitExperimentFoldNumber", self.InitExperimentFoldNumber)
        self.Log.Print("InitExperimentUID", self.InitExperimentUID)
        
        self.Learn.Print(self.Log)
                
#==================================================================================================

#     
#                 self.LearnParams.IsMargin = False      
#                 self.LearnParams.MaxTrainingEpochs = 50
#                 self.LearnParams.LearningRate = 0.00006
#                 self.LearnParams.Epsilon = 0.001    
#     
    
    
    
    
    
    





#==================================================================================================
class BaseDNA(object):
    #---------------------------------------------------------------------------------------------------------
    def __init__(self):
        pass
    #---------------------------------------------------------------------------------------------------------
    def ImportFromDict(self, p_dRetinaSettings):
        pass
    #---------------------------------------------------------------------------------------------------------







#==================================================================================================
class NeuralNetworkDefition(object):
    #------------------------------------------------------------------------------------
    def __init__(self, p_sModelName):
        #........ |  Instance Attributes | ..............................................        
        self.ModelName  = p_sModelName
        self.ModelDefinitionFileName = MODEL_DEFINITION_FOLDER + self.ModelName + ".nndef"
        
        self.BaseArchitecture=None
        self.GenotypeName=None
        self.CreationDate=None
        self.IsEvaluating=None
        self.IsEvaluatingByte=None
        self.FoldsToTrain=None
        self.FoldsFinished=None
        self.FoldsRemaining=None
                
        self.Classes=None
        self.Features=None
        self.Learn=NNLearnParams()
        self.DNA=BaseDNA() 
        self.Comments=[]
        #................................................................................
    #------------------------------------------------------------------------------------        
    def Load(self):
        if self.ModelDefinitionFileName is not None:
            with open(self.ModelDefinitionFileName) as oDefFile:    
                dJSON = json.load(oDefFile)
        self.BaseArchitecture=dJSON["BaseArchitecture"]
        self.GenotypeName=dJSON["GenotypeName"]
        self.CreationDate=dJSON["CreationDate"]
        self.IsEvaluatingByte=dJSON["IsEvaluating"]
        self.IsEvaluating=self.IsEvaluatingByte != 0
        self.FoldsToTrain=dJSON["FoldsToTrain"]
        
        self.FoldsFinished=GetValue(dJSON,"FoldsFinished")
        self.FoldsRemaining=GetValue(dJSON,"FoldsRemaining") 

        
        self.Classes=int(dJSON["Classes"])
        self.Features=dJSON["Features"]
        self.Features.append(self.Classes)
        
        dLearnParams=dJSON["LearnParameters"]
        self.Learn.ImportFromDict(dLearnParams)
        
        dRetinaSettings=dJSON["Genotype"]
        self.DNA.ImportFromDict(dRetinaSettings)
        
        self.Comments=[]
        for sKey, sValue  in dJSON.items():
            if sKey.startswith("#"):
                self.Comments.append(sKey + " " + sValue)

    #------------------------------------------------------------------------------------
    def Dump(self):
        print("-"*128)
        print(self.BaseArchitecture)
        print(self.GenotypeName)
        print(self.Features)
        self.Learn.Dump()
        print("-"*128)
        for sComment in self.Comments:
            print(sComment)        
    #------------------------------------------------------------------------------------        

