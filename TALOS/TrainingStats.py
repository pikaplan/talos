# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        NEURAL NETWORK TRAINING STATISTICS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import numpy as np
import tensorflow as tf
from io import BytesIO#StringIO 
from TALOS.Visualization import GraphConsts
import matplotlib.pyplot as plt
import TALOS.Constants as tcc
from TALOS.Constants import EpochPhase
from TALOS.Core import StopWatch, ScreenLogger, ERLString, BaseFolders
from TALOS.Visualization import LinePlot 
from TALOS.FileSystem import Storage
#TEMP
from TALOS.DataSets.Factory import DataSetFactory
from TALOS.HyperParams import NNLearnParams, NNLearnConfig

#==================================================================================================
class StatsColumnType(object):
    VALUE               = 0
    DELTA_VALUE         = 1
    #ARCTAN_DELTA_VALUE  = 2
    #COUNT               = 3
    
    
    @classmethod
    def ToString(cls, p_sBaseDescr, p_nType):
        p_sBaseDescr = p_sBaseDescr.replace(" ", "")
        if p_nType == StatsColumnType.VALUE:
            return p_sBaseDescr
        elif p_nType == StatsColumnType.DELTA_VALUE:
            return "D%s" % p_sBaseDescr
        #elif p_nType == StatsColumnType.ARCTAN_DELTA_VALUE:
        #    return "ArcTan(Δ%s)" % p_sBaseDescr
        return  
#==================================================================================================





        
            
    
    

#==================================================================================================
class NNTrainingStats():
    __verboseLevel = 3

    #------------------------------------------------------------------------------------
    def __init__(self, p_oLearnParams=None, p_oTensorboardWriter=None, p_oLog=None, p_nClassCount=1000, p_oParent=None):
        # ........ |  Instance Attributes | ..............................................
        self.Parent = p_oParent
        if self.Parent is not None:
            p_oLearnParams = self.Parent.Network.LearnParams
            p_oLog = self.Parent.Network.Log
            p_nClassCount = self.Parent.Network.Settings.ClassCount
            p_oTensorboardWriter = self.Parent.Writer
            
        assert p_oLearnParams is not None, "No learning parameters given"
        
        if p_oLog is None:
            self.Log = ScreenLogger()
        else:
            self.Log = p_oLog
        self.TensorboardWriter = p_oTensorboardWriter
        self.LearnParams = p_oLearnParams
        self.StopWatch = StopWatch()
        self.VerboseSteps = 20
        self.ClassCount = p_nClassCount
        self.SummaryImage = None
        
        # // Stats \\
        self.EpochSamples = None       
        self.EpochTimings = None   
        self.EpochTotalTime = None
        
        self.TrainError = None
        self.TrainAccuracy = None
        self.TrainTopKAccuracy = None
        self.TrainStepAvgF1Score = None
        self.TrainAvgF1Score = None
        self.TrainClassRecall = None
        self.TrainClassPrecision = None
        self.TrainClassF1Score = None
        self.TrainClassValidIndices = None
    
        self.ValError = None
        self.ValAccuracy = None
        self.ValAccuracyPerError = None
        self.ValOverfitError = None
        self.ValOverfitAccuracy = None
        self.ValOverfitRate = None
        self.ValLearningQuality = None
                
        self.ValAvgRecall = None
        self.ValAvgPrecision = None
        self.ValAvgF1Score = None    
        self.ValClassTrue = None
        self.ValClassFalse = None
        self.ValClassActual = None
        self.ValClassRecall = None
        self.ValClassPrecision = None
        self.ValClassF1Score = None
        self.ValMisclassified = None
        
        
        self.LearningRate = None
        
        # .... Averages ....
        self.TrainAvgError = None
        self.TrainAvgAccuracy = None
        
        # .... Minimum ....
        self.MinTrainAvgError = None
        self.MinValAvgError = None
        
        # // Collections \\
        self.Validations=None

        # // Training Setup \\
        self.MaxBatchIndex = None
   
            
        # // Control Attributes \\
        self.EpochIndex = 0
        self.EpochStepIndex = 0
        self.StepNumber = 0
        self.MaxEpoch = 0
        
        self.TrainBatchesCount = None
        self.ValBatchesCount = None
        self.MaxBatchIndex = None
        # PANTELIS: [2017-04-29] self.StartTime = datetime.now()        

        self.BestEpochs = [None] * self.LearnParams.NumEpochsToKeep
        self.StopConditionHits = [0] * len(self.LearnParams.StopConditionLimits)
        self.LastStopCondition = tcc.DONT_STOP
        self.StopReason = None
        self.StopID = None
        
        # // Settings \\
        self.FileName = None
        self.IsBatchAnalysis=True
        self.IsNewOverfitMargin=True      
                       
        # ................................................................................
        # Conditions for stopping the training 
        nEpochsCount = self.LearnParams.MaxTrainingEpochs
        self.MaxEpoch = nEpochsCount
        
        
        

        
        self.Validations=[]
        
        self.EpochSamples   = np.zeros((nEpochsCount), np.int32)
        self.EpochTimings   = np.zeros((nEpochsCount, EpochPhase.Count), np.float32)
        self.EpochRecallTime= np.zeros((nEpochsCount), np.float32)
        self.EpochTotalTime    = np.zeros((nEpochsCount), np.float32)
        #..posterior calc...
        self.TwoEpochTotalTime = np.zeros( ((nEpochsCount + 1) //2), np.float32) 
                
        if self.Parent is not None:
            self.TrainBatchesCount = self.Parent.Iterator.TrainingBatches
            self.ValBatchesCount = self.Parent.Iterator.ValidationBatches
        else:
            # For loading only
            self.TrainBatchesCount = 1
            self.ValBatchesCount = 1
            
            
        self.MaxBatchIndex = self.TrainBatchesCount - 1
        
        self.TrainError = np.zeros((nEpochsCount, self.TrainBatchesCount), np.float32)
        self.TrainAccuracy = np.zeros((nEpochsCount, self.TrainBatchesCount), np.float32)
        self.TrainTopKAccuracy = np.zeros((nEpochsCount, self.TrainBatchesCount), np.float32)
        self.TrainStepAvgF1Score = np.zeros((nEpochsCount, self.TrainBatchesCount), np.float32)
        self.TrainAvgError = np.zeros((nEpochsCount, 3), np.float32)
        self.TrainAvgAccuracy = np.zeros((nEpochsCount, 3), np.float32)
        self.TrainAvgF1Score = np.zeros((nEpochsCount, 3), np.float32)

        self.TrainClassRecall = np.zeros((nEpochsCount, self.TrainBatchesCount, self.ClassCount), np.float32)
        self.TrainClassPrecision = np.zeros((nEpochsCount, self.TrainBatchesCount, self.ClassCount), np.float32)
        self.TrainClassF1Score = np.zeros((nEpochsCount, self.TrainBatchesCount, self.ClassCount), np.float32)
        self.TrainClassValidIndices = np.zeros((nEpochsCount, self.TrainBatchesCount, self.ClassCount), np.float32)
        
        self.ValError = np.zeros((nEpochsCount, 3), np.float32)  # [:,0] value [:,1] delta [:,2] slope
        self.ValAccuracy = np.zeros((nEpochsCount, 3), np.float32)  # [:,0] value [:,1] delta [:,2] slope
        self.ValAccuracyPerError = np.zeros((nEpochsCount), np.float32)
        self.ValOverfitError = np.zeros((nEpochsCount, 2), np.float32)  # [:,0] value [:,1] delta [:,2] slope
        self.ValOverfitAccuracy = np.zeros((nEpochsCount, 2), np.float32)  # [:,0] value [:,1] delta [:,2] slope
        self.ValOverfitRate = np.zeros((nEpochsCount, 2), np.float32)  # [:,0] value [:,1] delta [:,2] slope
        self.ValLearningQuality = np.zeros((nEpochsCount), np.float32)
        
        
        self.ValAvgRecall = np.zeros((nEpochsCount, 2), np.float32)
        self.ValAvgPrecision = np.zeros((nEpochsCount, 2), np.float32)
        self.ValAvgF1Score = np.zeros((nEpochsCount, 2), np.float32)
        self.ValClassTrue = np.zeros((nEpochsCount, self.ClassCount), np.float32)
        self.ValClassFalse = np.zeros((nEpochsCount, self.ClassCount), np.float32)
        self.ValClassActual = np.zeros((nEpochsCount, self.ClassCount), np.float32)
        self.ValClassRecall = np.zeros((nEpochsCount, self.ClassCount), np.float32)
        self.ValClassPrecision = np.zeros((nEpochsCount, self.ClassCount), np.float32)
        self.ValClassF1Score = np.zeros((nEpochsCount, self.ClassCount), np.float32)
        self.ValMisclassified = np.zeros((nEpochsCount, self.ClassCount), np.float32)
        
        self.LearningRate =  np.zeros((nEpochsCount, 2), np.float32)  # [:,0] value [:,1] delta [:,2] slope


        self.MinTrainAvgError = np.inf
        self.MinValAvgError = np.inf

        # Begin time and display to screen
        self.StopWatch.Start()
        self.Log.Print("  |__ Starting time %s" % self.StopWatch.StartMomentStr)
    #------------------------------------------------------------------------------------
    def Finish(self, p_sFileName=None):
        if p_sFileName is not None:
            self.FileName = p_sFileName
        assert self.FileName is not None, "No filename given for the stats object"
        
        self.StopWatch.Stop()
        self.Log.Print("  |__ Finished at %s" % self.StopWatch.EndMomentStr)
        self.SaveStats(p_sFileName)
    #------------------------------------------------------------------------------------
    def SaveStats(self, p_sFileName=None):
        # TODO: LearnParams as dict
        if p_sFileName is not None:
            self.FileName = p_sFileName
        assert self.FileName is not None, "No filename given for the stats object"
        
        dTrainingStats = {
              "FileFormat"              : "TALOS10"
            , "EpochNumber"             : self.EpochIndex + 1
            , "StepNumber"              : self.StepNumber
            , "MaxEpoch"                : self.MaxEpoch
            , "TrainBatchesCount"       : self.TrainBatchesCount
            , "ValBatchesCount"         : self.ValBatchesCount
            , "MaxBatchIndex"           : self.MaxBatchIndex
           
            , "StartTime"               : self.StopWatch.StartMoment 
            , "EndTime"                 : self.StopWatch.EndMoment
            , "Elapsed"                 : self.StopWatch.Elapsed
            
            , "VerboseSteps"            : self.VerboseSteps
            
            , "MinTrainAverageError"    : self.MinTrainAvgError
            , "MinValAverageError"      : self.MinValAvgError
            
            , "TrainError"              : self.TrainError
            , "TrainAccuracy"           : self.TrainAccuracy
            , "TrainTopKAccuracy"       : self.TrainTopKAccuracy
            , "TrainStepAvgF1Score"     : self.TrainStepAvgF1Score
            , "TrainAverageError"       : self.TrainAvgError
            , "TrainAverageAccuracy"    : self.TrainAvgAccuracy
            , "TrainAvgF1Score"         : self.TrainAvgF1Score
            
            , "TrainClassRecall"        : self.TrainClassRecall
            , "TrainClassPrecision"     : self.TrainClassPrecision
            , "TrainClassF1Score"       : self.TrainClassF1Score
            , "TrainClassValidIndices"  : self.TrainClassValidIndices
                    
            , "ValError"                : self.ValError
            , "ValAccuracy"             : self.ValAccuracy
            , "ValAccuracyPerError"     : self.ValAccuracyPerError
            , "ValOverfitError"         : self.ValOverfitError
            , "ValOverfitAccuracy"      : self.ValOverfitAccuracy
            , "ValOverfitRate"          : self.ValOverfitRate
            , "ValLearningQuality"      : self.ValLearningQuality
            
            , "ValAvgRecall"            : self.ValAvgRecall
            , "ValAvgPrecision"         : self.ValAvgPrecision
            , "ValAvgF1Score"           : self.ValAvgF1Score
            , "ValClassTrue"            : self.ValClassTrue
            , "ValClassFalse"           : self.ValClassFalse
            , "ValClassActual"          : self.ValClassActual
            , "ValClassRecall"          : self.ValClassRecall
            , "ValClassPrecision"       : self.ValClassPrecision
            , "ValClassF1Score"         : self.ValClassF1Score
            , "ValMisclassified"        : self.ValMisclassified
            
            , "StopConditionHits"       : self.StopConditionHits
            , "LastStopCondition"       : self.LastStopCondition
            
            , "StopReason"              : self.StopReason
            , "StopID"                  : self.StopID
            , "BestEpochs"              : self.BestEpochs     
            
            , "EpochSamples"            : self.EpochSamples
            , "EpochTimings"            : self.EpochTimings
            , "EpochRecallTime"         : self.EpochRecallTime
            , "EpochTotalTime"          : self.EpochTotalTime 
            , "TwoEpochTotalTime"       : self.TwoEpochTotalTime      
         }
         
        Storage.SerializeObjectToFile(self.FileName, dTrainingStats, p_bIsOverwritting=True)
    #------------------------------------------------------------------------------------    
    def LoadStats(self, p_sFileName=None):
        # TODO: LearnParams as dict
        if p_sFileName is not None:
            self.FileName = p_sFileName
        assert self.FileName is not None, "No filename given for the stats object"
        
        oData = Storage.DeserializeObjectFromFile(self.FileName)

        if oData is not None:
            self.EpochIndex            = oData["EpochNumber"] - 1
            self.StepNumber            = oData["StepNumber"]
            self.MaxEpoch              = oData["MaxEpoch"]
            self.TrainBatchesCount     = oData["TrainBatchesCount"]
            self.ValBatchesCount       = oData["ValBatchesCount"]
            self.MaxBatchIndex         = oData["MaxBatchIndex"]
           
            self.StopWatch.StartMoment  = oData["StartTime"] 
            self.StopWatch.EndMoment    = oData["EndTime"] 
            self.StopWatch.Elapsed      = oData["Elapsed"]
                        
            self.VerboseSteps           = oData["VerboseSteps"]
                        
            self.MinTrainAvgError       = oData["MinTrainAverageError"]
            self.MinValAvgError         = oData["MinValAverageError"]
                        
            self.TrainError             = oData["TrainError"]
            self.TrainAccuracy          = oData["TrainAccuracy"]
            self.TrainTopKAccuracy      = oData["TrainTopKAccuracy"]
            self.TrainStepAvgF1Score    = oData["TrainStepAvgF1Score"]
            self.TrainAvgError          = oData["TrainAverageError"]
            self.TrainAvgAccuracy       = oData["TrainAverageAccuracy"]
            self.TrainAvgF1Score        = oData["TrainAvgF1Score"]
                        
            self.TrainClassRecall       = oData["TrainClassRecall"]
            self.TrainClassPrecision    = oData["TrainClassPrecision"]
            self.TrainClassF1Score      = oData["TrainClassF1Score" ]
            self.TrainClassValidIndices = oData["TrainClassValidIndices"]
                                
            self.ValError               = oData["ValError" ]
            self.ValAccuracy            = oData["ValAccuracy"]
            self.ValAccuracyPerError    = oData["ValAccuracyPerError"]
            self.ValOverfitError        = oData["ValOverfitError"]
            self.ValOverfitAccuracy     = oData["ValOverfitAccuracy"]
            self.ValOverfitRate         = oData["ValOverfitRate"]
            self.ValLearningQuality     = oData["ValLearningQuality"]
                        
            self.ValAvgRecall           = oData["ValAvgRecall"]
            self.ValAvgPrecision        = oData["ValAvgPrecision"]
            self.ValAvgF1Score          = oData["ValAvgF1Score"]
            self.ValClassTrue           = oData["ValClassTrue"]
            self.ValClassFalse          = oData["ValClassFalse"]
            self.ValClassActual         = oData["ValClassActual"]
            self.ValClassRecall         = oData["ValClassRecall"]
            self.ValClassPrecision      = oData["ValClassPrecision"]
            self.ValClassF1Score        = oData["ValClassF1Score"]
            self.ValMisclassified       = oData["ValMisclassified"]
                        
            self.StopConditionHits      = oData["StopConditionHits"]
            self.LastStopCondition      = oData["LastStopCondition"]
                        
            self.StopReason             = oData["StopReason"]
            self.StopID                 = oData["StopID"]
            self.BestEpochs             = oData["BestEpochs"]
                        
            self.EpochSamples           = oData["EpochSamples"]
            self.EpochTimings           = oData["EpochTimings"]
            self.EpochRecallTime        = oData["EpochRecallTime"]
            self.EpochTotalTime         = oData["EpochTotalTime"]
            if "TwoEpochTotalTime" in oData:
                self.TwoEpochTotalTime      = oData["TwoEpochTotalTime"]
    #------------------------------------------------------------------------------------
    def __findTimeNormalizer(self, p_nPhase=EpochPhase.RECALL_VALIDATE):
        nDiv = np.zeros((self.EpochSamples.shape[0]), np.float32)
        nDiv[:] = self.EpochSamples[0]
        nTimePerSample  = self.EpochTimings[:,p_nPhase] / nDiv
        nMed            = np.median(nTimePerSample)
            
        return nMed
    #------------------------------------------------------------------------------------
    def NormalizeTimings(self, p_nNormalizer):
        #0 recall/train
        #1 save model
        #2 validation
        #3 filter
        #4 savestats
        print("="*80)
        
        nMed = np.median(self.EpochTimings[:, EpochPhase.RECALL_VALIDATE])
        
        if p_nNormalizer is None:
            p_nRatio = 1.0
        else:
            p_nRatio = p_nNormalizer / nMed
        print("Ratio", p_nRatio)
        nTime1 = self.EpochTimings[:, EpochPhase.RECALL_TRAIN]
        nTime2 = self.EpochTimings[:, EpochPhase.RECALL_FILTER]
        
        #print(nTime1)
        #print("-"*40)
        #print(nTime2)
        
        
        
        self.NormEpochTiming = ((nTime1 + nTime2) / ( 2 * nMed))
        
        self.NormEpochTiming = (nTime1 + nTime2) * p_nRatio
        
        
        self.TotalTime = np.sum(self.NormEpochTiming)

        nCount = self.EpochTotalTime.shape[0]
        #nHalfCount = (nCount // 2) + 1
        
        for nIndex in range(0, nCount, 5):
            self.TwoEpochTotalTime[nIndex // 5] = 0.0
            if nIndex + 4 < nCount:
                self.TwoEpochTotalTime[nIndex // 5] = self.NormEpochTiming[nIndex] + \
                      self.NormEpochTiming[nIndex  + 1] + \
                      self.NormEpochTiming[nIndex  + 2] + \
                      self.NormEpochTiming[nIndex  + 3] + \
                      self.NormEpochTiming[nIndex  + 4]
            
            
                    
        
        #print(self.EpochTimings[EpochPhase.RECALL_VALIDATE,:])
        #print(self.EpochTimings[EpochPhase.RECALL_FILTER,:])

        
        return self.TotalTime, nMed
    #------------------------------------------------------------------------------------
    def PosteriorCalculation(self):
        print(self.EpochTotalTime.shape)
        print(self.TwoEpochTotalTime.shape)
        
        nCount = self.EpochTotalTime.shape[0]
        #nHalfCount = (nCount // 2) + 1
        
        for nIndex in range(0, nCount, 2):
            print(nIndex)
            self.TwoEpochTotalTime[nIndex // 2] = 0.0
            if nIndex + 1 < nCount:
                self.TwoEpochTotalTime[nIndex // 2] = self.EpochTotalTime[nIndex] + self.EpochTotalTime[nIndex  + 1]
            
    #------------------------------------------------------------------------------------
    def Clear(self):
        self.EpochIndex = 0
        self.EpochStepIndex = 0        
        self.StepNumber = 0
        self.MaxEpoch = 0
        
        self.Validations=[]
                
        self.TrainError[:,:] = 0.0
        self.TrainAccuracy[:,:] = 0.0
        self.TrainTopKAccuracy[:,:] = 0.0
        self.TrainStepAvgF1Score[:,:] = 0.0
        self.TrainAvgF1Score[:] = 0.0
        self.TrainClassRecall[:,:,:] = 0.0
        self.TrainClassPrecision[:,:,:] = 0.0
        self.TrainClassF1Score[:,:,:] = 0.0 
        self.TrainClassValidIndices[:,:,:] = False

        
        self.ValError[:,:] = 0.0
        self.ValAccuracy[:,:] = 0.0
        self.ValAccuracyPerError[:] = 0.0

        self.ValOverfitError[:,:] = 0.0
        self.ValOverfitAccuracy[:,:] = 0.0
        self.ValOverfitRate[:,:] = 0.0
        
        self.ValLearningQuality[:] = 0.0
        
        self.ValAvgRecall[:,:] = 0.0
        self.ValAvgPrecision[:,:] = 0.0
        self.ValAvgF1Score[:,:] = 0.0
        self.ValClassTrue[:,:] = 0.0
        self.ValClassFalse[:,:] = 0.0
        self.ValClassActual[:,:] = 0.0
        self.ValClassRecall[:,:] = 0.0
        self.ValClassPrecision[:,:] = 0.0
        self.ValClassF1Score[:,:] = 0.0
        self.ValMisclassified[:,:] = 0.0

        self.TrainAvgError[:,:] = 0.0
        self.TrainAvgAccuracy[:,:] = 0.0
        self.MinTrainAvgError = np.inf
        self.MinValAvgError = np.inf
        
        self.BestEpochs[:] = 0

        
        self.AdaptationCount = 0
        self.NewLearningRate = self.LearnParams.LearningRate
        if self.LearnParams.TrainingMethodType == tcc.TRN_MOMENTUM:
            self.NewMomentum = self.LearnParams.Momentum
        else:
            self.NewMomentum = 0

        
        self.LearnParams.Reset()
        
        self.StopConditionHits[:] = 0
        self.LastStopCondition = tcc.DONT_STOP
        
        self.StopReason             = None
        self.StopID                 = None
        self.BestEpochs             = None
                    
        self.EpochSamples[:]        = 0.0 
        self.EpochTimings[:,:]      = 0.0
        self.EpochRecallTime[:]     = 0.0
        self.EpochTotalTime[:]      = 0.0
        self.TwoEpochTotalTime[:]   = 0.0
    #------------------------------------------------------------------------------------
    def IsVerboseStep(self):
        return ((self.EpochStepIndex % self.VerboseSteps) == 0) or (self.EpochStepIndex >= self.MaxBatchIndex)
    #------------------------------------------------------------------------------------
    #TODO: UNDER_CONSTRUNCTION
    def CustomGraphs(self):
        COLOR_ORANGE = '#d95319'
        COLOR_GREEN = '#77ac30'

        tag = "stats"

        im_summaries = []
        
        # Write the image to a string
        #s = StringIO()
        nMax = self.EpochIndex + 1
        x=np.arange(0, nMax)
        y1=self.TrainAvgError[:nMax, 0]
        y2=self.ValError[:nMax, 0]
        plt.figure()
        
        print(x)
        print(y1)
        print(y2)
        
        #LinePlot("Epoch", x, ["CCE"], [y1,y2], ["Train", "Validation"], [COLOR_ORANGE, COLOR_GREEN] )
        LinePlot("Epoch", x, "CCE", [y1,y2], ["TrainCCE", "ValCCE"] )
        #plt.savefig(s, format="png")
        with BytesIO() as buf:
            plt.savefig(buf, format='png')
            buf.seek(0)
    
            if self.SummaryImage is None:
                # Create an Image object
                img_sum = tf.Summary.Image(encoded_image_string=buf.getvalue(), width=GraphConsts.DEFAULT_WIDTH, height=GraphConsts.DEFAULT_HEIGHT)
                # Create a Summary value
                im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, 1), image=img_sum))
                # Create and write Summary
                self.SummaryImage = tf.Summary(value=im_summaries)
            
            self.TensorboardWriter.add_summary(self.SummaryImage, self.StepNumber)
            self.TensorboardWriter.flush()
            buf.close() 
    #------------------------------------------------------------------------------------
    def __setEpochTimings(self, p_nEpochTimings):
        if p_nEpochTimings is not None:
            self.EpochTimings[self.EpochIndex,:] = p_nEpochTimings[:]
            self.EpochRecallTime[self.EpochIndex] = p_nEpochTimings[0] + np.sum(p_nEpochTimings[2:])
            self.EpochTotalTime[self.EpochIndex] = np.sum(p_nEpochTimings)
    #------------------------------------------------------------------------------------
    def AddBatchAnalysis(self, p_nClassRecall, p_nClassPrecision, p_nClassF1Score, p_nValidIndices, p_oSummary):
        self.TrainClassRecall[self.EpochIndex, self.EpochStepIndex,:] = p_nClassRecall[:]
        self.TrainClassPrecision[self.EpochIndex, self.EpochStepIndex,:] = p_nClassPrecision[:]
        self.TrainClassF1Score[self.EpochIndex, self.EpochStepIndex,:] = p_nClassF1Score[:]
        self.TrainClassValidIndices[self.EpochIndex, self.EpochStepIndex,:] = p_nValidIndices[:]
        if self.IsBatchAnalysis:
            self.VerboseTraining(p_oSummary)
            self.EpochStepIndex += 1
    #------------------------------------------------------------------------------------
    def AddBatch(self, p_nError, p_nAccuracy, p_nTopKAccuracy, p_nAvgF1Score, p_oSummary):
        self.StepNumber = self.StepNumber + 1
        self.TrainError[self.EpochIndex, self.EpochStepIndex] = p_nError
        self.TrainAccuracy[self.EpochIndex, self.EpochStepIndex] = p_nAccuracy
        self.TrainTopKAccuracy[self.EpochIndex, self.EpochStepIndex] = p_nTopKAccuracy
        self.TrainStepAvgF1Score[self.EpochIndex, self.EpochStepIndex] = p_nAvgF1Score
        self.CurrentSummary = p_oSummary
        if not self.IsBatchAnalysis:
            self.VerboseTraining(p_oSummary)
            self.EpochStepIndex += 1
    #------------------------------------------------------------------------------------
    def AddEpoch(self, p_nEpochSamples):
        self.EpochSamples[self.EpochIndex] = p_nEpochSamples
                
        nAvgTrainError     = np.mean(self.TrainError[self.EpochIndex, :self.EpochStepIndex])
        nAvgTrainAccuracy  = np.mean(self.TrainAccuracy[self.EpochIndex, :self.EpochStepIndex])
        nAvgF1Score        = np.mean(self.TrainStepAvgF1Score[self.EpochIndex, :self.EpochStepIndex]) 
        
        self.TrainAvgError[self.EpochIndex, 0]    = nAvgTrainError
        self.TrainAvgAccuracy[self.EpochIndex, 0] = nAvgTrainAccuracy
        self.TrainAvgF1Score[self.EpochIndex, 0] = nAvgF1Score
        
        if self.TrainAvgError[self.EpochIndex, 0] < self.MinTrainAvgError:
            self.MinTrainAvgError = self.TrainAvgError[self.EpochIndex, 0]
            
        if self.EpochIndex > 0:             
            self.TrainAvgError[self.EpochIndex, 1]    = self.TrainAvgError[self.EpochIndex, 0] - self.TrainAvgError[self.EpochIndex - 1, 0]
            self.TrainAvgAccuracy[self.EpochIndex, 1] = self.TrainAvgAccuracy[self.EpochIndex, 0] - self.TrainAvgAccuracy[self.EpochIndex - 1, 0]
            self.TrainAvgF1Score[self.EpochIndex, 1]  = self.TrainAvgF1Score[self.EpochIndex, 0] - self.TrainAvgF1Score[self.EpochIndex - 1, 0]
        
        if self.EpochIndex >= 9:
            idx = self.EpochIndex - 9
            self.TrainAvgError[self.EpochIndex, 2] = np.std(self.TrainAvgError[idx:idx + 10, 0])
            self.TrainAvgAccuracy[self.EpochIndex, 2] = np.std(self.TrainAvgAccuracy[idx:idx + 10, 0])
            self.TrainAvgF1Score[self.EpochIndex, 2] = np.std(self.TrainAvgF1Score[idx:idx + 10, 0])
            
        self.EpochStepIndex = 0

        if type(self).__verboseLevel >= 2:
            self.Log.Print("[%i] Epoch Samples: %d" % (self.EpochIndex + 1, self.EpochSamples[self.EpochIndex]))
    #------------------------------------------------------------------------------------
    def __addValidationMetrics(self, p_oValMetrics):
        self.Validations.append(p_oValMetrics)
        
        self.ValAccuracy[self.EpochIndex, 0] = p_oValMetrics.Recall
        self.ValAvgRecall[self.EpochIndex,0] = p_oValMetrics.Recall
        self.ValAvgPrecision[self.EpochIndex,0] = p_oValMetrics.Precision
        self.ValAvgF1Score[self.EpochIndex,0] = p_oValMetrics.F1Score
        
        if self.EpochIndex > 0:
            self.ValAccuracy[self.EpochIndex, 1] = p_oValMetrics.Recall -  - self.ValAccuracy[self.EpochIndex - 1, 0]

            self.ValAvgRecall[self.EpochIndex,1] = p_oValMetrics.Recall - self.ValAvgRecall[self.EpochIndex - 1, 0]
            self.ValAvgPrecision[self.EpochIndex,1] = p_oValMetrics.Precision - self.ValAvgPrecision[self.EpochIndex - 1, 0]
            self.ValAvgF1Score[self.EpochIndex,1] = p_oValMetrics.F1Score - self.ValAvgF1Score[self.EpochIndex - 1, 0]            
        
        
        self.ValClassTrue[self.EpochIndex,:] = p_oValMetrics.ClassTrue[:]
        self.ValClassFalse[self.EpochIndex,:] = p_oValMetrics.ClassFalse[:]
        self.ValClassActual[self.EpochIndex,:] = p_oValMetrics.ClassActual[:]
        self.ValClassRecall[self.EpochIndex,:] = p_oValMetrics.ClassRecall[:]
        self.ValClassPrecision[self.EpochIndex,:] = p_oValMetrics.ClassPrecision[:]
        self.ValClassF1Score[self.EpochIndex,:] = p_oValMetrics.ClassF1Score[:]
    #------------------------------------------------------------------------------------
    def __marginBasedOverfitRate(self, p_nTrnError, p_nValError, p_nTnAccuracy, p_nValAccuracy):
        nErrorWeight=0.1
        nAccuracyWeight=0.9
        nErrorInbalance = - (p_nTrnError - p_nValError) / p_nTrnError
        nAccuracyInbalance = (p_nTnAccuracy - p_nValAccuracy) / p_nTnAccuracy
        Result = nErrorInbalance * nErrorWeight + nAccuracyInbalance *nAccuracyWeight
        return Result, nErrorInbalance, nAccuracyInbalance
    #------------------------------------------------------------------------------------
    def __overfitMarginNew(self, p_nTraiError, p_nValError, p_nTrainF1Score, p_nValF1Score):
        # Increased profit comes from training loss that continues to increase, while validation loss reaches a minimum
        nErrorMargin    = (p_nValError - p_nTraiError) / p_nValError
        # Increased profit comes from increasing training accuracy, while validation accuracy stops increasing.
        nAccuracyMargin = (p_nTrainF1Score - p_nValF1Score) / p_nTrainF1Score
        
        # Overit rate of error is reduced when the overfit rate of accuracy is smaller
        nOverfitRate = nErrorMargin * 0.4 + nAccuracyMargin * 0.6
        
        return nOverfitRate, nErrorMargin, nAccuracyMargin
    #------------------------------------------------------------------------------------        
    def AddValidation(self, p_nValError, p_oValMetrics, p_oSummary, p_nLearning_rate=0):
        self.__addValidationMetrics(p_oValMetrics)
        
        # Uses the new overfit margin calculation to replace the overfit_rate
        if self.IsNewOverfitMargin:
            overfit_rate, overfit_err_pc, overfit_acc_pc = self.__overfitMarginNew(
                  self.TrainAvgError[self.EpochIndex, 0]
                , p_nValError
                , self.TrainAvgF1Score[self.EpochIndex, 0]
                , p_oValMetrics.F1Score
            )
        else:
            overfit_rate, overfit_err_pc, overfit_acc_pc = self.__marginBasedOverfitRate(
                      self.TrainAvgError[self.EpochIndex, 0]
                    , p_nValError
                    , self.TrainAvgAccuracy[self.EpochIndex, 0]
                    , p_oValMetrics.Recall
                )            
                
            

        if self.EpochIndex > 0:
            delta_err = p_nValError - self.ValError[self.EpochIndex - 1, 0]
            #delta_acc = p_oValMetrics.Recall - self.ValAccuracy[self.EpochIndex - 1, 0]
            delta_learning_rate = p_nLearning_rate - self.LearningRate[self.EpochIndex - 1, 0 ]
            
            delta_overfit_err = overfit_err_pc - self.ValOverfitError[self.EpochIndex - 1, 0]
            delta_overfit_acc = overfit_acc_pc - self.ValOverfitAccuracy[self.EpochIndex - 1, 0]
            delta_overfit_rate = overfit_rate - self.ValOverfitRate[self.EpochIndex - 1, 0 ]
        else:
            delta_err = 0.0
            #delta_acc = 0.0      
            delta_learning_rate = 0.0
            
            delta_overfit_err = 0.0  
            delta_overfit_acc = 0.0
            delta_overfit_rate = 0.0
            
        if p_nValError < self.MinValAvgError:
            self.MinValAvgError = p_nValError

        self.ValError[self.EpochIndex, 0] = p_nValError
        self.ValError[self.EpochIndex, 1] = delta_err
        
        self.ValAccuracyPerError[self.EpochIndex] = p_oValMetrics.Recall / p_nValError
            
        nAPEPenaltyLimit = self.LearnParams.QualityPenaltyOverfitAccuracyPCOver
        if self.IsNewOverfitMargin:
            nAPEPenalty = 0.0 
        else:
            # Penalize APE by OFR (1.0-relu(tanh(OverFitRate-PenaltyLimit))*PenaltyPercentage)
            overfit_tan = np.tanh(overfit_acc_pc - nAPEPenaltyLimit)
            nAPEPenalty = overfit_tan * (overfit_tan > 0) * nAPEPenaltyLimit

        self.ValLearningQuality[self.EpochIndex] = self.ValAccuracyPerError[self.EpochIndex] - nAPEPenalty
        # self.ValLearningQuality[self.EpochIndex]=(p_nAccuracy/p_nError)*np.exp(-overfit_rate/100.0) # Old Exponential Penalty


        if self.EpochIndex >= 9:
            idx = self.EpochIndex - 9
            self.ValError[self.EpochIndex, 2] = np.std(self.ValError[idx:idx + 10, 0])
            self.ValAccuracy[self.EpochIndex, 2] = np.std(self.ValAccuracy[idx:idx + 10, 0])
            
                    
        # //TEMP: TODO: Precision-Recall
        
        self.ValOverfitError[self.EpochIndex, 0] = overfit_err_pc
        self.ValOverfitError[self.EpochIndex, 1] = delta_overfit_err

        self.ValOverfitAccuracy[self.EpochIndex, 0] = overfit_acc_pc
        self.ValOverfitAccuracy[self.EpochIndex, 1] = delta_overfit_acc

        self.ValOverfitRate[self.EpochIndex, 0] = overfit_rate
        self.ValOverfitRate[self.EpochIndex, 1] = delta_overfit_rate
                  
        self.LearningRate[self.EpochIndex, 0] = p_nLearning_rate
        self.LearningRate[self.EpochIndex, 1] = delta_learning_rate
                        
        if p_oSummary is not None:                          
            self.TensorboardWriter.add_summary(p_oSummary, self.EpochIndex)
        
        return None
    #------------------------------------------------------------------------------------
    def IsMaxEpoch(self):
        return self.EpochIndex >= self.MaxEpoch
    #------------------------------------------------------------------------------------
    def IncEpoch(self):
        self.EpochIndex += 1
    #------------------------------------------------------------------------------------
    def VerboseTraining(self, p_oSummary):
        if self.IsVerboseStep():
            self.Log.Print("[%i]   (%03i/%03i) Training  ERR=%.5f  ACC=%.5f  TOPK=%.5f  F1=%.5f" % 
                    (   self.EpochIndex + 1
                      , self.EpochStepIndex + 1, self.Parent.Iterator.TrainingBatches
                      , self.TrainError[self.EpochIndex, self.EpochStepIndex]
                      , self.TrainAccuracy[self.EpochIndex, self.EpochStepIndex]
                      , self.TrainTopKAccuracy[self.EpochIndex, self.EpochStepIndex]
                      , self.TrainStepAvgF1Score[self.EpochIndex, self.EpochStepIndex]
                    )
                  )                

            if p_oSummary is not None:
                self.TensorboardWriter.add_summary(p_oSummary, self.StepNumber)
    #------------------------------------------------------------------------------------                      
    def PrintStatus(self, p_bIsValidating=True, p_nEpochTimings=None):
        self.__setEpochTimings(p_nEpochTimings)
                        
        if type(self).__verboseLevel >= 1:
            self.Log.Print("[%i] Training   | TERR=%.5f (%+.6f) TF1=%.3f%% (%+.3f%%) | Samples: %d" 
                           % (  self.EpochIndex + 1
                               ,self.TrainAvgError[self.EpochIndex, 0], self.TrainAvgError[self.EpochIndex, 1]
                               ,self.TrainAvgF1Score[self.EpochIndex, 0] * 100.0, self.TrainAvgF1Score[self.EpochIndex, 1] * 100.0
                               ,self.EpochSamples[self.EpochIndex]
                             ) 
                           )
            if p_bIsValidating:
                self.Log.Print("[%i] Validation | VERR=%.5f (%+.6f) VF1=%.3f%% (%+.3f%%) | Q=%.4f | OFR=%.2f%% | %s" 
                      % (self.EpochIndex + 1
                         , self.ValError[self.EpochIndex, 0], self.ValError[self.EpochIndex, 1]
                         , self.ValAvgF1Score[self.EpochIndex, 0] * 100.0, self.ValAvgF1Score[self.EpochIndex, 1] * 100.0
                         , self.ValLearningQuality[self.EpochIndex]
                         #, self.ValAccuracyPerError[self.EpochIndex]
                         , self.ValOverfitRate[self.EpochIndex, 0] * 100.0
                         #, self.ValOverfitError[self.EpochIndex, 0] * 100.0
                         #, self.ValOverfitAccuracy[self.EpochIndex, 0] * 100.0
                         , self.StopWatch.NowStr
                         )
                      )
                
            if self.LastStopCondition == tcc.DONT_STOP:
                sIndicator = "  "
            else:
                sIndicator = "++"

        if type(self).__verboseLevel >= 2:
            if p_bIsValidating:             
                self.Log.Print("      |    ")
                self.Log.Print("      |__  VRE=%.3f%% (%+.3f%%) | VPR=%.3f%% (%+.3f%%) | APE:%.4f | EOFR=%.2f%% / AOFR=%.2f%% |" % 
                                    (   self.ValAvgRecall[self.EpochIndex, 0] * 100.0
                                      , self.ValAvgRecall[self.EpochIndex, 1] * 100.0
                                      , self.ValAvgPrecision[self.EpochIndex, 0] * 100.0
                                      , self.ValAvgPrecision[self.EpochIndex, 1] * 100.0
                                      , self.ValAccuracyPerError[self.EpochIndex]
                                      , self.ValOverfitError[self.EpochIndex, 0] * 100.0
                                      , self.ValOverfitAccuracy[self.EpochIndex, 0] * 100.0
                                      )    
                               )
                oValidation = self.Validations[self.EpochIndex]
                self.Log.Print("      |__  VF1: m=%.4f s=%.4f range: %.4f <= f1 <= %.4f med=%.4f" % 
                               ( oValidation.F1ScoreMean, oValidation.F1ScoreStd, oValidation.F1ScoreMin, oValidation.F1ScoreMax, oValidation.F1ScoreMedian ) )
                
        if type(self).__verboseLevel >= 2:    
            if p_bIsValidating:            
                self.Log.Print("      |__  Health:  STD:[TERR=%.5f  TACC=%.4f%%  VERR=%.5f  VACC=%.4f%%] | CHECKS +,E,A = [%i%%|%i%%|%i%%] %s" 
                      % (
                            self.TrainAvgError[self.EpochIndex, 2]
                          , self.TrainAvgAccuracy[self.EpochIndex, 2] * 100.0
                          , self.ValError[self.EpochIndex, 2]
                          , self.ValAccuracy[self.EpochIndex, 2] * 100.0
                          , self.StopConditionHits[tcc.STOP_DELTA_VAL_ERROR_POSITIVE]
                          , self.StopConditionHits[tcc.STOP_DELTA_VAL_ERROR_LT]
                          , self.StopConditionHits[tcc.STOP_DELTA_VAL_ACCURACY_LT]
                          , sIndicator
                        ))
        
        if type(self).__verboseLevel >= 2:
            if p_nEpochTimings is not None:
                self.Log.Print("      |__  Timings: TRN:%.3f + SAV:%.3f + VAL:%.3f + FLT:%.3f + STA:%.3f = %.3f" 
                                  % (  self.EpochTimings[self.EpochIndex, 0], self.EpochTimings[self.EpochIndex, 1], self.EpochTimings[self.EpochIndex, 2]
                                      ,self.EpochTimings[self.EpochIndex, 3], self.EpochTimings[self.EpochIndex, 4], self.EpochTotalTime[self.EpochIndex]
                                    )       
                               )
        #self.CustomGraphs()
    #------------------------------------------------------------------------------------
    def CanPrintBestModels(self):
        bResult = True
        for nEpochIndex in self.BestEpochs:
            if nEpochIndex is None:
                bResult = False
                
        return bResult
    #------------------------------------------------------------------------------------
    def CheckBestModel(self):
        nBestModelsRanking = None  # PANTELIS: [2017-04-02]
        nEpochToSave = None
        nEpochToDelete = None
        
        bCanPrint = self.CanPrintBestModels()
        
        if type(self).__verboseLevel >= 3 and bCanPrint:
            oToPrint = [num + 1 for num in self.BestEpochs]
            self.Log.Print("  [>] Best epochs before check:", oToPrint)
            
        nChange = None    
        for positionBestEpoch, nEpochIndex in enumerate(self.BestEpochs):
            if nEpochIndex is None:
                nEpochIndex = 0
            
            bIsBetterQuality = self.ValLearningQuality[self.EpochIndex] > self.ValLearningQuality[nEpochIndex]
            if bIsBetterQuality:
                nBestModelsRanking = positionBestEpoch  # PANTELIS: [2017-04-02]
                nChange = abs((self.ValLearningQuality[self.EpochIndex] / self.ValLearningQuality[nEpochIndex]) - 1.0)
                self.BestEpochs = self.BestEpochs[:positionBestEpoch] + [self.EpochIndex] + self.BestEpochs[positionBestEpoch:]
                nEpochToSave = self.EpochIndex
                
                if type(self).__verboseLevel >= 2:
                    self.Log.Print("  [>] * Replacing top %i best epoch: From number %i (Q=%.4f) -> to current number %i (Q=%.4f)" % 
                          (positionBestEpoch + 1
                           , (nEpochIndex + 1)
                           , self.ValLearningQuality[nEpochIndex]
                           , (nEpochToSave + 1)
                           , self.ValLearningQuality[self.EpochIndex]
                          ))
                
                break
        
        # Keeps only best n epochs
        if len(self.BestEpochs) > self.LearnParams.NumEpochsToKeep:
            nEpochToDelete = self.BestEpochs[self.LearnParams.NumEpochsToKeep]
            self.BestEpochs = self.BestEpochs[0:self.LearnParams.NumEpochsToKeep]
            if nEpochToDelete is not None:
                if type(self).__verboseLevel >= 3:
                    self.Log.Print("  [>] * Epoch number to delete %i" % (nEpochToDelete + 1))
            
        # If the accuracy is below the save limit, dont perform any file system operations       
        if (nEpochToSave is not None) and (self.ValAccuracy[self.EpochIndex, 0] < self.LearnParams.AccuracyLowLimit):
            nEpochToSave = None
            nEpochToDelete = None
            if type(self).__verboseLevel >= 1:
                self.Log.Print("  [>] Not saving because accuracy %.2f < %.2f" % (self.ValAccuracy[self.EpochIndex, 0], self.LearnParams.AccuracyLowLimit))

        # Keeping all models above a certain limit of accuracy, as potentially the best ones
        if (nEpochToDelete is not None) and (self.ValAccuracy[self.EpochIndex, 0] >= self.LearnParams.AccuracyHighLimit):
            nEpochToDelete = None

        # Shows output to console
        if type(self).__verboseLevel >= 1 and bCanPrint:
            # Best epochs queue
            oToPrint = [num + 1 for num in self.BestEpochs]
            # Corresponding qualities to best epochs
            oQualities = [self.ValLearningQuality[num] for num in self.BestEpochs]

            self.Log.Print("  [>] Best epochs:", oToPrint, " Qualities: ", oQualities, end="")
            if nEpochToSave is not None:
                self.Log.Print("      Save=%i" % (nEpochToSave + 1), end="")
                if nChange is not None:
                    self.Log.Print(" (APE Change:%.2f%%)" % (nChange * 100.0), end="")
            if nEpochToDelete is not None:
                self.Log.Print("      Delete=%i" % (nEpochToDelete + 1), end="")
            self.Log.Print(" ")    
                
        return nBestModelsRanking, nEpochToSave, nEpochToDelete  # PANTELIS: [2017-04-02]
    #------------------------------------------------------------------------------------
    def DistanceToBestEpoch(self):
        Result = 0
        
        if self.BestEpochs[0] is not None:
            Result = self.EpochIndex - self.BestEpochs[0]
        else:
            Result = 0
            
        return Result    
    #------------------------------------------------------------------------------------
    def IsBestEpoch(self):
        return self.EpochIndex == self.BestEpochs[0]    
#==================================================================================================    
    
    
    
    
    
    
    
    
    
    
#==================================================================================================
class NNTrainingStatsByERL(NNTrainingStats):
    #------------------------------------------------------------------------------------
    def __init__(self, p_sERLString, p_sCustomBaseFolder = None):
        # Chicken-egg
        self.ERL = ERLString(p_sERLString)
        assert self.ERL.IsFull, "Only full ERL is supported. Current ERL://%s" % p_sERLString
        
        oLearnParams = NNLearnParams()
        
        #TODO: Move functionality to storage
        if p_sCustomBaseFolder is None:
            sBaseFolder = Storage.JoinPaths([BaseFolders.EXPERIMENTS_RUN, "%s-%s" % (self.ERL.DataSetName, self.ERL.Architecture)])
        else:
            sBaseFolder = p_sCustomBaseFolder
            
        sFileName = Storage.JoinPaths([  sBaseFolder
                                        ,"fold%.2d" % self.ERL.FoldNumber
                                        ,"%s" % self.ERL.ExperimentUID
                                        ,"config" 
                                        ,"learn-config-used-%s.cfg" % self.ERL.ExperimentUID
                                       ])
        
        assert Storage.IsExistingFile(sFileName), "File not found %s" % sFileName
        oSourceConfig = NNLearnConfig()
        oSourceConfig.Learn = oLearnParams
        oSourceConfig.LoadFromFile(sFileName)
        
        sDataSetPath = Storage.JoinPaths([BaseFolders.DATASETS, oSourceConfig.DataSetName])
        
        oDataSet = DataSetFactory.CreateVariation(oSourceConfig.DataSetName, sDataSetPath, oSourceConfig.DataSetVariation)
        oDataSet.Create()
        
        oIterator = oDataSet.Train.GetIterator(oSourceConfig.FoldNumber, oSourceConfig.BatchSize)
        
        oLearnParams.MiniBatches.TrainingBatches    = oIterator.TrainingBatches
        oLearnParams.MiniBatches.ValidationBatches  = oIterator.ValidationBatches
        
        #print(oLearnParams.MiniBatches.TrainingBatches)
        #print(oLearnParams.MiniBatches.ValidationBatches)
        
        
        super(NNTrainingStatsByERL, self).__init__(oLearnParams, None, None, p_nClassCount=20)
        
        
        
        #TODO: Move to storage
        sFileName = Storage.JoinPaths([  sBaseFolder
                                        ,"fold%.2d" % self.ERL.FoldNumber
                                        ,"%s" % self.ERL.ExperimentUID
                                        ,"stats" 
                                        ,"stats_%s.dat" % self.ERL.ExperimentUID
                                       ])
        
        self.FileName = sFileName
    #------------------------------------------------------------------------------------
#==================================================================================================    
    
    
    
    
    
    
    
    
    
#==================================================================================================
class NNSummaryScalars(object):
    def __init__(self, p_oParentTrainer):
        #..................... |  Instance Attributes | .................................
        self.ParentTrainer = p_oParentTrainer
        self.ParentNetwork = self.ParentTrainer.Network
        self.TrainStepsSummaryOp = None
        self.ValSummaryOp = None
        self.TrainSummaryOp = None
        #...................................................1.............................
        
    #------------------------------------------------------------------------------------
    def CreateTensors(self):
        ''' Create the summary tensors for writing into a tensorboard log '''
        tSumTrain = []
        tSumVal = []
        if False:
            with tf.name_scope("Process Monitor"):
                tSumTrain.append(tf.summary.scalar("EpochNumber"     , self.ParentTrainer.GlobalEpoch))
                tSumVal.append(tf.summary.scalar("LearningRate"      , self.ParentTrainer.TrainingLearningRate))


        with tf.name_scope("Training"):
            if self.ParentNetwork.TrainLoss is not None: 
                tSumTrain.append(tf.summary.scalar("1-CCE" , self.ParentNetwork.TrainLoss))
            if self.ParentNetwork.TrainOut.Metrics.AvgF1Score is not None:
                tSumTrain.append(tf.summary.scalar("2-F1", tf.squeeze(self.ParentNetwork.TrainOut.Metrics.AvgF1Score)))
            if self.ParentNetwork.TrainOut.Accuracy is not None:
                tSumTrain.append(tf.summary.scalar("3-Acc"     , self.ParentNetwork.TrainOut.Accuracy))
            if self.ParentNetwork.TrainOut.TopKAccuracy is not None:
                tSumTrain.append(tf.summary.scalar("4-TopKAcc" , tf.squeeze(self.ParentNetwork.TrainOut.TopKAccuracy)))
                
        with tf.name_scope("Validation"):
            if self.ParentNetwork.ValLoss is not None:
                tSumVal.append(tf.summary.scalar("1-CCE" , self.ParentNetwork.ValLoss))
            if self.ParentNetwork.ValOut.Metrics.AvgF1Score is not None:
                tSumVal.append(tf.summary.scalar("2-F1" , tf.squeeze(self.ParentNetwork.ValOut.Metrics.AvgF1Score)))
            if self.ParentNetwork.ValOut.Accuracy is not None:
                tSumVal.append(tf.summary.scalar("3-Acc" , self.ParentNetwork.ValOut.Accuracy))
            if self.ParentNetwork.ValOut.TopKAccuracy is not None:                
                tSumVal.append(tf.summary.scalar("4-TopKAcc" , tf.squeeze(self.ParentNetwork.ValOut.TopKAccuracy)))                
            
        self.TrainStepsSummaryOp = tf.summary.merge(tSumTrain)
        if len(tSumVal) > 0:
            self.ValSummaryOp = tf.summary.merge(tSumVal)   
    #------------------------------------------------------------------------------------
#==================================================================================================
    
    
    