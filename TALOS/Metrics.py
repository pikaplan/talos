# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        EVALUATION METRICS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import json
import numpy as np
#TODO: Remove dependency of sklearn, scipy
from sklearn import metrics
from TALOS.FileSystem import Storage



#---------------------------------------------------------------------------------------------------
def SimpleEncode(ndarray):
    return json.dumps(ndarray.tolist())
#---------------------------------------------------------------------------------------------------
def SimpleDecode(jsonDump):
    return np.array(json.loads(jsonDump))
#---------------------------------------------------------------------------------------------------






#==================================================================================================
class MetricsKind:
    SUPERVISED_CLASSIFICATION           = 0
    UNSUPERVISED_CLASSIFICATION         = 1
    SUPERVISED_CLASSIFICATION_EXTRA     = 2
    UNSUPERVISED_CLASSIFICATION_EXTRA   = 3
    UNSUPERVISED_CLUSTERING             = 4
    SUPERVISED_CLUSTERING               = 5
    REGRESSION                          = 6
    FUNCTION_APPROXIMATION              = 8
    RETRIEVAL                           = 10
    DETECTION                           = 12
    TRACKING                            = 14
#==================================================================================================     








#==================================================================================================
class ClassificationMetricsCalculator(object):
    __verboseLevel = 1
    
    #------------------------------------------------------------------------------------
    def __init__(self):
        #........ |  Instance Attributes | ..............................................
        # // Basic \\
        self.ClassTrue = None
        self.ClassFalse = None
        self.ClassActual = None
        
        self.ClassRecall = None
        self.ClassPrecision = None
        self.ClassF1Score = None
        
        self.Recall = None
        self.Precision = None
        self.F1Score = None    
        
        # // Extended Stats \\
        self.RecallMean = None
        self.PrecisionMean = None
        self.F1ScoreMean = None
        self.RecallMedian = None
        self.PrecisionMedian = None
        self.F1ScoreMedian = None
        self.RecallStd = None
        self.PrecisionStd = None
        self.F1ScoreStd = None
        
        self.RecallMin = None
        self.PrecisionMin = None
        self.F1ScoreMin = None        
        self.RecallMax = None
        self.PrecisionMax = None
        self.F1ScoreMax = None
        #................................................................................
    #------------------------------------------------------------------------------------
    def __calculateClassStats(self):
        self.RecallMean = np.mean(self.ClassRecall)
        self.PrecisionMean = np.mean(self.ClassPrecision)
        self.F1ScoreMean = np.mean(self.ClassF1Score)

        self.RecallMedian = np.median(self.ClassRecall)
        self.PrecisionMedian = np.median(self.ClassPrecision)
        self.F1ScoreMedian = np.median(self.ClassF1Score)
        
        self.RecallStd = np.std(self.ClassRecall)
        self.PrecisionStd = np.std(self.ClassPrecision)
        self.F1ScoreStd = np.std(self.ClassF1Score)

        self.RecallMin = np.min(self.ClassRecall)
        self.PrecisionMin = np.min(self.ClassPrecision)
        self.F1ScoreMin = np.min(self.ClassF1Score)

        self.RecallMax = np.max(self.ClassRecall)
        self.PrecisionMax = np.max(self.ClassPrecision)
        self.F1ScoreMax = np.max(self.ClassF1Score)
        
        if type(self).__verboseLevel >= 2: 
            print("Recall    Stats: Min:%.4f  Max:%4.f  Med:%.4f  Mean:%.4f  Std:%.4f" % (self.RecallMin, self.RecallMax, self.RecallMedian, self.RecallMean, self.RecallStd))               
            print("Precision Stats: Min:%.4f  Max:%4.f  Med:%.4f  Mean:%.4f  Std:%.4f" % (self.PrecisionMin, self.PrecisionMax, self.PrecisionMedian, self.PrecisionMean, self.PrecisionStd))
            print("F1Score   Stats: Min:%.4f  Max:%4.f  Med:%.4f  Mean:%.4f  Std:%.4f" % (self.F1ScoreMin, self.F1ScoreMax, self.F1ScoreMedian, self.F1ScoreMean, self.F1ScoreStd))
    #------------------------------------------------------------------------------------
    def Calculate(self, p_nTrue, p_nFalse, p_nActual):
        nTrue   = np.sum(np.asarray(p_nTrue), axis=0)
        nFalse  = np.sum(np.asarray(p_nFalse), axis=0)
        nActual = np.sum(np.asarray(p_nActual), axis=0)

        self.ClassTrue = nTrue
        self.ClassFalse = nFalse
        self.ClassActual = nActual
        
        #TODO: Confusion Matrix
        #print(nTrue)
        #print(nFalse)
        #print(nActual)
        #print(np.sum(nActual))


        self.ClassRecall     = nTrue / (nActual + 1e-7)
        self.ClassPrecision  = nTrue / (nTrue + nFalse + 1e-7)
        self.ClassF1Score    = (2.0 * ( self.ClassRecall * self.ClassPrecision) ) / ( self.ClassRecall + self.ClassPrecision + + 1e-7)
        if type(self).__verboseLevel >= 3:
            print("ClassRecall", self.ClassRecall)
            print("ClassPrecision", self.ClassPrecision)
            print("ClassF1Score", self.ClassF1Score)
            
        self.__calculateClassStats()
        
        
        #TODO: Weight by Support
        self.Recall = np.average(self.ClassRecall)
        self.Precision = np.average(self.ClassPrecision)
        self.F1Score = np.average(self.ClassF1Score) 
                
        
        if type(self).__verboseLevel >= 2:
            print("Avg Class Recall", self.Recall)
            print("Avg Class Precision", self.Precision)
            print("Avg Class F1Score", self.F1Score)
            #print("Recall:%f  Precision:%f  F1Score:%f" % (self.Recall, self.Precision, self.F1Score)) 
    #------------------------------------------------------------------------------------
                                                  
        
        
#==================================================================================================











        
#==================================================================================================
class ClassificationMetrics(object): 
    #------------------------------------------------------------------------------------
    def __init__(self):
        #........ |  Instance Attributes | ..............................................
        self.Kind = MetricsKind.SUPERVISED_CLASSIFICATION
        self.ActualClasses = None
        self.PredictedClasses = None
        self.PredictedProbsTop = None
        self.IDs = None
        self.TopCount = None
        
        
        self.Accuracy           = None
        self.TopKAccuracy       = None 
        self.AveragePrecision   = None
        self.AverageRecall      = None
        self.AverageF1Score     = None
        self.AverageSupport     = None
        
                
        self.Precision = None
        self.Recall = None
        self.F1Score = None
        self.Support = None
        
        self.ConfusionMatrix = None
        
        self.ClassCount = 0 
        #................................................................................
    #------------------------------------------------------------------------------------
    def CalculateTopK(self, p_nTopKappa, p_nTopKCorrect):
        self.TopKappa = p_nTopKappa 
        self.TopKAccuracy = np.mean(p_nTopKCorrect)
        print("TopKAccuracy", self.TopKAccuracy)
        
        #self.TopKAccuracy
    #------------------------------------------------------------------------------------
    def Calculate(self, p_nActual, p_nPredicted, p_nPredictedProbsTop=None):    
        self.ActualClasses      = p_nActual
        self.PredictedClasses   = p_nPredicted
        self.PredictedProbsTop  = p_nPredictedProbsTop
        if self.PredictedProbsTop is not None:
            self.TopCount = p_nPredictedProbsTop.shape[1]

        # Confusion matrix layout is  
        #            predicted
        #         - - - - - - - - 
        # actual |
        #        |
        self.ConfusionMatrix = metrics.confusion_matrix(self.ActualClasses, self.PredictedClasses)        

        self.Accuracy        = metrics.accuracy_score(self.ActualClasses, self.PredictedClasses)
        self.Precision, self.Recall, self.F1Score, self.Support = metrics.precision_recall_fscore_support(self.ActualClasses, self.PredictedClasses, average=None)
        self.AveragePrecision, self.AverageRecall, self.AverageF1Score, self.AverageSupport = metrics.precision_recall_fscore_support(self.ActualClasses, self.PredictedClasses,  average='weighted')
        
        self.ClassCount = self.Recall.shape[0]
    #------------------------------------------------------------------------------------
    def Save(self, p_sFileName):
        oData = {
                     "FileFormat"           : "TALOS008"
                    ,"Kind"                 : self.Kind
                    ,"IDs"                  : self.IDs
                    ,"Actual"               : self.ActualClasses
                    ,"Predicted"            : self.PredictedClasses
                    ,"PredictedProbsTop"    : self.PredictedProbsTop
                    ,"TopKappa"             : self.TopKappa
                    
                    ,"Accuracy"             : self.Accuracy
                    ,"TopKAccuracy"         : self.TopKAccuracy
                    ,"AveragePrecision"     : self.AveragePrecision                    
                    ,"AverageRecall"        : self.AverageRecall
                    ,"AverageF1Score"       : self.AverageF1Score
                    ,"AverageSupport"       : self.AverageSupport
                    #,"Top1Error"        : None
                    #,"Top5Error"        : None
                            
                    ,"ClassPrecision"       : self.Precision
                    ,"ClassRecall"          : self.Recall
                    ,"ClassF1Score"         : self.F1Score
                    ,"ClassSupport"         : self.Support
                    ,"ConfusionMatrix"      : self.ConfusionMatrix
                }
        Storage.SerializeObjectToFile(p_sFileName, oData)
    #------------------------------------------------------------------------------------
    def Load(self, p_sFileName):
        oData = Storage.DeserializeObjectFromFile(p_sFileName)
        assert oData is not None, "Evaluation results file not found %s" % p_sFileName
        self.IDs = oData["IDs"]
        self.Kind = oData["Kind"]
        self.ActualClasses = oData["Actual"] 
        self.PredictedClasses = oData["Predicted"]
        self.PredictedProbsTop = oData["PredictedProbsTop"]
        if self.PredictedProbsTop is not None:
            self.TopCount = self.PredictedProbsTop.shape[1]
        if "TopKappa" in oData:
            self.TopKappa = oData["TopKappa"]  
             
        if "Accuracy" in oData:
            self.Accuracy = oData["Accuracy"]    
        if "TopKAccuracy" in oData:
            self.TopKAccuracy = oData["TopKAccuracy"]   
        self.AveragePrecision = oData["AveragePrecision"]
        self.AverageRecall = oData["AverageRecall"]
        self.AverageF1Score = oData["AverageF1Score"]
        self.AverageSupport = oData["AverageSupport"]
        #self.Top1Error = oData["Top1Error"]
        #self.Top5Error = oData["Top5Error"]        
        self.Precision = oData["ClassPrecision"]
        self.Recall = oData["ClassRecall"]
        self.F1Score = oData["ClassF1Score"]
        self.Support = oData["ClassSupport"]
        
        self.ConfusionMatrix = oData["ConfusionMatrix"]

        self.ClassCount = self.Recall.shape[0]
    #------------------------------------------------------------------------------------
    def MissClassified(self, p_nClassIndex):
        oIDs = []
        for nIndex, nActual in enumerate(self.ActualClasses):
            if (nActual == p_nClassIndex) and (nActual != self.PredictedClasses[nIndex]):
                oIDs.append(self.IDs[nIndex])
        return oIDs
    #------------------------------------------------------------------------------------
#==================================================================================================
















 
 
#==================================================================================================
class ClassificationBest(object):
    #------------------------------------------------------------------------------------
    def __init__(self, p_sEvaluationResultsFolder=None):
        #........ |  Instance Attributes | ..............................................
        self.Folder = p_sEvaluationResultsFolder
        self.ResultFiles = None

        self.KeepEpochs         = 3
        self.TopCount           = 5
        
        self.EpochNumber=None
        self.FileNames=None
        self.Accuracy=None
        self.Recall=None
        self.Precision=None
        self.F1Score=None
        self.Points=None
        self.ClassCount=None
        self.IsBinary=False
        
        
        self.BestIndexes                = None
        
        # // Persistent Data \\
        self.BestEpochs                 = None
        
        self.Points                     = None
        self.Recall                     = None
        self.Precision                  = None
        self.F1Score                    = None
        self.CrossF1Score               = None
        self.ObjectiveF1Score           = None
        self.PositiveF1Score            = None
        
        self.BestPoints                 = None
        self.BestRecall                 = None
        self.BestPrecision              = None
        self.BestF1Score                = None
        self.BestCrossF1Score           = None
        self.BestObjectiveF1Score       = None
        self.BestPositiveF1Score        = None
                                    
        self.DiscardedEpochs            = None
        self.BestRecallEpochs           = None
        self.BestPrecisionEpochs        = None
        self.BestF1ScoreEpochs          = None
        self.BestCrossF1ScoreEpochs     = None
        self.BestObjectiveF1ScoreEpochs = None
        self.BestPositiveScoreEpochs    = None
        #................................................................................
        if self.Folder is not None:
            self.__listFiles()
    #------------------------------------------------------------------------------------
    def Save(self, p_sFileName):
        oData = {
                     "FileFormat"               : "TALOS008"
                    ,"IsBinary"                 : self.IsBinary
                    
                    ,"EpochNumber"              : self.EpochNumber
                    ,"FileNames"                : self.FileNames
                    ,"Accuracy"                 : self.Accuracy
                    ,"Recall"                   : self.Recall
                    ,"Precision"                : self.Precision
                    ,"F1Score"                  : self.F1Score
                    ,"CrossF1Score"             : self.CrossF1Score
                    ,"ObjectiveF1Score"         : self.ObjectiveF1Score
                    ,"PositiveF1Score"          : self.PositiveF1Score
                    
                    ,"BestEpochs"               : self.BestEpochs
                    ,"BestPoints"               : self.BestPoints
                    ,"BestRecall"               : self.BestRecall
                    ,"BestPrecision"            : self.BestPrecision
                    ,"BestF1Score"              : self.BestF1Score
                    ,"BestCrossF1Score"         : self.BestCrossF1Score
                    ,"BestObjectiveF1Score"     : self.BestObjectiveF1Score
                    ,"BestPositiveF1Score"      : self.BestPositiveF1Score
                            
                    ,"DiscardedEpochs"          : self.DiscardedEpochs
                    ,"BestRecallEpochs"         : self.BestRecallEpochs                    
                    ,"BestPrecisionEpochs"      : self.BestPrecisionEpochs                    
                    ,"BestF1ScoreEpochs"        : self.BestF1ScoreEpochs                    
                    ,"BestCrossF1ScoreEpochs"   : self.BestCrossF1ScoreEpochs    
                    ,"BestObjectiveF1ScoreEpochs": self.BestObjectiveF1ScoreEpochs                
                    ,"BestPositiveScoreEpochs"  : self.BestPositiveScoreEpochs                    
                }
        Storage.SerializeObjectToFile(p_sFileName, oData, p_bIsOverwritting=True)
    #------------------------------------------------------------------------------------
    def IndexOfEpoch(self, p_nEpochNumber):
        nFoundPos = None
        for nIndex, nEpochNumber in enumerate(self.EpochNumber):
            if nEpochNumber == p_nEpochNumber:
                nFoundPos = nIndex
                break
            
        return nFoundPos
    #------------------------------------------------------------------------------------
    def GetBestIndex(self):
        return self.IndexOfEpoch( self.BestEpochs[0] )
    #------------------------------------------------------------------------------------
    def Load(self, p_sFileName):
        oData = Storage.DeserializeObjectFromFile(p_sFileName, p_bIsVerbose=False)
        assert oData is not None, "File %s not found" % p_sFileName
        self.BestEpochs = oData["BestEpochs"]
        self.IsBinary = oData["IsBinary"]
        
        self.EpochNumber = oData["EpochNumber"]
        self.FileNames = oData["FileNames"]
        self.Accuracy = oData["Accuracy"]
        self.Recall = oData["Recall"]
        self.Precision = oData["Precision"]
        self.F1Score = oData["F1Score"]
        self.CrossF1Score = oData["CrossF1Score"]
        if "ObjectiveF1Score" in oData:
            self.ObjectiveF1Score = oData["ObjectiveF1Score"]
        self.PositiveF1Score = oData["PositiveF1Score"]
                            
        self.BestPoints = oData["BestPoints"]
        self.BestRecall = oData["BestRecall"]
        self.BestPrecision = oData["BestPrecision"]
        self.BestF1Score = oData["BestF1Score"]
        self.BestCrossF1Score = oData["BestCrossF1Score"]
        if "BestObjectiveF1Score" in oData:
            self.BestObjectiveF1Score = oData["BestObjectiveF1Score"]
        
        self.BestPositiveF1Score = oData["BestPositiveF1Score"]
        
        self.DiscardedEpochs = oData["DiscardedEpochs"]
        self.BestRecallEpochs = oData["BestRecallEpochs"]
        self.BestPrecisionEpochs = oData["BestPrecisionEpochs"]
        self.BestF1ScoreEpochs = oData["BestF1ScoreEpochs"]
        self.BestCrossF1ScoreEpochs = oData["BestCrossF1ScoreEpochs"]
        if "BestObjectiveF1ScoreEpochs" in oData:
            self.BestObjectiveF1ScoreEpochs = oData["BestObjectiveF1ScoreEpochs"]
        
        self.BestPositiveScoreEpochs = oData["BestPositiveScoreEpochs"]
    #------------------------------------------------------------------------------------
    def __listFiles(self):
        sEvaluationResultFiles = Storage.GetFilesSorted(self.Folder)
        self.FileNames = []
        self.ResultFiles = []
        for sFile in sEvaluationResultFiles:
            sFileNameFull = Storage.JoinPath(self.Folder, sFile)
            self.FileNames.append(sFileNameFull)
            self.ResultFiles.append( [sFile, sFileNameFull])
        
        nFileCount = len(self.ResultFiles)
        
        self.EpochNumber=np.zeros((nFileCount), np.float32 )
        self.Accuracy=np.zeros((nFileCount), np.float32 )
        self.Recall=np.zeros((nFileCount), np.float32 ) 
        self.Precision=np.zeros((nFileCount), np.float32 )
        self.F1Score=np.zeros((nFileCount), np.float32 )
        self.Points=np.zeros((nFileCount), np.float32 )
        self.CrossF1Score=np.zeros((nFileCount), np.float32 )
        self.ObjectiveF1Score=np.zeros((nFileCount), np.float32 )
        self.PositiveF1Score=np.zeros((nFileCount), np.float32 )        
    #------------------------------------------------------------------------------------
    def __loadAll(self):
        for nIndex, sFileRec in enumerate(self.ResultFiles):
            _, sEpochNumber, _ = Storage.SplitFileName(sFileRec[0])
            sFileNameFull = sFileRec[1]
            
            
            oMetrics = ClassificationMetrics()
            oMetrics.Load(sFileNameFull)

            print("Accuracy:%f  Top%dAccuracy%s" % (oMetrics.Accuracy, oMetrics.TopKappa, oMetrics.TopKAccuracy))
            
            self.EpochNumber[nIndex] = int(sEpochNumber)
            self.Accuracy[nIndex] = oMetrics.Accuracy
            self.Recall[nIndex] = oMetrics.AverageRecall  
            self.Precision[nIndex] = oMetrics.AveragePrecision
            self.F1Score[nIndex] = oMetrics.AverageF1Score
            if oMetrics.ClassCount == 2:
                self.IsBinary=True
                # Cross entropy of the F1 scores for binary classification.
                self.CrossF1Score[nIndex] = -(oMetrics.F1Score[0]*np.log10(oMetrics.F1Score[1])+ oMetrics.F1Score[1]*np.log10(oMetrics.F1Score[0]))
                self.ObjectiveF1Score[nIndex] = self.F1Score[nIndex] / self.CrossF1Score[nIndex]
                # Special binary classification, with the class 0 the class positives
                self.PositiveF1Score[nIndex] = oMetrics.F1Score[0]
                                            
                print(sEpochNumber, oMetrics.F1Score[0], oMetrics.F1Score[1], self.CrossF1Score[nIndex], self.ObjectiveF1Score[nIndex])
                
    #------------------------------------------------------------------------------------
    def _topKIndexes(self, p_nArray, p_nK):
        #print(np.sort(p_nArray))
        #print(np.argsort(p_nArray)[-p_nK:])
        nResult = []
        for nIndex in np.argsort(p_nArray)[-p_nK:]:
            nResult.insert(0, nIndex)

        return nResult
    #------------------------------------------------------------------------------------
    def _bottomKIndexes(self, p_nArray, p_nK):
        #print(np.sort(p_nArray))
        #print(np.argsort(p_nArray)[-p_nK:])
        nResult = []
        for nIndex in np.argsort(p_nArray)[:p_nK]:
            nResult.insert(0, nIndex)

        return nResult    
    #------------------------------------------------------------------------------------
    def _mergeIdx(self, p_nFirstList, p_nSecondList):
        return p_nFirstList + list(set(p_nSecondList) - set(p_nFirstList))
    #------------------------------------------------------------------------------------
    def Run(self):
        #Using point system from F1
        POINT_SYSTEM = [25,18,15,12,10,8,6,4,2,1]
        nTopCount = self.TopCount
        nTopPoints=np.asarray(POINT_SYSTEM[:nTopCount], np.float32)
        
        self.__loadAll()
        
        nTopRecallIdx = self._topKIndexes(self.Recall, nTopCount)
        nTopPrecisionIdx = self._topKIndexes(self.Precision, nTopCount)
        nTopF1ScoreIdx = self._topKIndexes(self.F1Score, nTopCount)
        if self.IsBinary:
            nTopCrossF1ScoreIdx = self._bottomKIndexes(self.CrossF1Score, nTopCount)
            nTopObjectiveF1ScoreIdx = self._topKIndexes(self.ObjectiveF1Score, nTopCount)
            nTopPositiveF1ScoreIdx = self._topKIndexes(self.PositiveF1Score, nTopCount)
        
        print("--------------------------")
        print("Top %d Recall                  "   % nTopCount, self.EpochNumber[nTopRecallIdx], self.Recall[nTopRecallIdx])
        print("Top %d Precision               " % nTopCount, self.EpochNumber[nTopPrecisionIdx], self.Precision[nTopPrecisionIdx])
        print("Top %d F1Score                 " % nTopCount  , self.EpochNumber[nTopF1ScoreIdx], self.F1Score[nTopF1ScoreIdx])
        if self.IsBinary:
            print("Top %d Cross Entropy of F1Score" % nTopCount  , self.EpochNumber[nTopCrossF1ScoreIdx], self.CrossF1Score[nTopCrossF1ScoreIdx])
            print("Top %d Object F1 Ratio         " % nTopCount  , self.EpochNumber[nTopCrossF1ScoreIdx], self.ObjectiveF1Score[nTopObjectiveF1ScoreIdx])
            print("Top %d F1Score for Positives   " % nTopCount     , self.EpochNumber[nTopPositiveF1ScoreIdx], self.PositiveF1Score[nTopPositiveF1ScoreIdx])
        
        if self.IsBinary:
            self.Points[nTopObjectiveF1ScoreIdx] += nTopPoints

            self.Points[nTopCrossF1ScoreIdx]    += nTopPoints / 2
            self.Points[nTopPositiveF1ScoreIdx] += nTopPoints / 8
            
            self.Points[nTopRecallIdx]       += (nTopPoints / 4)
            self.Points[nTopPrecisionIdx]    += (nTopPoints / 4)        
            self.Points[nTopF1ScoreIdx]      += nTopPoints / 2
        else:
            self.Points[nTopRecallIdx]       += (nTopPoints / 2)
            self.Points[nTopPrecisionIdx]    += (nTopPoints / 2)        
            self.Points[nTopF1ScoreIdx]      += (nTopPoints )
                        
        
        if self.IsBinary:
            nTopIdx =   self._mergeIdx(  nTopPositiveF1ScoreIdx, self._mergeIdx( nTopCrossF1ScoreIdx, self._mergeIdx(nTopF1ScoreIdx, self._mergeIdx(nTopRecallIdx, nTopPrecisionIdx)) )  )
        else:
            nTopIdx =   self._mergeIdx( nTopF1ScoreIdx, self._mergeIdx(nTopRecallIdx, nTopPrecisionIdx) )            
        
        
        print("--------------------------")
        print("epochs"      , self.EpochNumber[nTopIdx])
        print("points"      , self.Points[nTopIdx])
        if self.IsBinary:
            print("objective f1 ratio", self.ObjectiveF1Score[nTopIdx])
            print("cross f1"    , self.CrossF1Score[nTopIdx])
            print("positive f1" , self.PositiveF1Score[nTopIdx])
        print("recall"      , self.Recall[nTopIdx])
        print("precision"   , self.Precision[nTopIdx])
        print("f1"          , self.F1Score[nTopIdx])
        print("--------------------------")
        
        
        nBestIndexes = self._topKIndexes(self.Points, self.KeepEpochs)
        nBestEpochs = self.EpochNumber[nBestIndexes]
        
        self.BestIndexes        = nBestIndexes
        self.BestEpochs         = np.asarray(nBestEpochs, dtype=np.int32)
        
        self.BestPoints             = self.Points[nBestIndexes]
        self.BestRecall             = self.Recall[nBestIndexes]
        self.BestPrecision          = self.Precision[nBestIndexes]
        self.BestF1Score            = self.F1Score[nBestIndexes]
        self.BestCrossF1Score       = self.CrossF1Score[nBestIndexes]
        self.BestObjectiveF1Score   = self.ObjectiveF1Score[nBestIndexes]
        self.BestPositiveF1Score    = self.PositiveF1Score[nBestIndexes]
        
        self.DiscardedEpochs        = np.asarray(list(set(self.EpochNumber) - set(nBestEpochs)), dtype=np.int32)
        self.BestRecallEpochs       = self.EpochNumber[nTopRecallIdx]
        self.BestPrecisionEpochs    = self.EpochNumber[nTopPrecisionIdx]
        self.BestF1ScoreEpochs      = self.EpochNumber[nTopF1ScoreIdx]
        if self.IsBinary:
            self.BestCrossF1ScoreEpochs = self.EpochNumber[nTopCrossF1ScoreIdx]
            self.BestObjectiveF1ScoreEpochs = self.EpochNumber[nTopObjectiveF1ScoreIdx]
            self.BestPositiveScoreEpochs = self.EpochNumber[nTopPositiveF1ScoreIdx]
    #------------------------------------------------------------------------------------
    def ExportToText(self, p_sTextFileName, p_oExperiment=None):
        bIsAppending = p_oExperiment is not None
        
        if bIsAppending:
            sLearningConfigLines = Storage.ReadTextFile(p_oExperiment.RunSub.LearnConfigUsedFileName)
            sLearningLogLines = Storage.ReadTextFile(p_oExperiment.RunSub.LogFileName)
        
        # Dumps the class folders to a text file            
        with open(p_sTextFileName, "w") as oOutFile:
            print("="*80, file=oOutFile)
            print("epochs        :"   , self.BestEpochs, file=oOutFile)
            print("points        :" , self.BestPoints, file=oOutFile)
            if self.IsBinary:
                print("objective f1 ratio", self.BestObjectiveF1Score, file=oOutFile)
                print("cross f1      :"   , self.BestCrossF1Score, file=oOutFile)
                print("positive f1   :" , self.BestPositiveF1Score * 100, file=oOutFile)
                
            print("recall        :"   , self.BestRecall * 100, file=oOutFile)
            print("precision     :"   , self.BestPrecision * 100, file=oOutFile)
            print("f1 score      :"   , self.BestF1Score * 100, file=oOutFile)
              
            if bIsAppending:
                # Appends the related configuration that generated the results
                print("-"*80, file=oOutFile)
                for sLine in sLearningConfigLines:
                    print(sLine, file=oOutFile)
                print("="*80, file=oOutFile)
                # Appends the log at the end of the best models text file
                for sLine in sLearningLogLines:
                    print(sLine, file=oOutFile)
                print("-"*80, file=oOutFile)
                
            print("="*80, file=oOutFile)
    #------------------------------------------------------------------------------------
        
#==================================================================================================    










# 
# #==================================================================================================
# class ExperimentEvaluation(object):
#     __verboseLevel = 1
#     
#     #------------------------------------------------------------------------------------
#     def __init__(self, p_oExperiment):
#         #........ |  Instance Attributes | ..............................................
#         self.Experiment = p_oExperiment
#         self.EpochModels=[]
#         self.WinnerModels=[
#         self.DiscardedModels=[]     
#         #TEMP
#         self.IsDeletingDiscardedModels=True   
#         self.ExistModels=True
#         #................................................................................
#         assert self.Experiment is not None, "Neural network should be associated with an experiment"
#     #------------------------------------------------------------------------------------
#     def Evaluate(self):
#         self.DoOnExperimentEvaluate()
#     #------------------------------------------------------------------------------------
#     def DoOnExperimentEvaluate(self):
#         pass
#     #------------------------------------------------------------------------------------
#     def Save(self):
#         print("[>] Saving evaluation results for experiment %s run %s fold %i, started at %s." % 
#               (  self.Experiment.BaseTitle, self.Experiment.MinuteUID.UID
#                , self.Experiment.FoldNumber, self.Experiment.Minute.strftime(taCC.ISO_DATE_FMT_MINUTE)))
# 
#         if False: #TEMP: Proven ineffective for large models like ZFNet     
#             if self.Experiment.ExistsModel(taCC.C_MODEL_FILE_NAME_INITIAL):
#                 if type(self).__verboseLevel >=1 :
#                     print("  |__ Compressing initial weights: %s" % (self.Experiment.ExperimentModelFolder + taCC.C_MODEL_FILE_NAME_INITIAL))
#                 self.Experiment.CompressModel(taCC.C_MODEL_FILE_NAME_INITIAL)
#                 self.Experiment.DeleteModel(taCC.C_MODEL_FILE_NAME_INITIAL, True)
#         
#         # Compresses the winner models
#         for nIndex,oModel in enumerate(self.WinnerModels):
#             if oModel is not None:
#                 sModelName = oModel.ModelName
#                 
#                 if type(self).__verboseLevel >=1 :
#                     print("  |__ Compressing epoch winner model: %s" % (self.Experiment.ExperimentModelFolder + sModelName))
#                     
#                 sPickleFileName = "%s_k%02i_" % (self.Experiment.BaseTitle, self.Experiment.FoldNumber) + taCC.C_BASE_FILE_NAME_PREFIX + sModelName + ".pkl"
#                 Storage.SerializeObjectToFile(self.Experiment.ExperimentResultsFolder + sPickleFileName, oModel)
#                 Storage.SerializeObjectToFile(self.Experiment.ExperimentWinnersFolder + sPickleFileName, oModel)
#                 if False: #TEMP: Proven ineffective for large models like ZFNet
#                     self.Experiment.CompressModel(sModelName)
#                     
#                 if nIndex == 0:
#                     self.AppendToFoldBestsFile(oModel)
#             
#             
#         # Discards the rest of the models    
#         for oModel in self.DiscardedModels:
#             sModelName = oModel.ModelName
#             
#             if type(self).__verboseLevel >=1:
#                 print("  |__ Discarding epoch model: %s" % (self.Experiment.ExperimentModelFolder + sModelName))
#             sPickleFileName = "%s_k%02i_" % (self.Experiment.BaseTitle, self.Experiment.FoldNumber) + taCC.C_BASE_FILE_NAME_PREFIX + sModelName + ".pkl"                
#             Storage.SerializeObjectToFile(self.Experiment.ExperimentResultsFolder + sPickleFileName, oModel)
#             if self.IsDeletingDiscardedModels:
#                 self.Experiment.DeleteModel(sModelName)
#     #------------------------------------------------------------------------------------
#     def AppendToFoldBestsFile(self, p_oChampionModel):
#         oExperiment = self.Experiment
#             
#             
#         with open(oExperiment.ExperimentWinnersFolder + taCC.C_EXPERIMENT_FOLDS_BESTS_FILE_NAME, "a") as oFile:
#             # Write the header on the first fold
#             if oExperiment.FoldNumber == 1:
#                 sHeader="Title;Fold;EpochModel;Accuracy;Recall;Precision;F1Score"
#                 print(sHeader, file=oFile)
#                 
#             sLine = "%s;%i;%s;%.2f;%.2f;%.2f;%.2f" % (
#                              oExperiment.Title
#                             ,oExperiment.FoldNumber
#                             ,p_oChampionModel.ModelName
#                             ,p_oChampionModel.Accuracy * 100.0
#                             ,p_oChampionModel.AverageRecall * 100.0
#                             ,p_oChampionModel.AveragePrecision * 100.8
#                             ,p_oChampionModel.AverageF1Score * 100.8
#                         )
#             print(sLine, file=oFile)
#             oFile.close()
#                    
# #==================================================================================================
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         
# 
#     
# #==================================================================================================
# class ClassifierEvaluation(ExperimentEvaluation):
#     #------------------------------------------------------------------------------------
#     def __init__(self, p_oNetwork):
#         #........ |  Instance Attributes | ..............................................
#         self.Network    = p_oNetwork
#         self.NoModelsFound = False
#         #................................................................................
#         super(ClassifierEvaluation, self).__init__(p_oNetwork.Experiment)
#     #------------------------------------------------------------------------------------
#     def DoOnExperimentEvaluate(self):
#         self.EpochModels = []
#         sEpochModelNames = self.Experiment.ListSavedModels()
#         nCount = len(sEpochModelNames)
#         for nIndex, sModelName in enumerate(sEpochModelNames):
#             print("[>] Evaluating saved model %i/%i" % (nIndex + 1, nCount))
#             self.Network.RestoreWeights(self.Experiment.ExperimentModelFolder + sModelName)
#             nActual,nPredicted = self.Network.Predict()
#   
#             #//TODO: Better encapsulation in method call
#             self.Network.InputQueue.RecallQueueFeeder.ContinueToFeed()  
#             
#             oMetrics = ClassifierMetrics(sModelName)
#             self.EpochModels.append(oMetrics)
#             oMetrics.Calculate(nActual, nPredicted)
#               
#         self.DeterineWinners()                    
#     #------------------------------------------------------------------------------------
#     def DeterineWinners(self):
#         print("[>] Determining winner models")
#         # One top model for each corresponding metric
#         oTopModelsForMetric=[None]*4
# 
#         self.AveragePrecision   = None
#         self.AverageRecall      = None
#         self.AverageF1Score     = None
#         self.AverageSupport     = None
#         
#         # The following represents the priority order of the winner models, best of all is the one with best F1 score
#         nTopF1Score=0
#         nTopAccuracy=0
#         nTopRecall=0
#         nTopPrecision=0
#         for oModel in self.EpochModels:
#             if oModel.AverageF1Score > nTopF1Score:
#                 nTopF1Score = oModel.AverageF1Score
#                 oTopModelsForMetric[0] = oModel
#             
#             if oModel.Accuracy > nTopAccuracy:
#                 nTopAccuracy = oModel.Accuracy
#                 oTopModelsForMetric[1] = oModel
# 
#             if oModel.AverageRecall > nTopRecall:
#                 nTopRecall = oModel.AverageRecall
#                 oTopModelsForMetric[2] = oModel
#                 
#             if oModel.AveragePrecision > nTopPrecision:
#                 nTopPrecision = oModel.AveragePrecision
#                 oTopModelsForMetric[3] = oModel
# 
#             print("  |__ [%s] Recall=%.2f; Precision=%.2f; F1 Score=%.2f" % ( 
#                      oModel.ModelName
#                     ,oModel.AverageRecall * 100.0
#                     ,oModel.AveragePrecision * 100.0
#                     ,oModel.AverageF1Score * 100.0
#                  ))
#                 
#                 
#         # Consolidates the top performing models for all metrics
#         self.WinnerModels = []
#         for oTopModel in oTopModelsForMetric:
#             if not (oTopModel in self.WinnerModels):
#                 self.WinnerModels.append(oTopModel)
#         
#         self.DiscardedModels = []
#         for oModel in self.EpochModels:
#             if not (oModel in self.WinnerModels):
#                 self.DiscardedModels.append(oModel)
#                 
#         
#         bNoModelsFound=False
#         if self.WinnerModels is None:
#             bNoModelsFound = True
#         else:
#             if (len(self.WinnerModels) == 0):
#                 bNoModelsFound = True
#             elif (len(self.WinnerModels) == 1):
#                 print(self.WinnerModels)
#                 if self.WinnerModels[0] is None:
#                     bNoModelsFound = True
#         self.NoModelsFound = bNoModelsFound
#         
#         if self.NoModelsFound:
#             print("  |__ No models were saved during the training process")
#         else:
#             self.PrintWinners()
#                         
#             
#     #------------------------------------------------------------------------------------
#     def PrintWinners(self):
#         for nIndex, oWinnerModel in enumerate(self.WinnerModels):
#             print("  |__ Winner #%i: [%s] Recall=%.2f; Precision=%.2f; F1 Score=%.2f" % ( 
#                      nIndex + 1
#                     ,oWinnerModel.ModelName
#                     ,oWinnerModel.AverageRecall * 100.0
#                     ,oWinnerModel.AveragePrecision * 100.0
#                     ,oWinnerModel.AverageF1Score * 100.0
#                  ))
#     #------------------------------------------------------------------------------------    
# #==================================================================================================
#     
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
#         
#         
# 
# 
# 
# 
# 
# 
# 
# 
# 
# #==================================================================================================
# class ClassificationEvaluator(object):
#     #/TEMP: Still in 0.1-ALPHA
#     PredictedClasses = None
#     ActualClasses = None
#     VerboseLevel=1
#     
#     
#     ConfusionMatrix = None
#     ClassPrecision = None
#     ClassRecall = None
#     ClassF1Score = None
#     ClassSupport = None
#     
#     Accuracy = None
#     AveragePrecision=None
#     AverageRecall=None
#     AverageF1Score=None
#     AverageSupport=None
#     
#     WinnerAccuracy=[0.0]*2
#     WinnerPrecision=[0.0]*2
#     WinnerRecall=[0.0]*2
#     WinnerF1Score=[0.0]*2
#     
#     
#     #------------------------------------------------------------------------------------
#     def __init__(self, p_nActualClasses=None, p_nPredictedClasses=None):
#         self.ActualClasses=p_nActualClasses
#         self.PredictedClasses=p_nPredictedClasses
#     #------------------------------------------------------------------------------------
#     def Reset(self):
#         self.PredictedClasses = None
#         self.ActualClasses = None
#             
#         self.WinnerAccuracy[:]=0.0
#         self.WinnerPrecision[:]=0.0
#         self.WinnerRecall[:]=0.0
#         self.WinnerF1Score[:]=0.0
#     #------------------------------------------------------------------------------------    
#     def Evaluate(self, p_nActualClasses=None, p_nPredictedClasses=None, p_sModelName=None):
#         # Replaces the actual and predicted class numbers, if those provided in the method params
#         if p_nActualClasses is not None:
#             self.ActualClasses=p_nActualClasses
#         if p_nPredictedClasses is not None:        
#             self.PredictedClasses=p_nPredictedClasses    
#         
#         # In any case there should be actual and predicted class number
#         if (self.ActualClasses is None) or (self.PredictedClasses is None):
#             raise "Please provide actual and predicted class numbers"
# 
#         self.Accuracy=metrics.accuracy_score(self.ActualClasses, self.PredictedClasses)
#         self.ConfusionMatrix = metrics.confusion_matrix(self.ActualClasses, self.PredictedClasses)
#         
#         # class_recall = metrics.recall_score(val_true, val_pred, average=None)
#         # class_precision = metrics.precision_score(val_true, val_pred, average=None)
#         # class_f1_score = metrics.f1_score(val_true, val_pred, average=None)
#         self.ClassPrecision,self.ClassRecall,self.ClassF1Score,self.ClassSupport = metrics.precision_recall_fscore_support(self.ActualClasses, self.PredictedClasses,  average=None)
# 
#         #labels=None,
#         #beta : float, 1.0 by default
#         # The strength of recall versus precision in the F-score.
#         self.AveragePrecision,self.AverageRecall, self.AverageF1Score, self.AverageSupport = metrics.precision_recall_fscore_support(self.ActualClasses, self.PredictedClasses,  average='weighted')
#     
#         #print('validation accuracy:', val_accuracy)
#         if self.VerboseLevel == 2:
#             print('Accuracy', self.Accuracy)
#             print('Recall', self.ClassRecall)
#             print('Precision', self.ClassPrecision)
#             print('f1_score', self.ClassF1Score)
#             #np.savetxt(sRunName + '_precision.txt', class_precision, fmt='%10.7f', delimiter=';')
#             #np.savetxt(sRunName + '_recall.txt', class_recall, fmt='%10.7f', delimiter=';')
#             #np.savetxt(sRunName + '_f1_score.txt', class_f1_score, fmt='%10.7f', delimiter=';')
#             print('confusion_matrix')
#         
#         
#         self.ActualClasses=None
#         self.PredictedClasses=None
#         if p_sModelName is not None:
#             self.CheckForWinner(p_sModelName)
#     #------------------------------------------------------------------------------------
#     def CheckForWinner(self, p_sModelName):
#         if self.Accuracy >= self.WinnerAccuracy[0]:
#             self.WinnerAccuracy[0] = self.Accuracy
#             self.WinnerAccuracy[1] = p_sModelName
# 
#         if self.AverageRecall >= self.WinnerRecall[0]:
#             self.WinnerRecall[0] = self.AverageRecall
#             self.WinnerRecall[1] = p_sModelName            
# 
#         if self.AveragePrecision >= self.WinnerPrecision[0]:
#             self.WinnerPrecision[0] = self.AveragePrecision
#             self.WinnerPrecision[1] = p_sModelName
# 
#         if self.AverageF1Score >= self.WinnerF1Score[0]:
#             self.WinnerF1Score[0] = self.AverageF1Score
#             self.WinnerF1Score[1] = p_sModelName
# 
#     #------------------------------------------------------------------------------------
#     def PrintWinners(self):
#         print("Best Accuracy    %f, winner model is %s" % (self.WinnerAccuracy[0], self.WinnerAccuracy[1]))
#         print("Best Recall      %f, winner model is %s" % (self.WinnerRecall[0], self.WinnerRecall[1]))
#         print("Best Precision   %f, winner model is %s" % (self.WinnerPrecision[0], self.WinnerPrecision[1]))
#         print("Best F1 Score    %f, winner model is %s" % (self.WinnerF1Score[0], self.WinnerF1Score[1]))
#     #------------------------------------------------------------------------------------    
#     def Save(self,p_sBaseFileName, p_bIsWritingJSON=False):
#         class_index = np.asarray( [i for i,item in enumerate(self.ClassRecall)] )
#         
#         # Packs all into one neat file
#         acc_to_join=np.empty(self.ClassRecall.shape[0])
#         acc_to_join[0]=self.Accuracy
#         acc_to_join[1]=self.AverageRecall
#         acc_to_join[2]=self.AveragePrecision
#         acc_to_join[3]=self.AverageF1Score
#         all_figures = np.hstack(
#             (
#                  acc_to_join[np.newaxis].T
#                 ,self.ClassRecall[np.newaxis].T
#                 ,self.ClassPrecision[np.newaxis].T
#                 ,self.ClassF1Score[np.newaxis].T
#                 ,np.asarray(class_index)[np.newaxis].T
#                 ,self.ConfusionMatrix
#                 ,self.ClassSupport[np.newaxis].T        
#              ))
#         
#         sHeader = 'accuracy;recall;precision;f-score;class;'
#         for i,item in enumerate(class_index):
#             sHeader = sHeader + (str(item)) + ';'
#         sHeader = sHeader + 'support'   
#                  
#         #all_figures = np.vstack((np.asarray(sHeaders), all_figures))
#         
#        
#         # Writes the file
#         if p_bIsWritingJSON:
#             with open(p_sBaseFileName + '.csv', "w") as text_file:
#                 s = SimpleEncode(all_figures)
#                 text_file.write(s)
#                 #text_file.write("Purchase Amount: {0}".format(TotalAmount))
#         else:
#             np.savetxt(p_sBaseFileName + '.csv',all_figures, fmt='%10.7f', delimiter=';', header=sHeader)
#             np.save(p_sBaseFileName + '.num', all_figures)
#     #------------------------------------------------------------------------------------
# #==================================================================================================
#             
#             
            