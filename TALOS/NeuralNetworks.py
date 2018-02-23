# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        NEURAL NETWORK BASE CLASSES
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import numpy as np
import tensorflow as tf
import time as time
from TALOS.FileSystem import Storage, ModelsFolder
from TALOS.Core import Tensorboard, ScreenLogger 
from TALOS.Params import ModelParamCollection
from TALOS.Objects import TOBoolean
from TALOS.HyperParams import NNLearnParams, NNLearnConfig
from TALOS.Constants import TrainerType



#==================================================================================================
class PredictionData(object):
    #------------------------------------------------------------------------------------
    def __init__(self, p_nTotalSamples, p_nProbsCount, p_nClassCount=None):
        
        #........................ |  Instance Attributes | ..............................
        self.TotalSamples   = p_nTotalSamples
        self.ProbsCount     = p_nProbsCount
        if p_nClassCount is None:
            self.ClassCount     = self.ProbsCount
        else:
            self.ClassCount     = p_nClassCount
        
        self.SampleIDs          = np.zeros( (self.TotalSamples), dtype=np.int32)
        self.Actual             = np.zeros( (self.TotalSamples), dtype=np.int32)
        self.Predicted          = np.zeros( (self.TotalSamples), dtype=np.int32)
        self.PredictedProbs     = np.zeros( (self.TotalSamples, self.ProbsCount), dtype=np.float32)
        self.TopKappa           = self.ProbsCount
        self.TopKCorrect        = np.zeros( (self.TotalSamples, self.TopKappa), dtype=np.int32)
          
        self.ClassesTrue   = []
        self.ClassesFalse  = []
        self.ClassesActual = []
        self.TotalRecalled = 0
        #................................................................................
    #------------------------------------------------------------------------------------
    def AppendCounts(self, p_nTrueCount, p_nFalseCount, p_nActualCount):
        self.ClassesTrue.append(p_nTrueCount)
        self.ClassesFalse.append(p_nFalseCount)
        self.ClassesActual.append(p_nActualCount)        
    #------------------------------------------------------------------------------------
    def AppendPrediction(self, p_nBatchSize, p_nIDs, p_nTargets, p_nPredictedClasses, p_nPredictedClassProbs, p_nTopKCorrect):
        nCurrent = self.TotalRecalled
        self.SampleIDs[nCurrent : nCurrent + p_nBatchSize]          = p_nIDs[:]
        self.Actual[nCurrent : nCurrent + p_nBatchSize]             = p_nTargets[:,0]
        self.Predicted[nCurrent : nCurrent + p_nBatchSize]          = p_nPredictedClasses[:,0]    
        self.PredictedProbs[nCurrent : nCurrent + p_nBatchSize,:]   = p_nPredictedClassProbs[:,:self.ProbsCount]
        self.TopKCorrect[nCurrent: nCurrent + p_nBatchSize,:]       = p_nTopKCorrect[:,:self.TopKappa]
        self.TotalRecalled += p_nBatchSize
    #------------------------------------------------------------------------------------
#==================================================================================================







#==================================================================================================
class NNSavedState(object):
    __verboseLevel = 1
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent):
        #........ |  Instance Attributes | ..............................................
        self.Parent = p_oParent
        self.Log = self.Parent.Log
        assert self.Parent.Experiment is not None, "Neural network should have an experiment for NNSavedState."
        self.RunFolder = self.Parent.Experiment.RunSub
        self.ModelSizeOnDisk = None
        self.MaxSavedModels = 10
        self.SavedModels = []
        self.IsCompressingInitialState = True
        self.IsRestoringWeightsOnly = False
        
        self.ModelsFolder = None
        self.TrainerType = TrainerType.TALOS008_SOFTMAX_CLASSIFICATION
        #................................................................................
    #------------------------------------------------------------------------------------
    def __calculateDiskUsage(self, p_sFileName):
        if self.ModelsFolder is not None:
            self.ModelsFolder.Initialize(self.Parent.Log)
            self.ModelSizeOnDisk = self.ModelsFolder.ModelSizeInMBs
        else:
            sFolder, _, _ = Storage.SplitFileName(p_sFileName)
            self.ModelSizeOnDisk = Storage.GetFolderSize(sFolder)
            if self.Parent.Experiment is not None:
                self.MaMaxDiskSpaceForModels = self.Parent.Experiment.MaxDiskSpaceForModels
                self.MaxSavedModels = self.Parent.Experiment.MaxDiskSpaceForModels // self.ModelSizeOnDisk
    #------------------------------------------------------------------------------------            
    def __backup(self, p_sFileName, p_oSession):
        if type(self).__verboseLevel >= 1:
            self.Log.Print("  |__ Saving Weights to " + p_sFileName, end="")
        
        oVarList = tf.global_variables()
        self.Restorer = tf.train.Saver(oVarList, write_version=2)
        self.Restorer.save(p_oSession, p_sFileName)
        
        if self.__verboseLevel >= 1:
            self.Log.Print("-> Saved.")
        else:
            self.Log.Print(" ")
    #------------------------------------------------------------------------------------
    def __restore(self, p_sFileName, p_oSession):
        if type(self).__verboseLevel >= 1:
            self.Log.Print("  |__ Restoring Weights from " + p_sFileName, end="")
        
        if self.IsRestoringWeightsOnly:
            oList = self.Parent.ParamVars.Trainable
        else:
            oList = tf.global_variables()
            
        # Compatibility check
        oVarList = []
        for oItem in oList:
            bMustInclude = True
            if self.TrainerType == TrainerType.TALOS007_SOFTMAX_CLASSIFICATION:
                if oItem.name.startswith("TrainADAM/TrainADAM/LearningRate/value"):
                    bMustInclude = False
                    if type(self).__verboseLevel >= 1:
                        self.Log.Print("\n      |__ not loading variable [%s]" % oItem.name)
            
            if bMustInclude:
                oVarList.append(oItem)

        self.Restorer = tf.train.Saver(oVarList, write_version=2)
        self.Restorer.restore(p_oSession, p_sFileName)
        
        if self.__verboseLevel >= 2:
            self.DebugLayers(p_oSession)
        if self.__verboseLevel >= 1:
            self.Log.Print('-> Restored.')
    #------------------------------------------------------------------------------------
    def __getSaverParams(self, p_nEpochNumber, p_sFileName=None, p_oSession=None, p_bIsRunningExperiment=True):
        if p_sFileName is not None:
            sFileName = p_sFileName
        else:
            oExperiment = self.Parent.Experiment
            if oExperiment is not None:
                oSubFolder = oExperiment.RunSub
            else:
                oSubFolder = None   
            
            if oSubFolder is not None:
                if p_nEpochNumber == -1:
                    # Common initial weights for repeated training experiments
                    sFileName = oSubFolder.CommonInitialModelFileName
                elif p_nEpochNumber == 0:
                    sFileName = oSubFolder.InitialModelFileName
                else:
                    if p_bIsRunningExperiment:
                        Storage.EnsurePathExists(oSubFolder.ModelFolderTemplate % p_nEpochNumber)
                    sFileName = oSubFolder.ModelFileNameTemplate % p_nEpochNumber
        
        if p_oSession is not None:
            oSession = p_oSession
        else:
            oSession = self.Parent.ActiveSession
        
        assert sFileName is not None, "Filename not defined"
        assert oSession is not None , "Session not opened"
            
        return sFileName, oSession
    #------------------------------------------------------------------------------------
    def ImportInitialWeights(self, p_oInitialModelExperiment, p_bIsDecompressing=True):
        if p_bIsDecompressing:
            if not Storage.DecompressFile(p_oInitialModelExperiment.RunSub.InitialModelZipFileName, self.RunFolder.ExperimentFolder):
                raise Exception("Initial model zip file [%s] not found" % p_oInitialModelExperiment.RunSub.InitialModelFileName )
        
        _, sFileName, sExt = Storage.SplitFileName(p_oInitialModelExperiment.RunSub.InitialModelFileName)
        sModelFolder = self.RunFolder.ExperimentModelFolder
        sImportedFileName = Storage.JoinFileName(sModelFolder, sFileName, sExt)

        self.Log.Print("        |__ Initial model: %s" % sImportedFileName)
        
        self.LoadWeights(0, sImportedFileName)   
        
        if self.ModelSizeOnDisk is None:
            self.__calculateDiskUsage(sImportedFileName)
            
            # Done with loading and size calculation, deleting to save space  
            if self.ModelsFolder is not None:
                for sFile in self.ModelsFolder.InitialStateFiles:
                    Storage.DeleteFile(Storage.JoinPath(self.ModelsFolder.Folder, sFile))
    #------------------------------------------------------------------------------------
    def SaveInitialWeights(self):
        sFileName, oSession = self.__getSaverParams(0, p_sFileName=None, p_oSession=None, p_bIsRunningExperiment=True)        
        self.__backup(sFileName, oSession)
        if self.ModelSizeOnDisk is None:
            self.__calculateDiskUsage(sFileName)
        
        if self.IsCompressingInitialState:
            sModelFolder = self.RunFolder.ExperimentModelFolder
            bHasCreatedArchive, _ = Storage.CompressFolder(sModelFolder, self.RunFolder.InitialModelZipFileName)
            if bHasCreatedArchive:
                Storage.DeleteFolderFiles(sModelFolder)
    #------------------------------------------------------------------------------------
    def SaveWeights(self, p_nEpochNumber, p_sFileName=None, p_oSession=None):
        sFileName, oSession = self.__getSaverParams(p_nEpochNumber, p_sFileName, p_oSession, p_bIsRunningExperiment=True)        
        self.__backup(sFileName, oSession)
        if self.ModelSizeOnDisk is None:
            self.__calculateDiskUsage(sFileName)
    #------------------------------------------------------------------------------------
    def LoadWeights(self, p_nEpochNumber, p_sFileName=None, p_oSession=None):
        sFileName, oSession = self.__getSaverParams(p_nEpochNumber, p_sFileName, p_oSession, p_bIsRunningExperiment=False)
        self.__restore(sFileName, oSession)
    #------------------------------------------------------------------------------------        

#==================================================================================================








#==================================================================================================
class ArchitectureDef(object):
    __verboseLevel = 1
    
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent):
        #........ |  Instance Attributes | ..............................................
        self.Parent = p_oParent
        self.Items = []
        #................................................................................
    #------------------------------------------------------------------------------------
    def IndexOf(self, p_nDefItem):
        nFoundPos = None
        for nIndex,nItem in enumerate(self.Items):
            if nItem.ModuleIndex == p_nDefItem.ModuleIndex:
                nFoundPos = nIndex
                break
        return nFoundPos
    #------------------------------------------------------------------------------------
    def Add(self, p_nDefItem):
        if self.IndexOf(p_nDefItem) is None:
            self.Items.append(p_nDefItem) 
        #TODO
        #else
        # self.Items[nFoundIndex].Assign(p_nDefItem)
    #------------------------------------------------------------------------------------
    def Export(self, p_sTextFileName):
        with open(p_sTextFileName, "w") as oOutFile:
            for nDefItem in self.Items:
                nDefItem.WriteTo(oOutFile)
    #------------------------------------------------------------------------------------
#==================================================================================================    











#==================================================================================================
class BaseNeuralModule():
    __verboseLevel=1
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_nModuleIndex, p_nOutputFeatureDepth):
        #........ |  Instance Attributes | ..............................................
        self.Parent=p_oParent
        self.ModuleIndex=None
        self.Input=None
        self.Output=None
        self.InputFeatureDepth=0
        self.OutputFeatureDepth=p_nOutputFeatureDepth
        #................................................................................        
        self.Parent.Modules.append(self)
        nNewModuleIndex = len(self.Parent.Modules) - 1

        if p_nModuleIndex is not None:
            assert p_nModuleIndex==nNewModuleIndex
        self.ModuleIndex = nNewModuleIndex
        
    #------------------------------------------------------------------------------------
    def Initialize(self, p_bIsTrainingGraph):
        self.ModuleOnCreateTensors(p_bIsTrainingGraph)
    #------------------------------------------------------------------------------------
    def ModuleOnCreateTensors(self, p_bIsTrainingGraph):
        if p_bIsTrainingGraph:
            print("training")
        else:
            print("recall")
        raise Exception("Must implement virtual method ModuleOnCreateTensors()")

    #------------------------------------------------------------------------------------
#==================================================================================================











#==================================================================================================
class NNSetClassificationMetrics(object):
    #------------------------------------------------------------------------------------
    def __init__(self, p_sScopePrefix, p_tTrueBinCount, p_tFalseBinCount, p_tTargetBinCount):
        #........ |  Instance Attributes | ..............................................
        self.ScopePrefix = p_sScopePrefix
        self.TrueBinCount = p_tTrueBinCount        
        self.FalseBinCount = p_tFalseBinCount
        self.TargetBinCount = p_tTargetBinCount
        
        self.ClassPrecision = None
        self.ClassRecall = None
        self.ClassF1Score = None
        self.ValidIndices = None
        self.AvgPrecision = None
        self.AvgRecall = None
        self.AvgF1Score = None
        self.F1ScoreOfAvg = None
        #................................................................................
        assert self.FalseBinCount is not None, "Tensor for classified sample counts is not defined"
        assert self.TrueBinCount is not None, "Tensor for misclassified sample counts is not defined"
        assert self.TargetBinCount is not None, "Tensor for target sample counts is not defined"
    #------------------------------------------------------------------------------------
    def CreateTensors(self):
        with tf.variable_scope(self.ScopePrefix +"Metrics"):
            with tf.variable_scope("ClassPrecision"):
                self.ClassPrecision = self.TrueBinCount / (self.FalseBinCount + self.TrueBinCount + tf.constant(1e-7, dtype=tf.float32) )
            with tf.variable_scope("ClassRecall"):
                self.ClassRecall = self.TrueBinCount / (self.TargetBinCount + tf.constant(1e-7, dtype=tf.float32))
            with tf.variable_scope("ClassF1Score"):                
                self.ClassF1Score = ( tf.constant(2.0, dtype=tf.float32) * (self.ClassPrecision * self.ClassRecall)
                                      ) / (self.ClassPrecision + self.ClassRecall + tf.constant(1e-7, dtype=tf.float32))
            
            with tf.variable_scope("ValidClasses"):
                tWhere = tf.not_equal( self.TargetBinCount, tf.constant(0.0, dtype=tf.float32))
                self.ValidIndices = tWhere
                tIndices = tf.where(tWhere)
                tValidPrecision = tf.gather(self.ClassPrecision, tIndices)
                tValidRecall = tf.gather(self.ClassRecall, tIndices)
                tValidF1Score = tf.gather(self.ClassRecall, tIndices)
            
            with tf.variable_scope("AvgPrecision"):
                self.AvgPrecision = tf.reduce_mean(tValidPrecision, axis=0)
            with tf.variable_scope("AvgRecall"):
                self.AvgRecall = tf.reduce_mean(tValidRecall, axis=0)
            with tf.variable_scope("AvgF1Score"):
                self.AvgF1Score = tf.reduce_mean(tValidF1Score, axis=0)
            with tf.variable_scope("F1ScoreOfAvg"):
                self.F1ScoreOfAvg = (tf.constant(2.0, dtype=tf.float32) * (self.AvgPrecision * self.AvgRecall)
                                     ) / (self.AvgPrecision + self.AvgRecall + tf.constant(1e-7, dtype=tf.float32))
    #------------------------------------------------------------------------------------        
#==================================================================================================













    















#==================================================================================================
class NNSetEvaluation(object):
    #------------------------------------------------------------------------------------
    def __init__(self, p_sScopePrefix, p_tTargets, p_tPredictedProbs, p_nTopKappa=None):
        #..................... |  Instance Attributes | .................................
        self.ScopePrefix = p_sScopePrefix
        self.TopKappa = p_nTopKappa
        self.IsTopK = self.TopKappa is not None
        
        self.Targets        = p_tTargets
        self.PredictedProbs = p_tPredictedProbs
        
        assert self.Targets is not None
        assert self.PredictedProbs is not None
        
        self.Margin = None
        self.MarginClass=None
        
        self.Predictions = None
        self.PredictedClass = None
        
        self.ActualProbs = None
        self.Actuals = None
        self.ActualClass = None
        
        self.TopKProb = None
        self.TopKIndexes = None
        self.TopKCorrect = None
        self.TopKAccuracy = None
        
        self.Positives = None
        self.Negatives = None
        self.Accuracy = None
        
        self.TrueBinCount = None
        self.FalseBinCount = None
        self.ActualBinCount = None
        self.PredictedBinCount= None    
        
        nShape = self.PredictedProbs.get_shape()
        self.BatchSize = int(nShape[0])
        self.ClassCount = int(nShape[1])
        #................................................................................    
    #------------------------------------------------------------------------------------
    def CreateTensors(self):
        with tf.variable_scope(self.ScopePrefix + "Targets"):
            self.ActualProbs = tf.one_hot(tf.reshape(self.Targets, [self.BatchSize]), depth=self.ClassCount, on_value=1.0, off_value=0.0, dtype=tf.float32)
        
        with tf.name_scope(self.ScopePrefix + "Evaluation"):
            with tf.variable_scope("Predicted"):
                self.Predictions=tf.argmax(self.PredictedProbs, axis=1, output_type=tf.int32)
                self.PredictedClass=tf.reshape(tf.cast(self.Predictions, dtype=tf.int32), [-1,1])
            
            with tf.variable_scope("Actual"):
                self.Actuals = tf.argmax(self.ActualProbs, axis=1, output_type=tf.int32) 
                self.ActualClass = tf.reshape(tf.cast(self.Actuals, dtype=tf.int32), [-1,1])
    
            if self.IsTopK:
                with tf.variable_scope("TopK"):
                    self.TopKProb, self.TopKIndexes = tf.nn.top_k(self.PredictedProbs, self.TopKappa, sorted=True, name="Top%d" % self.TopKappa)
                    
                with tf.variable_scope("TopKAccuracy"):
                    tEqualTopK = tf.equal( self.TopKIndexes, self.ActualClass )
                    tBitsTopK = tf.cast(tEqualTopK, dtype=tf.int32)
                    self.TopKCorrect = tBitsTopK
                    self.TopKAccuracy = tf.reduce_mean( tf.cast( tf.reduce_sum(tBitsTopK, axis=1), dtype=tf.float32))

            with tf.name_scope("Confusion"):                
                tEqual   = tf.equal( self.PredictedClass, self.ActualClass ) 
                tUnequal = tf.logical_not( tEqual )
                self.Positives = tEqual   
                self.Negatives = tUnequal  
            
            with tf.name_scope("Accuracy"):            
                self.Accuracy = tf.reduce_mean(tf.cast(tEqual, tf.float32))
        
        self.__createCounts()

        self.Metrics = NNSetClassificationMetrics(self.ScopePrefix, self.TrueBinCount, self.FalseBinCount, self.ActualBinCount)
        self.Metrics.CreateTensors()
        
        nClass = self.MarginClass
        if nClass is not None:
            self.Margin  = nClass(self.ScopePrefix, self.PredictedProbs, self.TopKProb, self.ActualClass, self.Positives, self.Negatives )
            self.Margin.CreateTensors()        
    #------------------------------------------------------------------------------------
    def __createCounts(self):
        with tf.variable_scope(self.ScopePrefix + "Counts"):
            tTrueClasses = tf.boolean_mask(self.PredictedClass, self.Positives, name="Positive")
            tFalseClasses = tf.boolean_mask(self.PredictedClass, self.Negatives, name="Negative")
            with tf.variable_scope("PositiveCount"):
                self.TrueBinCount = tf.bincount(tTrueClasses, minlength=self.ClassCount, dtype=tf.float32)
            with tf.variable_scope("NegativeCount"):
                self.FalseBinCount = tf.bincount(tFalseClasses, minlength=self.ClassCount, dtype=tf.float32)
            with tf.variable_scope("ActualCount"):
                self.ActualBinCount = tf.bincount(self.Actuals, minlength=self.ClassCount, dtype=tf.float32)
            with tf.variable_scope("PredictedCount"):
                self.PredictedBinCount = tf.bincount(self.Predictions, minlength=self.ClassCount, dtype=tf.float32)
    #------------------------------------------------------------------------------------
#==================================================================================================        


















   














#==================================================================================================
class BaseNN():
    __verboseLevel=2
    
    GT_SYNC_RECALL_SYNC   = 0
    GT_SYNC_TRAIN_SYNC    = 1
    GT_ASYNC_DUAL_QUEUES  = 2
    
    #------------------------------------------------------------------------------------
    def __init__(self, p_sExperimentName, p_tInputQueue=None, p_oExperiment=None, p_oSettings=None):
        if type(self).__verboseLevel>=2:
            print("[>] Creating Neural Network Structure ...")

        #........ |  Instance Attributes | ..............................................
        self.Settings=p_oSettings 
        self.ExperimentName=p_sExperimentName
        self.Experiment=p_oExperiment  
        if self.Experiment is not None:
            self.Log=self.Experiment.Log
        else:
            self.Log=ScreenLogger()
        self.IsTrainable=False
        self.InputQueue=p_tInputQueue
        # Sets the global log for CNN layer debugging
        #SetActiveLog(self.Log)
        
        self.Layers=[]        
        self.Input=None
        self.Output=None
        
        self.WeightInitPseudoRandomSeed=None
        self.LearnParams=None
        self.ActiveSession=None
        #self.IsTrainingProcess       = None
        self.StartTrainingProcessOp  = None
        self.FinishTrainingProcessOp = None
        #self.NonTrainableInputLayer  = None
        self.NonTrainableOutputLayer = None
        
        #self.Modules=[None] # PANTELIS [2017-04-10] Moved from DCNN
        
        # // Modules \\
        self.Modules=[]
        self.Tensors=[]  
        self.InputModule=None
        self.OutputModule=None
        
          
        self.IsTrained=False
        self.IsInitialized=False

        # // Evaluation \\
        #self.LossInput=None
        self.TrainLossInput=None
        self.ValLossInput=None
        
        self.FlagIsTraining=None
        self.FlagIsTrainingForRecall=None
        
        #... Metrics ...
        self.MarginClass = None
        self.TrainOut = None
        self.ValOut = None
                
        # // Queue Objects \\
        self.TrainQInput=None
        self.TrainQTargets=None
        self.TrainLoss=None
        
        
        self.ValQInput=None
        self.ValQTargets=None
        self.ValLoss=None
        

        self.InputPlaceholder=None
        
        self.TrainClassEqual=None
        self.ValClassEqual=None
        
        self.ValClassPositives=None
        
        # // Composite Objects \\
        self.ParamVars = ModelParamCollection()
        self.State = NNSavedState(self)
        if self.Experiment is not None:
            self.State.ModelsFolder = ModelsFolder(self.Experiment.RunSub.ExperimentModelFolder)
        self.Architecture = ArchitectureDef(self)
        #................................................................................
        #self.NNOnCreate()
        self.CreateModules()
        # Checks the proper configuration for this neural network
        #if self.LearnParams is None:
        #    raise "No learning parameters defined on network"
        self.DoSetupLearning()
        
        return None
    #-----------------------------------------------------------------------------------
    def ExportArchitecture(self, p_oMainGraph):
        assert self.Experiment is not None, "Experiment environement is required for exporting the architecture"
        
        # Writes the graph for visualization    
        self.Experiment.WriteGraph(p_oMainGraph)
        self.Architecture.Export(self.Experiment.RunSub.NetworkArchitectureFileName)
    #------------------------------------------------------------------------------------
    def CreateModules(self):
        pass
    #------------------------------------------------------------------------------------
    def SetupLearning(self):
        pass    
    #------------------------------------------------------------------------------------
    def Finalize(self):
        tf.reset_default_graph()    
    #------------------------------------------------------------------------------------
    def NetworkFunctionEx(self, p_bIsTrainingGraph):
        # Creates the tensors of the neural network on the current subgraph, training or recall  
        for nIndex, oModule in enumerate(self.Modules):
            if nIndex == 0:
                if type(self).__verboseLevel >= 3:
                    print("Input tensor to neural network", p_bIsTrainingGraph, self.InputModule.Output.name)
                self.Modules[0] = self.InputModule
            else:
                oModule.Initialize(p_bIsTrainingGraph)
            oOutputModule=oModule
                            
        return oOutputModule.Output 
    #------------------------------------------------------------------------------------
    def Initialize(self, p_oSession, p_bIsTrainable=False, p_tInputQueue=None):
        if p_bIsTrainable:
            if p_tInputQueue is None:
                self.GraphType = BaseNN.GT_SYNC_TRAIN_SYNC                    
            else:
                self.GraphType = BaseNN.GT_ASYNC_DUAL_QUEUES
        else:
            self.GraphType = BaseNN.GT_SYNC_RECALL_SYNC
                    
        #PANTELIS [2017-05-07] Image Queue Full Support
        self.InputQueue=p_tInputQueue  
        
                
        if type(self).__verboseLevel>=2:
            self.Log.Print("[>] Initializing Neural Network and Session ...")             
        
        self.ActiveSession = p_oSession
        
        if self.GraphType == BaseNN.GT_SYNC_RECALL_SYNC:
            #self.__makeNetworkSyncRecall()
            raise Exception("Not supported.")
        elif self.GraphType == BaseNN.GT_SYNC_TRAIN_SYNC:
            self.__makeNetworkSyncTrain()
        elif self.GraphType == BaseNN.GT_ASYNC_DUAL_QUEUES:
            #self.__previewNetworkTemplate()
            raise Exception("Not supported.")
        else:
            raise Exception("Not supported.")
                
        # Runs the tensorflow initializers
        self.ActiveSession.run(tf.local_variables_initializer())
        self.ActiveSession.run(tf.global_variables_initializer())
        
        if type(self).__verboseLevel>=1:
            self.PrintModelParamCount()
        
        self.IsInitialized=True
    #------------------------------------------------------------------------------------
    def __OldmakeNetworkSyncRecall(self):
        if type(self).__verboseLevel>=2:
            print("|__ Sync recall network ...")             
        
        #TEMP: Integer target
        self.ValQTargets=tf.placeholder(dtype=tf.int32, shape=(1), name="Targets") 
        
        self.CategoricalCrossEntropy(self.Modules[0].Input, p_bIsTrainingGraph=False)
        self.ParamVars.AdjustAfterReuse(1) #PANTELIS: [2017-12-05] Perhaps this is uneeded

    #------------------------------------------------------------------------------------   
    def __OldmakeNetworkTemplate(self):
        if type(self).__verboseLevel>=2:
            print("|__ Async train/recall network with queue support ...")             
        
        
        self.TrainQInput=self.InputQueue.DeQSampleFeatures
        self.TrainQTargets=self.InputQueue.DeQSampleTargets
        
        self.ValQInput=self.InputQueue.DeQSampleFeaturesRecall
        self.ValQTargets=self.InputQueue.DeQSampleTargetsRecall
        
        fReusedModelTemplate = tf.make_template('NNModel', self.CategoricalCrossEntropy)
        
        # Creates only one network if the queue is not multithreaded
        if self.InputQueue.IsMultiThreaded:
            self.TrainOutput, self.TrainLoss, self.TrainAccuracy, self.TrainActual, self.TrainPredicted = fReusedModelTemplate(self.TrainQInput, True)#, self.TrainQTargets)
        self.ValOutput, self.ValLoss, self.ValAccuracy, self.RecallActual, self.RecallPredicted  = fReusedModelTemplate(self.ValQInput, False)#, self.ValQTargets)
        
        self.ParamVars.AdjustAfterReuse()
    #------------------------------------------------------------------------------------
    def __makeNetworkSyncTrain(self, p_bIsTraining=True):
        if type(self).__verboseLevel>=2:
            print("  |__ Sync train network ...")             
        
        fReusedModelTemplate = tf.make_template("NNModel", self.CategoricalCrossEntropy)
        # Creates the training tensors from cloned network
        if p_bIsTraining:
            # Input module will be initialized for training subgraph
            self.InputModule.Initialize(True)  
            self.TrainQInput = self.InputModule.TrainInput
            self.TrainQTargets = self.InputModule.TrainTargets
            self.TrainLoss = fReusedModelTemplate(self.TrainQInput, True)
        
        # Input module will be initialized for recall subgraph
        self.InputModule.Initialize(False)        
        self.ValQInput=self.InputModule.ValInput
        self.ValQTargets=self.InputModule.ValTargets
        
        self.ValLoss = fReusedModelTemplate(self.ValQInput, False)
        
        self.ParamVars.AdjustAfterReuse()        
    #------------------------------------------------------------------------------------
    def DoSetupLearning(self):
        self.LearnParams=NNLearnParams()
        
        oSourceConfig = None
        sFileName = self.Experiment.RunSub.LearnConfigFileName
        if Storage.IsExistingFile(sFileName):
            self.Log.Print("  |__ Training settings loaded from %s" % sFileName)
            oSourceConfig = NNLearnConfig()
            oSourceConfig.Learn = self.LearnParams
            oSourceConfig.LoadFromFile(sFileName)
                        
        self.SetupLearning()
        
        oConfig = NNLearnConfig()
        oConfig.AssignFrom(oSourceConfig)
        oConfig.Learn = self.LearnParams
        oConfig.SaveToFile(self.Experiment.RunSub.LearnConfigUsedFileName)        
    #------------------------------------------------------------------------------------
    def CategoricalCrossEntropy(self, p_tInput, p_bIsTrainingGraph):
        if type(self).__verboseLevel>=3:
            if p_bIsTrainingGraph:
                print("  |__ Network Training Function")
            else:
                print("  |__ Network Recall Function")
             
        assert p_tInput is not None, "Undefined neural network input."
        tOutput = self.NetworkFunctionEx(p_bIsTrainingGraph)
        #if p_bIsTrainingGraph:
        #    self.DoSetupLearning()
        
        assert tOutput is not None, "Undefined neural network output."
        #nBatchSize      = p_tInput.get_shape().as_list()[0]
        #nTargetsCount   = tOutput.get_shape().as_list()[1]
        
        # Determines the proper top-k accuracy
        if self.Settings.ClassCount == 2:
            nTopKappa = 1
        elif self.Settings.ClassCount <= 20:
            nTopKappa = 2
        else:
            nTopKappa = 5
        
        # Creates the evaluation and metric tensors
        if p_bIsTrainingGraph:
            #self.Metrics.CreateValidationTensors(sScopeNamePrefix, tPredicted, tActual)      
            self.TrainOut = NNSetEvaluation("Train", self.TrainQTargets, tOutput, p_nTopKappa=nTopKappa)
            self.TrainOut.MarginClass = self.MarginClass
            self.TrainOut.CreateTensors()
            if self.LearnParams.FilterCheckEpochs is not None:
                self.Log.Print("  |__ Using sample filtering during training")
        else:
            self.ValOut = NNSetEvaluation("Val", self.ValQTargets, tOutput, p_nTopKappa=nTopKappa)
            self.ValOut.MarginClass = self.MarginClass
            self.ValOut.CreateTensors()
            if self.LearnParams.FilterCheckEpochs is not None:
                self.Log.Print("  |__ Using sample filtering during validation")            
            
        # Different scope names for the two subgraphs, training and recall
        if p_bIsTrainingGraph:
            sScopeNamePrefix = "Train"
            tSoftMaxLayerSum = self.TrainLossInput
            tTargetProbs = self.TrainOut.ActualProbs
        else:
            sScopeNamePrefix = "Val"
            tSoftMaxLayerSum = self.ValLossInput
            tTargetProbs = self.ValOut.ActualProbs
            
        if type(self).__verboseLevel >= 3:
            print("tTargetProbs", tTargetProbs)
            print("tSoftMaxLayerSum", tSoftMaxLayerSum)
            print("tOutput", tOutput)
        
        # Categorical cross entropy as error function. The softmax_cross_entropy_with_logits should have as input 
        # the synaptic sum of the softmax layer instead of the softmax activation function output
        with tf.name_scope(sScopeNamePrefix + "Loss"):
            tLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tSoftMaxLayerSum, labels=tTargetProbs))
            tf.add_to_collection(tf.GraphKeys.LOSSES, tLoss)
                    
        return tLoss 
    #------------------------------------------------------------------------------------
    def VisualizeModelGraph(self, p_oRunningExperiment=None):
        if p_oRunningExperiment is None:
            p_oRunningExperiment = self.Experiment
        
        oTensorboard=Tensorboard(p_oExperiment=p_oRunningExperiment)
        oTensorboard.Initialize(self.ActiveSession)
        self.PrintModelParamCount()
    #------------------------------------------------------------------------------------
    def PrintModelParamCount(self):
        nPredefinedParams, nTrainableParams, nTrainableLayers = self.ParamVars.TotalParamCount()
        nTotalParamsInM = (nTrainableParams + nPredefinedParams) / 1000000.0
        self.Log.Print( "  |__ Model has %.2f M parameters in total" % (nTotalParamsInM))
        self.Log.Print( "      |__  Trainable params are %i in %i trainable layers" % (nTrainableParams, nTrainableLayers) )
        self.Log.Print( "      |__  Predefined params are %i" % (nPredefinedParams) )
        self.Log.Flush()
    #------------------------------------------------------------------------------------
    #DEPRECATED   
#     def InputLayer(self, p_nInputShape=[None, 28, 28, 1], p_tInputTensor=None):
#         if p_tInputTensor is None:
#             with tf.name_scope("Input"):
#                 tResult = tf.placeholder(dtype=np.float32, shape=p_nInputShape, name="nninput")
#         else:
#             tResult=p_tInputTensor
#         
#         self.Layers.append(tResult)        
#         self.SetModule(0, tResult)
#         return tResult           
    #------------------------------------------------------------------------------------
    def GetPreviousModuleOutput(self, p_nModuleIndex, p_nCustomTensor=None, p_bToFullyConnected=False):
        tResult=None
        nResultFeatures=None
        
        if p_nCustomTensor is not None:
            tResult = p_nCustomTensor
        elif p_nModuleIndex is None:
            if type(self).__verboseLevel >= 3:
                print("  [>] Free standing module #%i" % (p_nModuleIndex - 1))
            tResult = p_nCustomTensor
        elif p_nModuleIndex > 0:
            if type(self).__verboseLevel >= 3:
                print("  [>] Using module #%i" % (p_nModuleIndex - 1))
            tResult = self.Modules[p_nModuleIndex - 1].Output
            
        if tResult is None:
            raise Exception("previous layer tensor not found %i" % p_nModuleIndex)
        else:
            nResultShape=tResult.get_shape()
            
            nResultFeatures = None
            if (nResultShape.ndims==4):
                if p_bToFullyConnected:
                    # Features of convolutional layer to fully connected layer
                    nResultFeatures=int(nResultShape[1])*int(nResultShape[2])*int(nResultShape[3])
                else:
                    # Features of convolutional layer to next convolutional layer
                    nResultFeatures=int(nResultShape[3])
            elif (nResultShape.ndims==2):
                if p_bToFullyConnected:
                    # Features of fully connected layer to fully connected layer
                    nResultFeatures=int(nResultShape[1])
            
        return tResult, nResultFeatures
    #------------------------------------------------------------------------------------    
    def GetPreviousModuleOf(self, p_nModuleIndex, p_nCustomTensor=None, p_bToFullyConnected=False):
        Result=None
        if p_nCustomTensor is not None:
            Result = p_nCustomTensor
        elif p_nModuleIndex is None:
            if type(self).__verboseLevel >= 2:
                print("  [>] Free standing module #%i" % (p_nModuleIndex - 1))
            Result = p_nCustomTensor
        elif p_nModuleIndex > 0:
            if type(self).__verboseLevel >= 2:
                print("  [>] Using module #%i" % (p_nModuleIndex - 1))
            Result = self.Modules[p_nModuleIndex - 1]
            
        if Result is None:
            raise "previous layer tensor not found %i" % p_nModuleIndex
        else:
            nResultFeatures = None
            nResultShape=Result.get_shape()

            if (nResultShape.ndims==4):
                if p_bToFullyConnected:
                    # Features of convolutional layer to fully connected layer
                    nResultFeatures=int(nResultShape[1])*int(nResultShape[2])*int(nResultShape[3])
                else:
                    # Features of convolutional layer to next convolutional layer
                    nResultFeatures=int(nResultShape[3])
                    
            elif (nResultShape.ndims==2):
                if p_bToFullyConnected:
                    # Features of fully connected layer to fully connected layer
                    nResultFeatures=int(nResultShape[1])
            
            
        return Result, nResultFeatures
    #------------------------------------------------------------------------------------
    # PANTELIS [2017-04-10] Moved to BaseNN
    # PANTELIS [2017-12-05] Using the tensors collection
    def SetTensor(self, p_nModuleIndex, p_tTensor):
        if p_nModuleIndex is not None:
            nLen = len(self.Tensors)
            if p_nModuleIndex >= nLen:
                nDiff = p_nModuleIndex - len(self.Tensors) + 1
                if type(self).__verboseLevel >= 3:
                    print("Extending Module array by %i" % nDiff)
                
                self.Tensors.extend([None]*nDiff)
                if type(self).__verboseLevel >= 3:
                    print("Extending Module Array by %i for Module #%i from %i to %i" % (nDiff, p_nModuleIndex, nLen, len(self.Tensors)) )
                    
            self.Tensors[p_nModuleIndex] = p_tTensor
    #------------------------------------------------------------------------------------                    
    def GetIsTrainingFlag(self, p_bIsTrainingGraph):
        if p_bIsTrainingGraph:
            if self.FlagIsTraining is None:
                self.FlagIsTraining=TOBoolean("IsTraining", True)
            Result = self.FlagIsTraining.Tensor
            if type(self).__verboseLevel >=3:
                print("Training Flag Returned")
        else:
            if self.FlagIsTrainingForRecall is None:
                self.FlagIsTrainingForRecall=TOBoolean("IsTrainingRecall", False)
            Result = self.FlagIsTrainingForRecall.Tensor
            if type(self).__verboseLevel >=3:
                print("Recall Flag Returned")
        return Result
    #------------------------------------------------------------------------------------
    def DebugLayers(self, p_oSession):
        AllTrainableVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.Vars={}
        for i,item in enumerate(AllTrainableVars):
            #self.Vars[item.name] = item.eval(p_oSession)
            print(i, item.name, item.get_shape())
    #------------------------------------------------------------------------------------
    def PredictMiniBatch(self):
        nRecallMiniBatchSize = self.InputQueue.Setup.SmallBatchSamplesCount
        
        nPredicted = np.zeros((nRecallMiniBatchSize), np.float32)
        nActual = np.zeros((nRecallMiniBatchSize), np.float32)
                
        print("  |__ Recalling ..." )
        nOut = self.ActiveSession.run([self.RecallPredicted, self.RecallActual])
        print("      |__ Done." )
        
        nPredicted[:] = nOut[0][:]
        nActual[:] = nOut[1][:]
        
        return nPredicted, nActual
    #------------------------------------------------------------------------------------
    def PredictEval(self, p_oIterator):
        MAX_PROBS_COUNT = 10
        
        nProbsCount = self.Settings.ClassCount
        if nProbsCount > MAX_PROBS_COUNT:
            nProbsCount = MAX_PROBS_COUNT
        
        if type(self).__verboseLevel >= 1:
            self.Log.Print("  |__ Start recall")
        oRecallIterator = p_oIterator
        #oRecallIterator.Start()

        oData = PredictionData(oRecallIterator.TotalSamples, self.ValOut.TopKappa)
        #oData.TopKappa = self.ValOut.TopKappa

#         nSampleIDs          = np.zeros( (oMarginIterator.TotalSamples), dtype=np.int32)
#         nActual             = np.zeros( (oMarginIterator.TotalSamples), dtype=np.int32)
#         nPredicted          = np.zeros( (oMarginIterator.TotalSamples), dtype=np.int32)
#         nPredictedProbs     = np.zeros( (oMarginIterator.TotalSamples, nProbsCount), dtype=np.float32)
                
        nStep=0
        #nTotalRecalled=0
        bContinue=True            
        while bContinue:
            nRange = oRecallIterator.FirstBatch()
            while not oRecallIterator.EndOfData():
                nIDs     = oRecallIterator.Pager.IDs[nRange[0]:nRange[1]]
                nSamples = oRecallIterator.Pager.Samples[nRange[0]:nRange[1]]
                nTargets = oRecallIterator.Pager.Targets[nRange[0]:nRange[1]]
                
                
                nBatchSize = nSamples.shape[0]
                bIsIncompleBatch = nBatchSize != oRecallIterator.BatchSize
                if bIsIncompleBatch:
                    if type(self).__verboseLevel >=3:
                        print(nSamples.shape)
                    nSamples = np.concatenate( (   nSamples
                                                 , np.zeros( (oRecallIterator.BatchSize - nBatchSize,nSamples.shape[1], nSamples.shape[2], nSamples.shape[3]), dtype=nSamples.dtype ) 
                                                )
                                              , axis=0)
                
                
                nPredictedClasses, nPredictedClassProbs, \
                    nTrueCount, nFalseCount, nActualCount, nTopKCorrect = self.ActiveSession.run( 
                      [  
                         self.ValOut.PredictedClass
                        ,self.ValOut.PredictedProbs 
                        
                        ,self.ValOut.Metrics.TrueBinCount
                        ,self.ValOut.Metrics.FalseBinCount
                        ,self.ValOut.Metrics.TargetBinCount
                        ,self.ValOut.TopKCorrect
                      ] 
                      ,feed_dict={ self.ValQInput : nSamples, self.ValQTargets: nTargets }
                    )    
                
               
                if bIsIncompleBatch:
                    nPredictedClasses    = nPredictedClasses[:nBatchSize]
                    nPredictedClassProbs = nPredictedClassProbs[:nBatchSize]
                    
                    if type(self).__verboseLevel >=3:
                        print(oData.PredictedProbs.shape)
                
                    
                oData.AppendCounts(nTrueCount, nFalseCount, nActualCount)
                oData.AppendPrediction(nBatchSize, nIDs, nTargets, nPredictedClasses, nPredictedClassProbs, nTopKCorrect)
                                    
#                 nActual[nTotalRecalled:nTotalRecalled+nBatchSize] = nTargets[:,0]
#                 nPredicted[nTotalRecalled:nTotalRecalled+nBatchSize] = nPredictedClasses[:,0]    
#                 nPredictedProbs[nTotalRecalled:nTotalRecalled+nBatchSize,:] = nPredictedClassProbs[:,:nProbsCount]
#                 nSampleIDs[nTotalRecalled:nTotalRecalled+nBatchSize] = nIDs[:]
                #nTotalRecalled += nBatchSize
                
                if type(self).__verboseLevel >= 1:
                    self.Log.Print("             |__ Recalled batch %i/%i  [%s,%s] total:%d" % 
                                     (   nStep + 1, oRecallIterator.TrainingBatches
                                        ,nRange[0], nRange[1], oData.TotalRecalled# nTotalRecalled 
                                      )
                                   )
                    
                nStep += 1
                nRange = oRecallIterator.NextBatch()
                if type(self).__verboseLevel >= 3:
                    print(nRange, oRecallIterator.SampleIndex, oRecallIterator.TotalCachedSamples)

                
                
                
            if oRecallIterator.GetIsEpochFinished():
                bContinue = False
            else:
                oRecallIterator.Resume()

        #return nActual, nPredicted, nPredictedProbs, nSampleIDs,
        return oData 
    #------------------------------------------------------------------------------------
    def PredictOld(self, p_bIsForceVerbosing=False):
        nSampleCount         = self.InputQueue.Setup.TotalSampleCount
        nTotalRecallSteps    = self.InputQueue.Setup.ValidationStepsCount
        nRecallMiniBatchSize = self.InputQueue.Setup.SmallBatchSamplesCount
        
        nTestPredicted = np.zeros((nSampleCount), np.float32)
        nTestActual = np.zeros((nSampleCount), np.float32)
        if type(self).__verboseLevel>=1:
            print("  |__ Recalling ..." )
            print("      |__ Total samples in UT=%i in %i steps of batch size %i" % (nSampleCount, nTotalRecallSteps, nRecallMiniBatchSize) )
            
        for nStep in range(0, nTotalRecallSteps):
            nStart = nStep*nRecallMiniBatchSize
            nEnd   = nStart + nRecallMiniBatchSize
            
            nOut = self.ActiveSession.run([self.RecallPredicted, self.RecallActual])
            
            nTestPredicted[nStart:nEnd] = nOut[0][:]
            nTestActual[nStart:nEnd] = nOut[1][:]
            
            nInQueueSamples = self.InputQueue.Count(False)
            if (type(self).__verboseLevel>=2) or p_bIsForceVerbosing:
                if p_bIsForceVerbosing:
                    bWillPrint=(nStep % 5 == 0)
                else:
                    bWillPrint=True
                if bWillPrint:
                    print("      |__ (%i/%i) Recalling samples:%i to %i (in queue=%i) " % (nStep+1, nTotalRecallSteps, nStart, nEnd, nInQueueSamples))
            
            if nInQueueSamples < nRecallMiniBatchSize:
                # Allow the queue to fill up 
                # TODO Loop it               
                time.sleep(1.5)

                         
        return nTestActual,nTestPredicted
    #------------------------------------------------------------------------------------
    def RecallActivationMaps(self, p_nModuleIndexes):
        
        assert p_nModuleIndexes is not None, "Please provide a list of module indexes"

        nSampleCount         = self.InputQueue.Setup.TotalSampleCount
        nTotalRecallSteps    = self.InputQueue.Setup.ValidationStepsCount
        nRecallMiniBatchSize = self.InputQueue.Setup.SmallBatchSamplesCount
        
        oModules = []
        oActivations = []
        oCategories=np.zeros((nSampleCount), np.int32)
        for nModuleIndex in p_nModuleIndexes: 
            tModule         = self.Modules[nModuleIndex]
            oModules.append(tModule)
            oModuleShape    = tModule.get_shape().as_list()
    
            if type(self).__verboseLevel>=1:
                print("  |__ Module shape", oModuleShape)
    
            nActivations=np.zeros((nSampleCount, int(oModuleShape[1]), int(oModuleShape[2]), int(oModuleShape[3])), np.float32)
            
            if type(self).__verboseLevel>=1:
                print("  |__ Activations shape", nActivations.shape)
                print("  |__ Recalling ..." )
                print("      |__ Total samples in UT=%i in %i steps of batch size %i" % (nSampleCount, nTotalRecallSteps, nRecallMiniBatchSize) )
            oActivations.append(nActivations)
            
            
        if (len(oModules) >= 1) and (len(oModules) <= 4):
            #One activation map or combo of multiple activation maps at once
            for nStep in range(0, nTotalRecallSteps):
                nStart = nStep*nRecallMiniBatchSize
                nEnd   = nStart + nRecallMiniBatchSize
                
                if len(oModules) == 4:
                    nOut = self.ActiveSession.run([self.RecallActual, oModules[0],oModules[1], oModules[2], oModules[3]])                    
                elif len(oModules) == 3:                
                    nOut = self.ActiveSession.run([self.RecallActual, oModules[0],oModules[1], oModules[2]])
                elif len(oModules) == 2:
                    nOut = self.ActiveSession.run([self.RecallActual, oModules[0],oModules[1]])
                elif len(oModules) == 1:
                    nOut = self.ActiveSession.run([self.RecallActual, oModules[0]])
                
                oCategories[nStart:nEnd] = nOut[0]    
                for nModuleIndex, _ in enumerate(oModules):
                    oActivations[nModuleIndex][nStart:nEnd] = nOut[nModuleIndex+1][:]
                   
                      
                if (nStep == 0) or ((nStep+1) % 5 == 0):
                    print("      |__ (%i/%i) Activation maps %s for samples:%i to %i" % (nStep+1, nTotalRecallSteps, tuple(p_nModuleIndexes), nStart, nEnd))
                                        
            self.InputQueue.ResumeRecallQueue()
        else:
            # For more than 4 activation maps
            for nModuleIndex, tModule in enumerate(oModules):
                for nStep in range(0, nTotalRecallSteps):
                    nStart = nStep*nRecallMiniBatchSize
                    nEnd   = nStart + nRecallMiniBatchSize
                    nOut = self.ActiveSession.run([self.RecallActual, tModule])
                    
                    oCategories[nStart:nEnd] = nOut[0]
                    oActivations[nModuleIndex][nStart:nEnd] = nOut[1][:]
                      
                    if (nStep == 0) or ((nStep+1) % 5 == 0):
                        print("      |__ (%i/%i) Activation maps %s for Recalling samples:%i to %i" % (nStep+1, nTotalRecallSteps, tuple(p_nModuleIndexes), nStart, nEnd))
                        
                self.InputQueue.ResumeRecallQueue()
            
            
          
                 
        return oActivations, oCategories    
    #------------------------------------------------------------------------------------
               
#==================================================================================================              



















#==================================================================================================
class NNImageClasifierSettings(object):
    #------------------------------------------------------------------------------------
    def __init__(self, p_nClassCount, p_nImageDimensions, p_nBatchSize=15):
        #........ |  Instance Attributes | ..............................................
        self.ClassCount = p_nClassCount
        self.ImageDimensions = p_nImageDimensions
        self.BatchSize = p_nBatchSize
        #................................................................................
    #------------------------------------------------------------------------------------
#==================================================================================================











    