# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        TENSORFLOW SESSION WRAPPERS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import tensorflow as tf
from TALOS.Training import NNClassificationTrainer
from TALOS.Experiments import ExperimentFolder, ClassificationEvaluator, ExperimentQueueSystem
from TALOS.Models.Factory import CNNModelFactory
from TALOS.DataSets.Factory import DataSetFactory
from TALOS.Core import SessionConfig, Exec
from TALOS.Learning.SVMBasedMargin import NNClassificationMarginSVMBased
from TALOS.Learning.SalesBasedMargin import NNClassificationMarginProfitLoss
from TALOS.Plots import LearningComparison, LearningComparisonSettings


#==================================================================================================
class TrainingSession(object):
    #------------------------------------------------------------------------------------
    def __init__(self):
        #........................ |  Instance Attributes | ..............................
        self.CommandLine = Exec.GetCommandLine()        
        #................................................................................
    #------------------------------------------------------------------------------------
    def DoTraining(self):
        pass
    #------------------------------------------------------------------------------------
    def Run(self):
        self.DoTraining()
    #------------------------------------------------------------------------------------            
#==================================================================================================    







#==================================================================================================
class CNNTrainingSession(TrainingSession):
    #------------------------------------------------------------------------------------
    def DoTraining(self):
        
        # Loads the next configuration in queue. If the queue is empty a configuration file is ensured 
        # by copying the given template to the queue
        oSystem = ExperimentQueueSystem()
        oConfig = oSystem.LoadNextConfig()
        
        if oConfig is None:
            print("No pending experiment configuration to train or evaluate")
            exit(0)
        
        # Activates the experiment and archives the configuration
        oExperiment = ExperimentFolder(p_oLearnConfig=oConfig)
        oExperiment.Activate()
        
        
        
        # Loads the dataset dictionaries 
        oDataSet = DataSetFactory.CreateVariation(  oConfig.DataSetName, oConfig.DataSetVariation
                                                   ,oExperiment.GetDataSetFolder(oConfig.DataSetName)  )
        oDataSet.Create(oExperiment.Log)
        
        # Creates the neural network definition
        oNet = CNNModelFactory.Create(oConfig.Architecture, oDataSet, oExperiment)
        
        if oNet.LearnParams.MarginType == 0:
            oNet.MarginClass = NNClassificationMarginSVMBased
        elif oNet.LearnParams.MarginType in [1,2,3]:
            oNet.MarginClass = NNClassificationMarginProfitLoss
        
        # Creates the main graph
        MainGraph = tf.Graph()
        with MainGraph.as_default():
            # Set the Tensorflow random seed for reproducibility of the training process        
            tf.set_random_seed(oExperiment.RandomSeed)
            
            # Creates the main session and the neural network in the current graph
            MainSession = tf.Session(config = SessionConfig(p_nMaxGPUMemory=2.0, p_nAvailableGPUMemory=3.5))
            oNet.Initialize(MainSession, p_bIsTrainable=True)
            
            # Proceed to training of the neural network model
            if oConfig.IsTraining:
                # Exports the graph to the experiment folder
                oNet.ExportArchitecture(MainGraph)
                
                # Creates and initializes a trainer for the network
                oTrainer = NNClassificationTrainer(oNet)
                #MarginBasedFiltering(oTrainer) 
                oTrainer.Initialize(MainSession, oExperiment.GetInitialExperiment())
        
                # Fits the model to the data
                oTrainer.Fit(oDataSet.Train)
                oExperiment.End()
            
            # Proceed to evaluation of the trained model
            if oConfig.IsEvaluating:
                if False:
                    # Render Plots   
                    oSettings = LearningComparisonSettings(oConfig)
                    if not oSettings.LoadFromFile():
                        oSettings.SetDefaults()
                        
                    oSettings.CompareWithThis(oExperiment)
                    oSettings.SaveToFile()
                
                    oPlot = LearningComparison(oSettings)
                    oPlot.Initialize()
                    oPlot.Render()    
                    
                oEvaluator = ClassificationEvaluator(oNet, p_bIsDiscardingModels=True)
                
                # Run evaluation on all saved epochs
                oIterator = oDataSet.Validation.GetIterator(None, oExperiment.BatchSize)
                #oEvaluator.IsDeterminingBestModels = False
                oEvaluator.Evaluate(oIterator)
                
                if oConfig.IsDeterminingBest:
                    oEvaluator.DetermineBestModels(oExperiment)
                        
        nCounter = oSystem.IncCounter()
        oSystem.ArchiveConfig(oConfig.FileName, oConfig.ArchiveFileName(nCounter))
    #------------------------------------------------------------------------------------
#==================================================================================================        