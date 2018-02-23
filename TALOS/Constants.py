# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        CONSTANTS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================


# ----------------------------------------------------------------------
#    DataLayer Constants
# ----------------------------------------------------------------------
USE_GROUND_TRUTH_SET        = 0
USE_UNKNOWN_TEST_SET        = 1
USE_QUERY_TEST_SET          = 2
USE_FULL_DATA_SET           = 3
USE_CUSTOM_QUERY_TEST_SET   = 4
USE_SINGLE_IMAGE            = 5



# ----------------------------------------------------------------------
#    Logging Constants
# ----------------------------------------------------------------------
LOG_DATETIME_FORMAT="%Y-%m-%d %H:%M:%S"
LOG_MINUTESTAMP_FORMAT="%Y%m%d%H%M"

LOG_TYPE_GENERIC        = 0
LOG_TYPE_PROCESS        = 1
LOG_TYPE_SETTINGS       = 2
LOG_TYPE_NNTRAINING     = 3 
LOG_TYPE_NNTESTING      = 4

# ----------------------------------------------------------------------
#    Date/Time Formats 
# ----------------------------------------------------------------------
ISO_DATE_FMT_MINUTE = "%Y-%m-%d %H:%M"





# ----------------------------------------------------------------------
#    Filename Conventions 
# ----------------------------------------------------------------------
#[TO BE REFACTORED]

# Model file name constants
C_MODEL_FILE_EXTENSION       = ".nn.meta"
C_MODEL_FILE_TEMPLATE_PREFIX = "ModelEpoch"
C_MODEL_FILE_TEMPLATE = C_MODEL_FILE_TEMPLATE_PREFIX + "%03i.nn"

C_MODEL_FILE_NAME_FINAL_EPOCH_PREFIX = "ModelFinal"
C_MODEL_FILE_NAME_FINAL         = C_MODEL_FILE_NAME_FINAL_EPOCH_PREFIX + ".nn"
C_MODEL_FILE_NAME_FINAL_EPOCH   = C_MODEL_FILE_NAME_FINAL_EPOCH_PREFIX + "Epoch%03i.nn"

C_MODEL_FILE_NAME_INITIAL       = "InitialWeights.nn"
C_MODEL_FILE_NAME_NON_OVERFIT   = "ModelNonOverfit.nn"


C_EXPERIMENT_UNFISHED_FLAG_FILE_NAME = ".unfinished"
C_EXPERIMENT_SYSTEM_INFO_FILE_NAME = "system-info.txt"
C_EXPERIMENT_FOLDS_INFO_FILE_NAME = "folds-info.txt"
C_EXPERIMENT_FOLDS_UIDS_FILE_NAME = "folds-uid.txt"
C_EXPERIMENT_FOLDS_BESTS_FILE_NAME = "folds-bests.csv"    

# ----------------------------------------------------------------------
#    Padding  
# ----------------------------------------------------------------------
#[TO BE REFACTORED]
TF_ZERO_PADDING = "SAME"
TF_NO_PADDING   = "VALID"


# ----------------------------------------------------------------------
#    Neuron Activation Functions  
# ----------------------------------------------------------------------
NAF_LINEAR=1            
NAF_TANH=2
NAF_SIGMOID=3
NAF_SOFTMAX=4
NAF_RELU=5
NAF_RELU6=6
NAF_SOFTPLUS=7
NAF_SOFTSIGN=8
NAF_EXP_RELU=9
NAF_LEEKY_RELU=10
NAF_ELU=11
NAF_BN_ELU=12
NAF_RESTRICTED_LINEAR=13
NAF_PRELU=14

# ----------------------------------------------------------------------
#    Neural Network Training Constants  
# ----------------------------------------------------------------------
TRN_MOMENTUM = 1                      
TRN_SGD = 2
TRN_ADAM = 3
TRN_MOMENTUM_LR_DECAY = 4


# ----------------------------------------------------------------------
#    Normalization Types
# ----------------------------------------------------------------------
NORMALIZE_NONE              = 0
NORMALIZE_MIN_MAX           = 1
NORMALIZE_MIN_MAX_NONLINEAR = 2
NORMALIZE_LRN               = 2





# Defaults for training methods tested with a CNN on MNIST
DEFAULT_LR_TRN_MOMENTUM=0.005               #[94]   96.60%  OFR=0.0000% (Adaptive: [73] 98.6%  OFR=0.0008%)
DEFAULT_MO_TRN_MOMENTUM=0.8                 #        -||-
DEFAULT_LR_TRN_SGD=0.01                     #[173]  95.70%  OFR=0.0000% (Adaptive: [95] 97.7%  OFR=0.0000%)
DEFAULT_LR_TRN_ADAM=0.001                   #[28]   99.00%  OFR=0.0066%


# Stop condition types
DONT_STOP                       = -1
STOP_DELTA_VAL_ERROR_POSITIVE   = 0                         
STOP_DELTA_VAL_ERROR_LT         = 1
STOP_DELTA_VAL_ACCURACY_LT      = 2
STOP_DELTA_VAL_ERROR_BIG_INC    = 3
STOP_DELTA_VAL_ACCURACY_BIG_DEC = 4

STOP_SMALL_ERROR               = 10
STOP_MAX_EPOCH_REACHED         = 11
STOP_MAX_ACCURACY              = 12

STOP_STD_TRAINING_AVG_ERROR    = 20 
STOP_STD_TRAINING_AVG_ACCURACY = 21
STOP_STD_VALIDATION_ERROR      = 22

STOP_BEST_EPOCH_NOT_CHANGING   = 30
STOP_DUE_TO_OVERFITTING        = 31



#------------------------------------------------------------------------------------
def TrainingMethodTitle(p_nTrainingMethodType):
    if p_nTrainingMethodType==TRN_MOMENTUM:
        return "Momentum"
    elif p_nTrainingMethodType==TRN_SGD:
        return "SGD"
    elif p_nTrainingMethodType==TRN_ADAM:
        return "ADAM"
    elif p_nTrainingMethodType==TRN_MOMENTUM_LR_DECAY:
        return "MomentumLRDecay"
    else:    
        raise Exception("unsupported training method")  
#------------------------------------------------------------------------------------
        
        
# ----------------------------------------------------------------------
#    Data set types  
# ----------------------------------------------------------------------
DS_TRAINING=0
DS_VALIDATION=1
DS_TESTING=2




#------------------------------------------------------------------------------------
class EpochPhase(object):
    RECALL_TRAIN    = 0
    SAVE_MODEL      = 1
    RECALL_VALIDATE = 2
    RECALL_FILTER   = 3
    SAVE_STATS      = 4
    
    Count = 5
#------------------------------------------------------------------------------------
class TrainerType(object):
    UNKNOWN                         = 0
    TALOS007_SOFTMAX_CLASSIFICATION = 1007
    TALOS008_SOFTMAX_CLASSIFICATION = 1008
#------------------------------------------------------------------------------------    




        
        
        