from os.path import join

############################################
################CNN PART####################
############################################
ROOT_CNN_DIR = r'../CNN/DATA'
embedded_data = join(ROOT_CNN_DIR, 'embedded data')
labeled_data = join(ROOT_CNN_DIR, 'labeled data')
mapped_data = join(ROOT_CNN_DIR, 'mapped data')
mappednoembed_data = join(ROOT_CNN_DIR, 'mappednoembed data')
newembedded_data = join(ROOT_CNN_DIR, 'new_embedded data')

source_file = join(ROOT_CNN_DIR, 'source_file')

ROOT_DATASET_DIR_CNN = r'../Dataset'
antCNN = join(ROOT_DATASET_DIR_CNN, 'ant')
camelCNN = join(ROOT_DATASET_DIR_CNN, 'camel')
ivyCNN = join(ROOT_DATASET_DIR_CNN, 'ivy')
jeditCNN = join(ROOT_DATASET_DIR_CNN, 'jedit')
log4jCNN = join(ROOT_DATASET_DIR_CNN, 'log4j')
luceneCNN = join(ROOT_DATASET_DIR_CNN, 'lucene')
poiCNN = join(ROOT_DATASET_DIR_CNN, 'poi')
synapseCNN = join(ROOT_DATASET_DIR_CNN, 'synapse')
velocityCNN = join(ROOT_DATASET_DIR_CNN, 'velocity')
xalanCNN = join(ROOT_DATASET_DIR_CNN, 'xalan')
xercesCNN = join(ROOT_DATASET_DIR_CNN, 'xerces')
propCNN = join(ROOT_DATASET_DIR_CNN, 'prop')




############################################
###########STANDARD PART####################
############################################
ROOT_DATASET_DIR = r'./Dataset'
ant = join(ROOT_DATASET_DIR, 'ant')
camel = join(ROOT_DATASET_DIR, 'camel')
ivy = join(ROOT_DATASET_DIR, 'ivy')
jedit = join(ROOT_DATASET_DIR, 'jedit')
log4j = join(ROOT_DATASET_DIR, 'log4j')
lucene = join(ROOT_DATASET_DIR, 'lucene')
poi = join(ROOT_DATASET_DIR, 'poi')
synapse = join(ROOT_DATASET_DIR, 'synapse')
velocity = join(ROOT_DATASET_DIR, 'velocity')
xalan = join(ROOT_DATASET_DIR, 'xalan')
xerces = join(ROOT_DATASET_DIR, 'xerces')
prop = join(ROOT_DATASET_DIR, 'prop')


needed = ['name', 'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom',
       'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic',
       'cbm', 'amc', 'max_cc', 'avg_cc', 'bug']
features_withbug = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom',
       'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic',
       'cbm', 'amc', 'max_cc', 'avg_cc', 'bug']
features = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom',
       'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic',
       'cbm', 'amc', 'max_cc', 'avg_cc']