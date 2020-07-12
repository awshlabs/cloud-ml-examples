 #
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
    multi-GPU and multi-CPU HPO workflow    
    includes data loading, splitting, model training, and scoring/inference

    RandomForest + Dask API : 
    XGBoost + Dask API : https://xgboost.readthedocs.io/en/latest/tutorials/dask.html
"""

# CPU imports
try:
    import dask, sklearn, pandas
    from sklearn.metrics import accuracy_score as sklearn_accuracy_score
    
except: pass # Exception as error: print( f' unable to load GPU lib : {error} ')

# GPU imports
try:
    import cudf, dask_cudf, cuml, cupy
    from dask_cuda import LocalCUDACluster
    from cuml.dask.common.utils import persist_across_workers
    from cuml.metrics import accuracy_score as cuml_accuracy_score

except: pass # Exception as error: print( f' unable to load GPU lib : {error} ')

# shared imports
import sys, os, time, logging, json, pprint, argparse, glob, pickle
import warnings; warnings.simplefilter("ignore", (UserWarning, FutureWarning))

import xgboost
import numpy

from dask.distributed import wait, Client, LocalCluster

default_model_type = 'XGBoost'
default_compute_type = 'single-GPU'
default_cv_folds = 1

class SageMakerML ( object ):

    def __init__ ( self, input_args, dataset_path = '*.parquet' ):

        self.model_type, self.compute_type, self.CV_folds = parse_job_name()

        print(f'initializing cloud ML, compute: {self.compute_type}, model: {self.model_type}\n')

        # configure input/data, and model storage directories
        self.CSP_paths =  { 
            'train_data' : '/opt/ml/input/data/training/',
            'model_store' : '/opt/ml/model' 
        }

        # in the case of pandas read-parquet we need to remove the wildcards
        if 'single-CPU' in self.compute_type:
            self.target_files = self.CSP_paths['train_data'] + str(dataset_path).split('*')[0]
        else:
            self.target_files = self.CSP_paths['train_data'] + str(dataset_path)
 
        self.scores = []        
        self.best_score = -1 # monotonically increasing metric

        # parse input parameters for HPO        
        self.model_params = self.parse_hyper_parameter_inputs ( input_args )

        # determine the available compute on the node
        self.n_workers = 0
        self.cluster, self.client = self.initialize_compute(worker_limit=32)

    # -------------------------------------------------------------------------------------------------------------
    #  parse ML model parameters [ e.g., passed in by cloud HPO ]
    # -------------------------------------------------------------------------------------------------------------
    def parse_hyper_parameter_inputs ( self, input_args ):
        print('parsing model hyper-parameters from command line arguments...\n')
        parser = argparse.ArgumentParser ()

        if 'XGBoost' in self.model_type:
            parser.add_argument( '--max_depth',       type = int,   default = 6 )
            parser.add_argument( '--num_boost_round', type = int,   default = 32 )
            parser.add_argument( '--learning_rate',   type = float, default = 0.3 )    
            parser.add_argument( '--subsample',       type = float, default = .5 )
            parser.add_argument( '--lambda_l2',       type = float, default = .2 )            
            parser.add_argument( '--gamma',           type = float, default = 0. )            
            parser.add_argument( '--alpha',           type = float, default = 0. )
            
            args, unknown_args = parser.parse_known_args( input_args )
            
            model_params = {            
                'max_depth' : args.max_depth,
                'num_boost_round': args.num_boost_round,
                'learning_rate': args.learning_rate,
                'gamma': args.gamma,
                'lambda': args.lambda_l2,
                'random_state' : 0,
                'verbosity' : 0,
                'objective' : 'binary:logistic'
            }        

            if 'GPU' in self.compute_type:
                model_params.update( { 'tree_method': 'gpu_hist' })
                model_params.update( { 'single_precision_histogram': True})
            else:
                model_params.update( { 'tree_method': 'hist' })

        elif 'RandomForest' in self.model_type:
            parser.add_argument( '--max_depth',    type = int,   default = 6 )
            parser.add_argument( '--n_estimators', type = int,   default = 32 )            
            parser.add_argument( '--max_features', type = float, default = .5 )

            args, unknown_args = parser.parse_known_args( input_args )

            model_params = {            
                'max_depth' : args.max_depth,
                'n_estimators' : args.n_estimators,        
                'max_features': args.max_features,
                'seed' : 0,
            }
        else:
            raise Exception(f"!error: unknown model type {self.model_type}")

        pprint.pprint( model_params, indent = 5 ); print( '\n' )
        return model_params

    # -------------------------------------------------------------------------------------------------------------
    #  initialize compute cluster
    # -------------------------------------------------------------------------------------------------------------
    def initialize_compute(self):
        """ 
            xgboost has a known issue where training fails if any worker has no data partition
            so when initializing a dask cluster for xgboost we may need to limit the number of workers
            see 3rd limitations bullet @ https://xgboost.readthedocs.io/en/latest/tutorials/dask.html 
        """

        with PerfTimer( f'create {self.compute_type} cluster'):

            cluster = None;  client = None

            n_files = len( glob.glob(self.target_files) )
            assert( n_files > 0 )

            # initialize CPU or GPU cluster
            if 'multi-GPU' in self.compute_type:

                self.n_workers = cupy.cuda.runtime.getDeviceCount()

                if 'XGBoost' in self.model_type:
                    self.n_workers = min( n_files, self.n_workers ) 
                    
                cluster = LocalCUDACluster( n_workers = self.n_workers )
                client = Client( cluster )
                print(f'dask multi-GPU cluster with {self.n_workers} workers ')
                
            elif 'multi-CPU' in self.compute_type:
                self.n_workers = os.cpu_count()

                if 'XGBoost' in self.model_type:
                    self.n_workers = min( n_files, self.n_workers )                

                cluster = LocalCluster(  n_workers = self.n_workers,
                                         threads_per_worker = 1 )

                client = Client( cluster )
                print(f'dask multi-CPU cluster with {self.n_workers} workers')

            return cluster, client
            
    # -------------------------------------------------------------------------------------------------------------
    # ETL
    # -------------------------------------------------------------------------------------------------------------
    airline_label_column = 'ArrDel15'
    airline_columns = [ 'Year', 'Quarter', 'Month', 'DayOfWeek', 
                        'Flight_Number_Reporting_Airline', 'DOT_ID_Reporting_Airline',
                        'OriginCityMarketID', 'DestCityMarketID',
                        'DepTime', 'DepDelay', 'DepDel15', 'ArrDel15',
                        'AirTime', 'Distance' ]
    
    def ETL ( self, cached_dataset = None, random_seed = 0, 
              columns = airline_columns, label_column = airline_label_column ):
        """
            run ETL [  ingest -> rebalance -> drop missing -> split -> persist ]
            after the first run the dataset is cached, so only split is re-run (re-shuffled)
        """        
        with PerfTimer( 'ETL' ):
            if 'single' in self.compute_type:

                if 'CPU' in self.compute_type:
                    from sklearn.model_selection import train_test_split
                    dataset = pandas.read_parquet( self.target_files, columns = columns, engine='pyarrow' ) 
                    dataset = dataset.dropna()
                    X_train, X_test, y_train, y_test = train_test_split( dataset.loc[:, dataset.columns != label_column], 
                                                                         dataset[label_column], random_state = random_seed )

                elif 'GPU' in self.compute_type:
                    from cuml.preprocessing.model_selection import train_test_split
                    dataset = cudf.read_parquet( self.target_files, columns = columns )
                    dataset = dataset.dropna()                
                    X_train, X_test, y_train, y_test = train_test_split( dataset, label_column, random_state = random_seed )

                return X_train, X_test, y_train, y_test, dataset

            elif 'multi' in self.compute_type:
                from dask_ml.model_selection import train_test_split

                if cached_dataset is None:

                    if 'CPU' in self.compute_type:
                        dataset = dask.dataframe.read_parquet( self.target_files, columns = columns, engine='pyarrow') 
                    elif 'GPU' in self.compute_type:
                        dataset = dask_cudf.read_parquet( self.target_files, columns = columns )
                    
                    # drop missing values [ ~2.5% -- predominantly cancelled flights ]
                    dataset = dataset.dropna()

                    # repartition [ inplace ], rebalance ratio of workers & data partitions
                    initial_npartitions = dataset.npartitions    
                    dataset = dataset.repartition( npartitions = self.n_workers )
                    
                else:
                    print( f"using cache [ skiping ingestion, dropna, and repartition ]")                
                    if 'multi-CPU'in self.compute_type:
                        assert( type(cached_dataset) == dask.dataframe.core.DataFrame )
                    if 'multi-GPU'in self.compute_type:                    
                        assert( type(cached_dataset) == dask_cudf.core.DataFrame )

                    dataset = cached_dataset

                # split [ always runs, regardless of whether dataset is cached ]
                train, test = train_test_split( dataset, random_state = random_seed ) 

                # build X [ features ], y [ labels ] for the train and test subsets
                y_train = train[label_column].astype('int32')
                X_train = train.drop(label_column, axis = 1).astype('float32')

                y_test = test[label_column].astype('int32')
                X_test = test.drop(label_column, axis = 1).astype('float32')

                # force computation / persist
                if 'multi-CPU' in self.compute_type:
                    X_train = X_train.persist(); X_test = X_test.persist()
                    y_train = y_train.persist(); y_test = y_test.persist()

                elif 'multi-GPU' in self.compute_type:
                    workers = self.client.has_what().keys()
                    X_train, X_test, y_train, y_test = \
                        persist_across_workers( self.client,  [ X_train, X_test, y_train, y_test ],
                                                workers = workers )

                    wait( [X_train, X_test, y_train, y_test] );

                # return [ CPU/GPU ] dask dataframes 
                return X_train, X_test, y_train, y_test, dataset  
        
        return None

    # -------------------------------------------------------------------------------------------------------------
    # train
    # -------------------------------------------------------------------------------------------------------------
    def train_model ( self, X_train, y_train):
        with PerfTimer( f'training {self.model_type} classifier on {self.compute_type}'):
            
            if 'XGBoost' in self.model_type:
                
                boosting_rounds = self.model_params.pop('num_boost_round') # avoids warning messages

                if 'single' in self.compute_type:
                    dtrain = xgboost.DMatrix(data = X_train, label = y_train)
                    trained_model = xgboost.train( dtrain = dtrain, params = self.model_params, 
                                                   num_boost_round = boosting_rounds )
                elif 'multi' in self.compute_type:
                    dtrain = xgboost.dask.DaskDMatrix( self.client, X_train, y_train)
                    xgboost_output = xgboost.dask.train( self.client, self.model_params, dtrain, 
                                                        num_boost_round = boosting_rounds, 
                                                        evals=[(dtrain, 'train')] )                                                        
                    trained_model = xgboost_output['booster']

            elif 'RandomForest' in self.model_type:

                if 'GPU' in self.compute_type:
                    if 'multi' in self.compute_type:
                        from cuml.dask.ensemble import RandomForestClassifier
                    elif 'single' in self.compute_type:
                        from cuml.ensemble import RandomForestClassifier

                    rf_model = RandomForestClassifier ( n_estimators = self.model_params['n_estimators'],
                                                        max_depth = self.model_params['max_depth'],
                                                        max_features = self.model_params['max_features'],
                                                        n_bins = 32 )
                elif 'CPU' in self.compute_type:
                    from sklearn.ensemble import RandomForestClassifier
                    rf_model = RandomForestClassifier ( n_estimators = self.model_params['n_estimators'],
                                                        max_depth = self.model_params['max_depth'],
                                                        max_features = self.model_params['max_features'], 
                                                        n_jobs=-1 )

                trained_model = rf_model.fit( X_train.astype('float32'), y_train.astype('int32') )
                
        return trained_model        
    
    # -------------------------------------------------------------------------------------------------------------
    # predict / score
    # -------------------------------------------------------------------------------------------------------------
    def predict ( self, trained_model, X_test, y_test, threshold = 0.5 ):
        with PerfTimer(f'predict [ {self.model_type} ]'):
            
            if 'XGBoost' in self.model_type:              
                if 'single' in self.compute_type:  
                    dtest = xgboost.DMatrix( X_test, y_test)
                    predictions = trained_model.predict( dtest )
                    predictions = (predictions > threshold ) * 1.0

                elif 'multi' in self.compute_type:  
                    dtest = xgboost.dask.DaskDMatrix( self.client, X_test, y_test)
                    predictions = xgboost.dask.predict( self.client, trained_model, dtest).compute() 
                    predictions = (predictions > threshold ) * 1.0                    
                    y_test = y_test.compute()
                    
                if 'GPU' in self.compute_type:                
                    test_accuracy = cuml_accuracy_score ( y_test, predictions )
                elif 'CPU' in self.compute_type:
                    test_accuracy = sklearn_accuracy_score ( y_test, predictions )

            elif 'RandomForest' in self.model_type:
                if 'single' in self.compute_type:  
                    test_accuracy = trained_model.score( X_test, y_test )
                    
                elif 'multi' in self.compute_type:                    

                    if 'GPU' in self.compute_type:
                        y_test = y_test.compute()   
                        predictions = trained_model.predict( X_test ).compute()
                        test_accuracy = cuml_accuracy_score ( y_test, predictions )

                    elif 'CPU' in self.compute_type:
                        test_accuracy = sklearn_accuracy_score ( y_test, trained_model.predict( X_test ) )

            # accumulate internal list    
            self.scores += [ test_accuracy ]            
            return test_accuracy
    
    # -------------------------------------------------------------------------------------------------------------
    # emit
    # -------------------------------------------------------------------------------------------------------------
    # emit score so sagemaker can parse it (using string REGEX)
    def emit_final_score ( self ):
        print(f'self.scores = {self.scores}')
        print(f'\n\t test-accuracy: {numpy.mean(self.scores)}; \n')

    # -------------------------------------------------------------------------------------------------------------
    # save model
    # -------------------------------------------------------------------------------------------------------------
    def save_model ( self, model, output_filename='saved_model' ):
        output_filename = self.CSP_paths['model_store'] + '/' + str( output_filename )

        with PerfTimer( f'saving model into {output_filename}' ):
            if 'XGBoost' in self.model_type:
                model.save_model( output_filename )
            elif 'RandomForest' in self.model_type:
                with open(output_filename, 'wb') as pickle_output_handle:
                    pickle.dump( model, pickle_output_handle, 
                                 protocol=pickle.HIGHEST_PROTOCOL )

# use job name to define model type, compute, data
def parse_job_name():
    print('\nparsing compute & algorithm choices from job-name...\n')    
    model_type = default_model_type
    compute_type = default_compute_type
    cv_folds = default_cv_folds    

    try:
        if 'SM_TRAINING_ENV' in os.environ:
            env_params = json.loads( os.environ['SM_TRAINING_ENV'] )
            job_name = env_params['job_name']

            # compute            
            compute_selection = job_name.split('-')[1].lower()
            if 'mgpu' in compute_selection:
                compute_type = 'multi-GPU'
                
            elif 'mcpu' in compute_selection:
                compute_type = 'multi-CPU'
            elif 'scpu' in compute_selection:
                compute_type = 'single-CPU'
            elif 'sgpu' in compute_selection:
                compute_type = 'single-GPU'
            
            print(f'compute_type :: {compute_type}')

            
            # parse model type
            model_selection = job_name.split('-')[2].lower()
            if 'rf' in model_selection:
                model_type = 'RandomForest'
            elif 'xgb' in model_selection:
                model_type = 'XGBoost'
            
            # parse CV folds
            cv_folds = int(job_name.split('-')[3].split('y')[0])
            
    except Exception as error:
        print( error )

    assert ( model_type in ['RandomForest', 'XGBoost'] )
    assert ( compute_type in ['single-GPU', 'multi-GPU', 'single-CPU', 'multi-CPU'] )
    assert ( cv_folds >= 1 )
    
    return model_type, compute_type, cv_folds

# perf_counter = highest available timer resolution 
class PerfTimer:
    def __init__(self, name_string = '' ):
        self.start = None
        self.duration = None
        self.name_string = name_string

    def __enter__( self ):
        self.start = time.perf_counter()
        return self

    def __exit__( self, *args ):        
        self.duration = time.perf_counter() - self.start
        print(f"|-> {self.name_string} : {self.duration:.4f}\n")
