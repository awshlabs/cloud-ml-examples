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
    logic running in each HPO estimator
    + exception handling
    + model saving
    + spot instances
    + removed logging
    + single and multi-GPU XGBoost and RandomForest [ via dask ]
"""

import sys, os, time, traceback
from rapids_cloud_ml import SageMakerML

dataset_cache = None

if __name__ == "__main__":

    print( '\n --- start of HPO estimator experiment/HPO-run --- \n ')
    start_time = time.time()

    # parse inputs and build cluster
    rapids_sagemaker = SageMakerML( input_args = sys.argv[1:] )

    try:
        # [ optional cross-validation] improves robustness/confidence in the best hyper-params        
        for i_fold in range ( rapids_sagemaker.CV_folds ):
            
            # run ETL [  ingest -> repartition -> drop missing -> split -> persist ]
            X_train, X_test, y_train, y_test, dataset_cache = rapids_sagemaker.ETL ( dataset_cache, i_fold ) 
                        
            # train model
            trained_model = rapids_sagemaker.train_model ( X_train, y_train )

            # evaluate perf
            score = rapids_sagemaker.predict ( trained_model, X_test, y_test )
            
        # save
        rapids_sagemaker.save_model ( trained_model )
                
        # emit final score (e.g., mean across folds) to sagemaker
        rapids_sagemaker.emit_final_score()
        
        print(f'total elapsed time = { round( time.time() - start_time) } seconds ')
        sys.exit(0) # success exit code

    except Exception as error:

        trc = traceback.format_exc()           
        print( ' ! exception: ' + str(error) + '\n' + trc, file = sys.stderr)
        sys.exit(-1) # a non-zero exit code causes the training job to be marked as failed