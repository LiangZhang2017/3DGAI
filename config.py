
import os
import pandas as pd
import numpy as np
from models.MTF.mtfRunModel import MtfRunModel
from helper import make_dir

class model_config:

    def __init__(self,args):
        self.args=args

    def generate_paradic(self):
        if self.args.LearningModel[0] =='MTF':
            course_str=self.args.Course[0]
            model_str=self.args.LearningModel[0]
            features_dim_range=list(range(2,4,1)) # latent features.
            lambda_t=0.001 # lambda_t and lambda_q are hyper-parameters to control the weights of regularization term of T and S.
            lambda_q=0.01
            lambda_bias=0.001   # can be set as False
            lambda_w=0.0001  #for rank-based TF
            lr=0.0001
            max_iter=1000
            validation = True # if the best tensor is prefered, it should be "False".
            metrics = ["rmse", "mae", "auc"]
            para={'course_str':course_str,
                  'model_str':model_str,
                  'features_dim_range':features_dim_range,
                  'lambda_t':lambda_t,
                  'lambda_q':lambda_q,
                  'lambda_bias':lambda_bias,
                  'lambda_w':lambda_w,
                  'lr':lr,
                  'max_iter':max_iter,
                  'metrics':metrics,
                  'validation':validation,
                  'learning_stage': 'Medium',
                  'is_rank':True
                  }
        return para
    def preprocessing(self):
        file_path = os.getcwd() + self.args.data_path[0] + f"/{self.args.Course[0]}" + f"/{self.args.Lesson_Id[1]}.csv"
        data = pd.read_csv(file_path)

        num_learners = data['Student_Id'].nunique()
        num_questions=data['Question_Id'].nunique()
        num_attempts=data['Attempt_Count'].nunique()

        pre_dict = {
            'exper_data': data,
            'num_learners': num_learners,
            'num_attempts': num_attempts,
            'num_questions': num_questions
        }

        return pre_dict

    def main(self):
        pre_dict=self.preprocessing()
        learning_parameters = self.generate_paradic()

        if self.args.LearningModel[0] == 'MTF':
            Mtf_config={
                'exper_data': pre_dict['exper_data'],
                'num_learners': pre_dict['num_learners'],
                'num_questions': pre_dict['num_questions'],
                'num_attempts': pre_dict['num_attempts'],
                "features_dim_range": learning_parameters['features_dim_range'],
                "is_cross_validation": learning_parameters['validation'],
                "cvt": 1,
                "cvfold": 5,
                'lambda_t':learning_parameters['lambda_t'],
                'lambda_q':learning_parameters['lambda_q'],
                'lambda_bias':learning_parameters['lambda_bias'],
                'lambda_w':learning_parameters['lambda_w'],
                'lr': learning_parameters['lr'],
                'max_iter':learning_parameters['max_iter'],
                'metrics':learning_parameters['metrics'],
                'is_rank':learning_parameters['is_rank']
            }

            is_cross_validation = Mtf_config['is_cross_validation']

            if is_cross_validation is True:
                output_path="results/{}/{}".format(self.args.LearningModel[0],self.args.Lesson_Id[1])
            else:
                output_path = "results/{}/{}/{}".format('Best',self.args.LearningModel[0], self.args.Lesson_Id[1])

            make_dir(os.getcwd() + '/SimLearnerModel/' + output_path)

            optdim_all, optmae_all, optrmse_all, optparameters_all, train_perf = MtfRunModel(Mtf_config,output_path)








