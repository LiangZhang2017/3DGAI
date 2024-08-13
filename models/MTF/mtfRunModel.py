
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import os
import pandas as pd
import torch

from models.MTF.mtf import MTF

def MtfRunModel(model_config,output_path):
    is_cross_validation = model_config['is_cross_validation']
    CV = model_config['cvfold']
    exper_data = model_config['exper_data']

    exper_data['Student_Id'] = exper_data['Student_Id'] - 1
    exper_data['Attempt_Count']=exper_data['Attempt_Count']-1
    exper_data['Question_Id'] = exper_data['Question_Id'] - 1

    print("exper_data['Attempt_Count'] is", exper_data)

    cvt = model_config['cvt']
    cv_num = 0

    optdim_all = []
    optmae_all = []
    optrmse_all = []
    optparameters_all = []

    features_dim_range = model_config['features_dim_range']

    if is_cross_validation is True:
        studentIds = np.unique(exper_data['Student_Id'])

        for cv in range(1, cvt + 1):
            kf = KFold(n_splits=CV, random_state=cv, shuffle=True)
            for k, (train, test) in enumerate(kf.split(studentIds)):
                print("The k is {}".format(k))
                # print("train is {}".format(len(train)))
                # print("test is {}".format(len(test)))

                correc_rec = []
                eval_result_rec = []
                parameters_rec = []
                dim_rec = []

                train_perf_list = []

                train_data = exper_data.loc[exper_data['Student_Id'].isin(studentIds[train])].to_numpy()
                test_data = exper_data.loc[exper_data['Student_Id'].isin(studentIds[test])].to_numpy()

                print("exper_data headers is {}".format(exper_data))

                model_config['train_data'] = train_data
                model_config['test_data'] = test_data

                for features_dim in features_dim_range:
                    model_config['features_dim'] = features_dim

                    cv_num = cv_num + 1
                    model = MTF(model_config)

                    print("Start of Training")
                    train_perf = model.training()
                    print("End of Training")

                    train_perf_list.append(train_perf)

                    test_perf = model.testing(test_data)
                    print("Result about the test_perf")

                    dim_rec.append(features_dim)
                    correc_rec.append(test_perf['rmse'])
                    eval_result_rec.append([test_perf['mae'], test_perf['rmse']])
                    parameters_rec.append([model.T, model.U, model.V, model.parameters])

                ord = np.argsort(np.array(correc_rec) * (-1))
                optIndex = ord[0]

                optdim = dim_rec[optIndex]
                opteval = eval_result_rec[optIndex]
                optparameters = parameters_rec[optIndex]

                optmae_all.append(opteval[0])
                optrmse_all.append(opteval[1])
                optparameters_all.append(optparameters)

        return optdim_all, optmae_all, optrmse_all, optparameters_all, train_perf_list

    else:
        '''
        Use all exper_data as the baseline matrix for sampling
        '''
        learning_stage = model_config['learning_stage']

        model_config['train_data'] = exper_data.to_numpy(copy=True)
        model_config['test_data'] = None

        train_perf_list = []

        # save the original data to Tensor mode
        num_learner = model_config['num_learner']
        num_attempts = model_config['num_attempts']
        num_questions = model_config['num_questions']

        raw_T = np.full((num_learner, num_attempts, num_questions), np.nan)

        for (learner, question, attempt, obs) in model_config['train_data']:
            raw_T[learner, attempt, question] = obs

        for features_dim in features_dim_range:
            model_config['features_dim'] = features_dim
            model = MTF(model_config)

            print("Start of Training")
            train_perf = model.training()
            print("End of Training")

            train_perf_list.append(train_perf)

            optdim_all.append(features_dim)
            optmae_all.append(train_perf[0])
            optrmse_all.append(train_perf[1])

            saveBestTrain(raw_T, model.T, model.U, model.V, model.bias_s, model.bias_t, model.bias_q, model.global_bias,
                          output_path, learning_stage)

        return optdim_all, optmae_all, optrmse_all, optparameters_all, train_perf_list


def saveBestTrain(raw_T, T, U, V, bias_s, bias_t, bias_q, global_bias, output_path, learning_stage):
    print("Save the best matrix")

    saveBestTrainPath = os.getcwd() + '/SimLearnerModel/' + output_path

    print("saveBestTrainPath is {}".format(saveBestTrainPath))

    torch.save(raw_T, saveBestTrainPath + '/RawT_tensor{}.pt'.format(learning_stage))
    torch.save(T, saveBestTrainPath + '/T_tensor{}.pt'.format(learning_stage))
    torch.save(U, saveBestTrainPath + '/U_tensor{}.pt'.format(learning_stage))
    torch.save(V, saveBestTrainPath + '/V_tensor{}.pt'.format(learning_stage))
    torch.save(global_bias, saveBestTrainPath + '/global_bias_tensor{}.pt'.format(learning_stage))
    torch.save(bias_s, saveBestTrainPath + '/bias_s_tensor{}.pt'.format(learning_stage))
    torch.save(bias_t, saveBestTrainPath + '/bias_t_tensor{}.pt'.format(learning_stage))
    torch.save(bias_q, saveBestTrainPath + '/bias_q_tensor{}.pt'.format(learning_stage))