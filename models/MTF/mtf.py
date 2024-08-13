import math
import numpy as np
import warnings
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error,mean_absolute_error,roc_auc_score
from scipy.special import expit

warnings.filterwarnings("ignore")

class MTF(object):

    def __init__(self,config):

        np.random.seed(1)
        self.train_data=config['train_data']
        self.test_data=config['test_data']
        self.num_learners=config['num_learners']
        self.num_attempts=config['num_attempts']
        self.num_questions=config['num_questions']
        self.num_features=config['features_dim']
        self.lambda_t=config['lambda_t']
        self.lambda_q=config['lambda_q']
        self.lambda_bias=config['lambda_bias']
        self.lambda_w=config['lambda_w']
        self.lr=config['lr']
        self.max_iter=config['max_iter']
        self.metrics=config['metrics']
        self.is_rank=config['is_rank']

        self.use_bias_t=True
        self.use_global_bias=True
        self.binarized_question=True

        self.test_obs_list=[]
        self.test_pred_list=[]
        self.parameters=[]

        self.U=np.random.random_sample((self.num_learners,self.num_features))
        self.V=np.random.random_sample((self.num_features,self.num_attempts,self.num_questions))

        self.T=[0]

        self.bias_s=np.zeros(self.num_learners)
        self.bias_t=np.zeros(self.num_attempts)
        self.bias_q=np.zeros(self.num_questions)

        self.global_bias=np.mean(self.train_data,axis=0)[3] #The performance (Answer_Score)

    def get_question_prediction(self,learner,attempt,question):
        """
        predict value at tensor T[attempt, student, question]
        all the following indexes start from zero indexing
        :param attempt: attempt index
        :param student: student index
        :param question: question index
        :return: predicted value of tensor T[learner, attempt, question]
        """

        pred=np.dot(self.U[learner,:],self.V[:,attempt,question]) #vector*vector

        if self.use_bias_t:
            if self.use_global_bias:
                pred += self.bias_s[learner]+self.bias_t[attempt]+self.bias_q[question]+self.global_bias # MTF Equation
            else:
                pred += self.bias_s[learner]+self.bias_t[attempt]+self.bias_q[question]
        else:
            if self.use_global_bias:
                pred += self.bias_s[learner]+self.bias_q[question]+self.global_bias
            else:
                pred += self.bias_s[learner]+self.bias_q[question]

        if self.binarized_question:
            pred=expit(pred)  #Sigmoid functions most often show a return value (y axis) in the range 0 to 1.

        return pred

    def get_loss(self):
        """
        Override the function in super class
        """
        loss, square_loss, reg_bias,ranking_gain=0.,0.,0.,0.
        square_loss_q=0.
        q_count=0.

        train_obs = []
        train_pred = []

        for (learner, question, attempt, obs) in self.train_data: # obs refers the score that we observe
            learner, question, attempt, obs =int(learner),int(question),int(attempt),int(obs)
            pred=self.get_question_prediction(learner,attempt,question)
            square_loss_q+=(obs-pred)**2

            train_obs.append(obs)
            train_pred.append(pred)

            q_count+=1

        q_mae = mean_absolute_error(train_obs, train_pred)

        print("####################")

        reg_U=LA.norm(self.U)**2   # Frobenius norm, 2-norm
        reg_V=LA.norm(self.V)**2

        reg_features=self.lambda_t*reg_U+self.lambda_q*reg_V
        q_rmse=np.square(square_loss_q/q_count) if q_count!=0 else 0   #Root-mean-square deviation

        if self.lambda_bias:
            if self.use_bias_t:
                #reg_bias=self.lambda_bias*(LA.norm(self.bias_s)**2+LA.norm(self.bias_t)**2+LA.norm(self.bias_q)**2)
                reg_bias = self.lambda_bias * (LA.norm(self.bias_s)**2 + LA.norm(self.bias_t)**2 + LA.norm(self.bias_q)**2)
            else:
                reg_bias=self.lambda_bias*(LA.norm(self.bias_s)**2+LA.norm(self.bias_q)**2)
        else:
            reg_bias=0

        trans_V=np.transpose(self.V, (1, 0, 2))

        pred_tensor = np.dot(self.U, trans_V) #learner*attempt*question

        sum_n=0.
        if self.is_rank is True:
            for attempt in range(0,self.num_attempts):
                if attempt > 0:
                    for n in range(attempt-1,attempt):
                        slice_n=np.subtract(pred_tensor[:,attempt,:],pred_tensor[:,n,:])
                        slice_sig=np.log(expit(slice_n))
                        sum_n+=np.sum(slice_sig)
                    ranking_gain = self.lambda_w * sum_n
                else:
                    ranking_gain = 0.

            loss=square_loss_q+reg_features+reg_bias-ranking_gain
        else:
            loss = square_loss_q + reg_features + reg_bias

        print("Overall Loss {}".format(loss))

        metrics_all=[q_mae,q_rmse]

        return loss,metrics_all,reg_features,reg_bias

    def grad_Q_K(self,learner,attempt,question,obs=None):
        """
        compute the gradient of loss w.r.t a specific concept-question-attempts association
        of a question in Tensor,
        :param attempt: index
        :param student:  index
        :param question:  index
        :param obs: the value at Y[attempt, student, question]
        :return:
        """
        grad=np.zeros_like(self.V[:,attempt,question])
        if obs is not None:
            pred=self.get_question_prediction(learner,attempt,question)
            if self.binarized_question:
                grad=-2. *(obs-pred)*pred*(1.-pred)*self.U[learner,:]+2.* self.lambda_q*self.V[:,attempt,question]
            else:
                grad=-2. *(obs-pred)*self.U[learner,:]+2.* self.lambda_q*self.V[:,attempt,question]
        return grad

    def grad_T_ij(self,learner,attempt,question,obs=None):
        grad=np.zeros_like(self.U[learner,:])

        if obs is not None:
            pred=self.get_question_prediction(learner,attempt,question)
            if(self.binarized_question):
                grad=-2.*(obs-pred)*pred*(1.-pred)*self.V[:,attempt,question]+2.*self.lambda_t*self.U[learner,:]
            else:
                grad=-2.*(obs-pred)*self.V[:,attempt,question]+2.*self.lambda_t*self.U[learner,:]
        return grad

    def grad_bias_q(self,learner,attempt,question,obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_q
        :param attempt:
        :param student:
        :param question:
        :param obs:
        :return:
        """
        grad=0.

        if obs is not None:
            pred=self.get_question_prediction(learner,attempt,question)
            if self.binarized_question:
                grad-=2.*(obs-pred)*pred*(1.-pred)+2.*self.lambda_bias*self.bias_q[question]
            else:
                grad-=2.*(obs-pred)+2.*self.lambda_bias*self.bias_q[question]
        return grad

    def grad_bias_s(self,learner,attempt,question,obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_s
        :param attempt:
        :param student:
        :param material: material material of that resource, here is the question
        :param obs:
        :return:
        """
        grad=0.
        if obs is not None:
            pred=self.get_question_prediction(learner,attempt,question)
            if self.binarized_question:
                grad-=2.*(obs-pred)*pred*(1.-pred)+2.0*self.lambda_bias*self.bias_s[learner]
            else:
                grad-=2.*(obs-pred)+2.0*self.lambda_bias*self.bias_s[learner]
        return grad

    def grad_bias_t(self,learner,attempt,question,obs=None):
        grad=0.
        if obs is not None:
            pred=self.get_question_prediction(learner,attempt,question)
            if self.binarized_question:
                grad -=2.*(obs-pred)*pred*(1.-pred)+2.0*self.lambda_bias*self.bias_t[attempt]
            else:
                grad-=2.*(obs-pred)+2.0*self.lambda_bias*self.bias_t[attempt]

        return grad

    def grad_global_bias(self,learner,attempt,question,obs=None):
        grad=0.

        if obs is not None:
            pred=self.get_question_prediction(learner,attempt,question)
            if self.binarized_question:
                grad-=2.0*(obs-pred)*pred*(1.-pred)+2.*self.lambda_bias*self.global_bias
            else:
                grad-=2.0*(obs-pred)+2.*self.lambda_bias*self.global_bias
        return grad

    def optimize_sgd(self,learner,attempt,question,obs=None):
        """
       train the T and Q with stochastic gradient descent
       :param attempt:
       :param student:
       :param material: material material of that resource, it's question here
       :return:
        """
        grad_q=self.grad_Q_K(learner,attempt,question,obs)
        self.V[:,attempt,question]-=self.lr*grad_q
        self.V[:,attempt,question][self.V[:,attempt,question]<0.]=0.
        if self.lambda_q==0.:
            sum_val=np.sum(self.V[:,attempt,question])
            if sum_val!=0:
                self.V[:,attempt,question]/=sum_val

        grad_t=self.grad_T_ij(learner,attempt,question,obs)
        self.U[learner,:]-=self.lr*grad_t

        self.bias_q[question]-=self.lr*self.grad_bias_q(learner,attempt,question,obs)
        self.bias_s[learner]-=self.lr*self.grad_bias_s(learner,attempt,question,obs)

        if self.use_bias_t:
            self.bias_t[attempt]-=self.lr*self.grad_bias_t(learner,attempt,question,obs)

        if self.use_global_bias:
            self.global_bias-=self.lr*self.grad_global_bias(learner,attempt,question,obs)

    def testing(self,test_data):
        """
        :return: performance metrics mean squared error, RMSE, and mean absolute error
        """
        curr_pred_list = []
        curr_obs_list = []

        for(learner,question,attempt,obs) in self.test_data:
            learner, question, attempt, obs=int(learner),int(question),int(attempt),int(obs)
            curr_obs_list.append(obs)
            pred=self.get_question_prediction(learner,attempt,question)
            curr_pred_list.append(pred)
            self.test_obs_list.append(obs)
            self.test_pred_list.append(pred)

        return self.eval(curr_obs_list,curr_pred_list)

    def eval(self,obs_list,pred_list):
        """
        evaluate the prediction performance
        :param obs_list:
        :param pred_list:
        :return:
        """
        assert len(obs_list)==len(pred_list)

        count=len(obs_list)
        perf_dict={}
        if len(pred_list)==0:
            return perf_dict
        else:
            perf_dict['count']=count

        for metric in self.metrics:
            if metric=='rmse':
                rmse=mean_squared_error(obs_list,pred_list,squared=False)
                perf_dict[metric]=rmse
            elif metric=='mae':
                mae=mean_absolute_error(obs_list,pred_list)
                perf_dict[metric]=mae
        return perf_dict

    def training(self):
        #loss,q_count,q_rmse,reg_features,reg_bias =self.get_loss()
        loss,metrics_all,reg_features,reg_bias=self.get_loss()
        loss_list=[loss]

        train_perf=[]
        converge=False
        iter_num=0
        min_iter=10
        best_U,best_V =[0]*2

        while not converge:
            np.random.shuffle(self.train_data)
            best_U=np.copy(self.U)
            best_V=np.copy(self.V)
            best_bias_s=np.copy(self.bias_s)
            best_bias_t=np.copy(self.bias_t)
            best_bias_q=np.copy(self.bias_q)

            for(learner, question, attempt, obs) in self.train_data:
                learner, question, attempt, obs =int(learner),int(question),int(attempt),int(obs)
                self.optimize_sgd(learner,attempt,question,obs)

            sorted_train_data=sorted(self.train_data,key=lambda x:[x[0],x[1]])

            # for(learner,question,attempt,obs) in sorted_train_data:
            #     if attempt<self.num_attempts-1:
            #         self.V[:,attempt+1,question]=2*self.V[:,attempt,question]+\
            #             np.true_divide(2*(1-self.V[:,attempt,question]),1+np.exp(-self.lr*self.V[learner,:]))-1

            loss,metrics_all,reg_features,reg_bias=self.get_loss()
            train_perf.append([metrics_all[0],metrics_all[1]])

            if iter_num==self.max_iter:
                loss_list.append(loss)
                converge=True
            elif iter_num>=min_iter and loss >=np.mean(loss_list[-5:]):
                converge=True
            elif loss==np.nan:
                self.lr*=0.1
            elif loss>loss_list[-1]:
                loss_list.append(loss)
                self.lr*=0.5
                iter_num+=1
            else:
                loss_list.append(loss)
                iter_num+=1

        self.U=best_U
        self.V=best_V
        self.bias_s=best_bias_s
        self.bias_t=best_bias_t
        self.bias_q=best_bias_q

        trans_V = np.transpose(self.V, (1, 0, 2))

        T=np.dot(self.U,trans_V)+self.global_bias
        for i in range(0,self.num_learners):
            T[i,:,:]=T[i,:,:]+self.bias_s[i]
        for j in range(0,self.num_attempts):
            T[:,j,:]=T[:,j,:]+self.bias_t[j]
        for k in range(0,self.num_questions):
            T[:,:,k]=T[:,:,k]+self.bias_q[k]

        T=np.where(T > 100, 1, T)
        T=np.where(T < -100, 0, T)

        T = expit(T)

        self.T=T

        self.parameters.append([self.bias_s])
        self.parameters.append([self.bias_t])
        self.parameters.append([self.bias_q])
        self.parameters.append([self.global_bias])

        return train_perf[-1]
