from utils.tools import dotdict
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np



fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


arg = dotdict()

# NEW OPTIONS : #

arg.scale = True

arg.test_size = 0.2                 # default is 0.2 which makes the training 0.7 ! #
arg.kind_of_scaler = 'Standard'            # default is 'Standard'. Another Option is 'MinMax' (recommended) #
arg.name_of_col_with_date = 'date'     # default is 'date'. Name of your date column in your dataset #

arg.kind_of_optim = 'default'        # default is 'Adam'.
                                    #other options : 'AdamW', 'SparseAdam', 'SGD', 'RMSprop', 'RAdam', 'NAdam' ,'LBFGS',
                                    # 'Adamax' 'ASGD' 'Adadelta' 'Adagrad'

import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np

# Setting the seed
fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)




# NEW OPTIONS : #
arg.scale = True
arg.test_size = 0.2
arg.kind_of_scaler = 'MinMax'
arg.name_of_col_with_date = 'date'
arg.kind_of_optim = 'default'
arg.criter = 'default'
arg.do_visual = False
arg.max_use_of_row = 'No Lim'#It also can be 'All Except a Week' or 'All Except 3 Days'
#        #      #

arg.is_training = 1
arg.model_id = 'test12'
arg.model = 'LSTM'
arg.data = 'custom'

arg.features = 'S'

arg.target = 'Close'
arg.freq = 'b'
arg.checkpoints = './checkpoints/'



arg.seq_len = 5
arg.label_len = 1
arg.pred_len = 1



arg.enc_in = 1
arg.dec_in = 1
arg.c_out = 1


arg.d_model = 512
arg.n_heads = 4
arg.e_layers = 2
arg.d_layers = 128
arg.d_ff = 128
arg.moving_avg = 10
arg.factor = 1
arg.distil = True
arg.dropout = 0.01
arg.embed = 'timeF'
arg.activation = 'geLU'
arg.num_workers = 1
arg.itr = 1
arg.train_epochs = 20
arg.batch_size = 16
arg.patience = 5
arg.learning_rate = 0.00001
arg.des = 'test'
arg.loss = 'MSE'
arg.lradj = 'type1'
arg.use_amp = False
arg.use_gpu = True if torch.cuda.is_available() else False
arg.gpu = 0
arg.use_multi_gpu = False
arg.devices = '0,1,2,3'
arg.exp_name = 'MTSF'
arg.channel_independence = False
arg.inverse = False
arg.class_strategy = 'projection'
arg.efficient_training = False
arg.use_norm = True
arg.partial_start_index = 0


# LSTM specific arguments
arg.input_size=1
arg.hidden_size=64
arg.output_size=1
arg.num_layers=1
arg.dropout=0.001




print('Args in experiment:')
print(arg)



if input("Press Enter To Start :" ) == '' :
    pass
else:
    exit()

if arg.exp_name == 'partial_train':                                         # See Figure 8 of our paper, for the detail
    Exp = Exp_Long_Term_Forecast_Partial
else:                                                                       # MTSF: multivariate time series forecasting
    Exp = Exp_Long_Term_Forecast

if arg.is_training:
    for ii in range(arg.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                arg.model_id,
                arg.model,
                arg.data,
                arg.features,
                arg.seq_len,
                arg.label_len,
                arg.pred_len,
                arg.d_model,
                arg.n_heads,
                arg.e_layers,
                arg.d_layers,
                arg.d_ff,
                arg.factor,
                arg.embed,
                arg.distil,
                arg.des,
                arg.class_strategy, ii)
        
        exp = Exp(arg)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        
        train_losses = exp.train_losses##### --->>> Use These To Plot the Loss Values
        test_losses = exp.test_losses####   --->>> Use These To Plot the Loss Values
        
        exp.test(setting)
        
        if arg.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)
        
        torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            arg.model_id,
            arg.model,
            arg.data,
            arg.features,
            arg.seq_len,
            arg.label_len,
            arg.pred_len,
            arg.d_model,
            arg.n_heads,
            arg.e_layers,
            arg.d_layers,
            arg.d_ff,
            arg.factor,
            arg.embed,
            arg.distil,
            arg.des,
            arg.class_strategy, ii)
        
        exp = Exp(arg)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

#end#
