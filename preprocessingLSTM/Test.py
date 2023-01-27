from dataset import VidFMRIDataset,VidDataset
from torch.utils.data import DataLoader
import os
import numpy as np

import torch
import timm
import tqdm
import glob
import pickle
from runner import runner_gpu as runner
from runner.loss import WeightedMSELoss
from runner.metric import vectorized_correlation
from model import Vid2FMRIModel
import runner.utils as runner_utils
from sklearn.preprocessing import StandardScaler
import settings


def append_to_history(history_dict, target, output, loss, score):
    history_dict["loss"].append(loss)
    history_dict["score"].append(score)
    history_dict["outputs"].append(output)
    history_dict["targets"].append(target)

def train_and_validate(args):
    # SEED and split
    runner_utils.seed_everything(42)

    train_vid_files = args.vid_files[0:900]
    valid_vid_files = args.vid_files[900:1000]
    train_fmri_data = args.fmri_data[0:900, :, :]
    valid_fmri_data = args.fmri_data[900:1000, :, :]


    # scaler_model = StandardScaler()
    # for dir in args.vid_files:
    # #print(list_dir[i*n:(i+1)*n])
    #     y = []
    #     x = np.load(dir)
    #     y.append(x)
    #     scaler_model.partial_fit(np.asarray(y))
    # # for vid_fn in args.vid_files:
    # #   scaler_model.transform(np.asarray([np.load(vid_fn)]))[0]

    # create model
    model = Vid2FMRIModel(num_of_features=args.num_of_features,embed_size=args.embed_size, output_size=args.output_size, rnn_features=args.rnn_features, dropout_rate=args.dropout_rate)
    model = model.to(args.device)
    #train data loader
    train_dataset = VidFMRIDataset(train_vid_files, train_fmri_data,fmri_transform="combination_augment")
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    # valid loader
    valid_dataset = VidFMRIDataset(valid_vid_files, valid_fmri_data,  fmri_transform="mean_over_rep")
    valid_loader = DataLoader(valid_dataset, num_workers=args.num_workers, batch_size=1, shuffle=False)
    # loss, optimizer and scheduler
    train_criterion = WeightedMSELoss(reduction="mean")
    valid_criterion = torch.nn.MSELoss()
    optimizer = args.optimizer(model.parameters(), lr=args.lr, **args.optimizer_params)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=args.lr,
                    epochs=args.epochs,
                    steps_per_epoch=len(train_loader),
                    div_factor=10,
                    final_div_factor=1,
                    pct_start=0.1,
                    anneal_strategy="cos",
                )
    # history dicts
    train_history = {"loss": [], "score": [], "outputs": [], "targets": []}
    valid_history = {"loss": [], "score": [], "outputs": [], "targets": []}
    model_history = {"best": {"state" : None, "score": 0, "epoch": -1}, 
                     "last": {"state" : None, "score": 0, "epoch": -1}}
    
    # train - validate - save
    for epoch in range(1, args.epochs +1):
        targets_all, outputs_all, loss, score = runner.train_epoch(args, model, train_loader, train_criterion, optimizer, scheduler, epoch)
        append_to_history(train_history, None, None, loss, score)
        runner_utils.print_score(outputs_all, targets_all ,"\n\t")

        targets_all, outputs_all, loss, score = runner.validate(args, model, valid_loader, valid_criterion)
        append_to_history(valid_history, targets_all, outputs_all, loss, score)
        runner_utils.print_score(outputs_all, targets_all, "\t")

        model_history["last"]["state"] = model.state_dict()
        model_history["last"]["score"] = score
        model_history["last"]["epoch"] = epoch

        if (score > model_history["best"]["score"]):
            model_history["best"]["state"] = model.state_dict()
            model_history["best"]["score"] = score
            model_history["best"]["epoch"] = epoch

        #output_history = {"train": train_history, "valid": valid_history, "mapping": args.fmri_mapping}
        # data_handler.save_dict(output_history, args.output_valid_fn)
        torch.save(model_history, args.output_model_fn)
    

    args.model_history = model_history
    #args.output_history = output_history

def test_and_submit(args):
    runner_utils.seed_everything(42)

    model = Vid2FMRIModel(num_of_features=args.num_of_features,embed_size=args.embed_size, output_size=args.output_size, rnn_features=args.rnn_features, dropout_rate=args.dropout_rate)
    model = model.to(args.device)
    model_state = torch.load(args.output_model_fn)
    model_state = model_state["last"]["state"]
    model.load_state_dict(model_state)
    test_dataset = VidDataset(args.vid_files)
    test_loader  = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=1, shuffle=False)
    t = test_loader
    model.eval()
    outputs_all = np.zeros((args.vid_files.shape[0],args.output_size))
    with torch.no_grad():
        for i, sample in enumerate(t):
            input  = sample['vid_data'].to(args.device)
            output = model(input)
            outputs_all[i]=output
            print(i,sub)
    np.save(args.results_dir,np.asarray(outputs_all))

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        ret_di = u.load()
    return ret_di

def run(sub,roi):

    class base_args:
        embed_size = 1024
    
        rnn_features=False
        dropout_rate=0.2
        lr = 1e-3
        epochs = 4
        batch_size = 4
        num_workers = 2
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.AdamW
        optimizer_params = {
            "weight_decay": 0.02
        }
    args_test = base_args()
    args_test.vid_dir = "alexnet/"
    args_test.fmri_dir = "participants_data_v2021/mini_track/"+sub+"/"+roi+".pkl"
    args_test.video_list = glob.glob(args_test.vid_dir + '/*_layer_5.npy')
    args_test.video_list.sort()
    args_test.vid_files = np.asarray(args_test.video_list[:1000])
    args_test.test_vid_files = np.asarray(args_test.video_list[1000:])
    args_test.fmri_data = load_dict(args_test.fmri_dir)['train']
    
    args_test.num_of_features = np.load(args_test.vid_files[0]).shape[1]
    args_test.embed_size  = args_test.num_of_features
    print(np.load(args_test.vid_files[0]).shape,"Sssss")
    args_test.output_size=args_test.fmri_data.shape[2]
    args_test.output_model_fn = "output/"+sub+"_"+roi+".pt"
    
    train_and_validate(args_test)

def test(sub,roi):
    class base_args:
        embed_size = 1024
    
        rnn_features=True
        dropout_rate=0.2
        lr = 1e-3
        epochs = 4
        batch_size = 10
        num_workers = 4
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.AdamW
        optimizer_params = {
            "weight_decay": 0.02
        }
    args_test = base_args()
    args_test.vid_dir = "alexnet/"
    args_test.fmri_dir = "participants_data_v2021/mini_track/"+sub+"/"+roi+".pkl"
    args_test.video_list = glob.glob(args_test.vid_dir + '/*_layer_5.npy')
    args_test.video_list.sort()
    args_test.vid_files = np.asarray(args_test.video_list[1000:])
    args_test.fmri_data = load_dict(args_test.fmri_dir)['train']
    
    args_test.num_of_features = np.load(args_test.vid_files[0]).shape[1]
    print(np.load(args_test.vid_files[0]).shape,"Sssss")
    
    args_test.embed_size=args_test.num_of_features
    args_test.output_size=args_test.fmri_data.shape[2]
    args_test.output_model_fn = "output/"+sub+"_"+roi+".pt"
    results_dir = os.path.join('./results','alexnet_devkit', 'layer_5', "mini_track", sub)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    args_test.results_dir = os.path.join(results_dir,roi+'_test.npy')
    test_and_submit(args_test)


if __name__ == '__main__':
    run("sub01","LOC")
    # for ROI in settings.ROIs:
    #     for sub in settings.subs:  
    #         run(sub,ROI)
    # for sub in settings.subs:   
    #     test(sub,'LOC')
    # model_state = torch.load("output/"+"sub01"+"_"+"LOC"+".pt")
    # print(model_state["last"]["state"])
    # print(model_state["best"]["state"])
    ##print(np.load(os.path.join('./results','alexnet_devkit', 'layer_5', "mini_track", "sub01","LOC_test.npy")).shape)
