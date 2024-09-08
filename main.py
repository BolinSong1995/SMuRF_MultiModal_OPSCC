import os
import torch
import pickle

import numpy as np
import torch.backends.cudnn as cudnn

from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm.auto import tqdm
# from torchviz import make_dot
from losses import MultiTaskLoss, MMOLoss
from datasets import RadPathDataset, PathDataset, RadDataset, custom_collate, custom_collate_pathology
from models import FusionModelBi, Model, RModel, PModel
from utils import *
from parameters import parse_args
import scipy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

def one_epoch(args, split, model, optim, loader, criterion):
    if split == "train":
        model.train()
    else:
        model.eval()
    total = 0
    sum_loss = 0
    all_preds_grade = []
    all_preds_hazard = []
    all_grade = []
    all_time = []
    all_event = []
    all_ID = []

    

    if type(criterion) == list:
        mmo_criterion = criterion[-1]
        criterion = criterion[0]
    if args.feature_type == "radiology":
        # for batch in loader:
        #     print(batch[3].shape)
        for i, (mod1, mod2, grade, time, event, ID) in enumerate(loader):
            mod1, mod2, grade, time, event = mod1.to(device), mod2.to(device), grade.to(device), time.to(device), event.to(device)
            batch = mod1.shape[0]
            if type(criterion) == list:
                loss_mmo = args.gamma * mmo_criterion(mod1, mod2)
            else:
                loss_mmo = 0
            model.to(device)
            pred = model(mod1, mod2)
            
            if args.batch_size==1:
                if args.task == "multitask":
                    pred_grade, pred_hazard = pred
                elif args.task == "grade":
                    pred_grade, pred_hazard = pred[0], torch.empty(1)
                elif args.task == "survival":
                    pred_grade, pred_hazard = torch.empty(1), pred[0]
                else:
                    raise NotImplementedError(
                        f'task method {args.task} is not implemented')
            else:
                if args.task == "multitask":
                    pred_grade, pred_hazard = pred
                elif args.task == "grade":
                    pred_grade, pred_hazard = pred.squeeze(), torch.empty(1)
                elif args.task == "survival":
                    pred_grade, pred_hazard = torch.empty(1), pred.squeeze()
                else:
                    raise NotImplementedError(
                        f'task method {args.task} is not implemented')
            
            # print(pred_grade, grade)
            loss_task = criterion(args.task, pred_grade,
                                pred_hazard, grade, time, event)
            loss = loss_task + args.gamma*loss_mmo
            if split == 'train':
                optim.zero_grad()
                loss.backward()
                optim.step()

            total += batch
            sum_loss += batch * (loss.item())
            all_preds_grade.append(pred_grade)
            all_preds_hazard.append(pred_hazard)
            all_grade.append(grade)
            all_time.append(time)
            all_event.append(event)
            all_ID.append(ID)
            

        all_grade = torch.concat(all_grade)
        all_time = torch.concat(all_time)
        all_event = torch.concat(all_event)
        # print(all_ID)
        # print(all_time)
        # print(all_event)
        # print(all_preds_hazard)


    elif args.feature_type == "pathology":
        for i, (mod3, grade, time, event, ID) in enumerate(loader):

            grade, time, event = grade.to(device), time.to(device), event.to(device)
            mod3 = torch.reshape(mod3, (mod3.shape[0]*mod3.shape[1], mod3.shape[2], mod3.shape[3], mod3.shape[4])).to(device)

            batch = grade.shape[0]
            
            loss_mmo = 0
            
            # print(batch)
            model.to(device)
            pred = model(mod3, batch)
            
            if args.batch_size==1:
                if args.task == "multitask":
                    pred_grade, pred_hazard = pred
                elif args.task == "grade":
                    pred_grade, pred_hazard = pred[0], torch.empty(1)
                elif args.task == "survival":
                    pred_grade, pred_hazard = torch.empty(1), pred[0]
                else:
                    raise NotImplementedError(
                        f'task method {args.task} is not implemented')
            else:
                if args.task == "multitask":
                    pred_grade, pred_hazard = pred
                elif args.task == "grade":
                    pred_grade, pred_hazard = pred.squeeze(), torch.empty(1)
                elif args.task == "survival":
                    pred_grade, pred_hazard = torch.empty(1), pred.squeeze()
                else:
                    raise NotImplementedError(
                        f'task method {args.task} is not implemented')
            
            loss_task = criterion(args.task, pred_grade,pred_hazard, grade, time, event)
            
            loss = loss_task + args.gamma*loss_mmo

            if split == 'train':
                optim.zero_grad()
                loss.backward()
                optim.step()

            total += batch
            sum_loss += batch * (loss.item())
            all_preds_grade.append(pred_grade)
            all_preds_hazard.append(pred_hazard)
            all_grade.append(grade)
            all_time.append(time)
            all_event.append(event)
            all_ID.append(ID)
    

        all_grade = torch.concat(all_grade)
        all_time = torch.concat(all_time)
        all_event = torch.concat(all_event)
        # print(all_ID)
        # print(all_time)
        # print(all_event)
        # print(all_preds_hazard)

    else:
        for i, (mod1, mod2, mod3, grade, time, event, ID) in enumerate(loader):
            # if i%1==0:
            #     print(f"Sample {i}/{len(loader)}")

            mod1, mod2, grade, time, event = mod1.to(device), mod2.to(device), grade.to(device), time.to(device), event.to(device)
            mod3 = torch.reshape(mod3, (mod3.shape[0]*mod3.shape[1], mod3.shape[2], mod3.shape[3], mod3.shape[4])).to(device)

            batch = mod1.shape[0]
            if type(criterion) == list:
                loss_mmo = args.gamma * mmo_criterion(mod1, mod2)
            else:
                loss_mmo = 0
            
            # print(batch)
            model.to(device)
            pred = model(mod1, mod2, mod3, batch)
            
            if args.batch_size==1:
                if args.task == "multitask":
                    pred_grade, pred_hazard = pred
                elif args.task == "grade":
                    pred_grade, pred_hazard = pred[0], torch.empty(1)
                elif args.task == "survival":
                    pred_grade, pred_hazard = torch.empty(1), pred[0]
                else:
                    raise NotImplementedError(
                        f'task method {args.task} is not implemented')
            else:
                if args.task == "multitask":
                    pred_grade, pred_hazard = pred
                elif args.task == "grade":
                    pred_grade, pred_hazard = pred.squeeze(), torch.empty(1)
                elif args.task == "survival":
                    pred_grade, pred_hazard = torch.empty(1), pred.squeeze()
                else:
                    raise NotImplementedError(
                        f'task method {args.task} is not implemented')
            
            loss_task = criterion(args.task, pred_grade,pred_hazard, grade, time, event)
            loss = loss_task + args.gamma*loss_mmo
            
            if split == 'train':
                optim.zero_grad()
                loss.backward()
                optim.step()

            total += batch
            sum_loss += batch * (loss.item())
            all_preds_grade.append(pred_grade)
            all_preds_hazard.append(pred_hazard)
            all_grade.append(grade)
            all_time.append(time)
            all_event.append(event)
            all_ID.append(ID)
            

        all_grade = torch.concat(all_grade)
        all_time = torch.concat(all_time)
        all_event = torch.concat(all_event)
        # print(all_ID)
        # print(all_time)
        # print(all_event)
        # print(all_preds_hazard)
    
    if args.task == "grade" :
        all_preds_grade = torch.concat(all_preds_grade)
        return sum_loss / total, (all_preds_grade, None, all_grade, all_time, all_event, all_ID)
    elif args.task == "multitask":
        all_preds_grade = torch.concat(all_preds_grade)
        all_preds_hazard = torch.concat(all_preds_hazard)
        return sum_loss / total, (all_preds_grade, all_preds_hazard, all_grade, all_time, all_event, all_ID)
    else: 
        all_preds_hazard = torch.concat(all_preds_hazard)
        return sum_loss / total, (None, all_preds_hazard, all_grade, all_time, all_event, all_ID)
    

def train_model(args, data, model, criterion, optim, scheduler, train_index, test_index, device):

    if args.feature_type == "radiology":
        train_set = RadDataset(
            data, args.dataroot, index=train_index)
        val_set = RadDataset(data, args.dataroot, index=test_index)
        train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    elif args.feature_type == "pathology":
        train_set = PathDataset(
            data, args.dataroot, index=train_index)
        val_set = PathDataset(data, args.dataroot, index=test_index)
        train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn = custom_collate_pathology)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, collate_fn = custom_collate_pathology)

    else:
        train_set = RadPathDataset(
            data, args.dataroot, index=train_index)
        val_set = RadPathDataset(data, args.dataroot, index=test_index)
        train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    

    metric_logger = {'train': {'loss': [], 'cindex': [], 'grad_auc': []},
                     'val': {'loss': [], 'cindex': [], 'grad_auc': []}}
    
    torch.cuda.manual_seed_all(2023)
    torch.manual_seed(2023)
    np.random.seed(2023)

    best_loss = 20  # Initialize to a large value
    for epoch in tqdm(range(args.epoch_count, args.niter+args.n_epochs+1)):

        # Get the current learning rate
        current_lr = optim.param_groups[0]['lr']
        print(f"Epoch {epoch}, Learning rate: {current_lr}")
        
        loss, preds = one_epoch(args,
                                "train", model, optim, train_loader, criterion)
        scheduler.step()
        vloss, vpreds = one_epoch(args,
                                    "val", model, None, val_loader, criterion)
        
        # writer.add_scalars('raptomic_30epoch', {'Train':loss,
        #                         'Validation':vloss}, epoch)

        if epoch % args.print_freq == 0:
            print(f"epoch {epoch}")

            ci_train, auc_train = compute_metrics(args, preds)
            metric_logger['train']['loss'].append(loss)
            metric_logger['train']['cindex'].append(ci_train)
            metric_logger['train']['grad_auc'].append(auc_train)

            print(f"Training loss = {loss}")
            print(f"Train C-index (survival) = {ci_train}")
            print(f"Train AUC (grade) = {auc_train}")

            ci_val, auc_val = compute_metrics(args, vpreds)
            metric_logger['val']['loss'].append(vloss)
            metric_logger['val']['cindex'].append(ci_val)
            metric_logger['val']['grad_auc'].append(auc_val)

            print(f"Validation loss = {vloss}")
            print(f"Val C-index (survival) = {ci_val}")
            print(f"Val AUC (grade) = {auc_val}")

            
            if (epoch > 10) and (vloss < best_loss):
                best_loss = vloss

                torch.save({
                'args': args,
                'epoch': epoch,
                'model_state_dict': model.cpu().state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'metrics': metric_logger,
                'train_loss': loss,
                'train_pred': preds,
                'val_loss': vloss,
                'val_pred': vpreds},
                os.path.join(args.checkpoints_dir, args.exp_name, model_name, f'{model_name}_best_loss.pt'))
    # writer.close()

    return model, optim, metric_logger


def train_cv(args, device):

    model_name = args.fusion_type+'_'+args.task+'_'+str(args.n_epochs)+'_'+str(args.lr)+'_'+str(args.feature_type)
    cv = StratifiedKFold(
        n_splits=args.cv_splits,
        shuffle=True,
        random_state=2023,
    )
    criterion = MultiTaskLoss()
    if args.mmo_loss:
        criterion = [MultiTaskLoss(), MMOLoss()]
    if args.feature_type == "handcrafted":
        data = extract_csv(os.path.join(
            args.dataroot, "data_table_output.csv"))
        features, labels = data.iloc[:, :14], data.iloc[:, 14:]
    results_ci = []
    results_auc = []
    for split, (train_index, test_index) in enumerate(cv.split(features, labels['grade'])):
        cudnn.deterministic = True
        torch.cuda.manual_seed_all(2023)
        torch.manual_seed(2023)
        np.random.seed(2023)
        print("*******************************************")
        print(f"************** FOLD {split} **************")
        print("*******************************************")
        dim_in = features.shape[1]
        if args.fusion_type == "concatenation":
            model = FusionModelBi(args, dim_in, args.dim_out)
        else:
            model = FusionModelBi(args, dim_in//2, args.dim_out)
        optim = define_optimizer(args, model)
        scheduler = define_scheduler(args, optim)
        if split == 0:
            # print(model)
            print("Number of Trainable Parameters: %d" %
                  count_parameters(model))
            print("Optimizer Type:", args.optimizer_type)
            print("Activation Type:", args.act_type)

        model, optim, metric_logger = train_model(
            args, data, model, criterion, optim, scheduler, train_index, test_index, device)

        loss_train, ci_train, auc_train, preds_train, loss_val, ci_val, auc_val, preds_val = train_model(
            args, data, model, criterion, optim, scheduler, train_index, test_index, device, inference=True)

        print(
            f"[Final] Apply model to training set: Loss = {loss_train}, C-Index = {ci_train}, AUC grade= {auc_train}")
        print(
            f"[Final] Apply model to validation set: Loss = {loss_val}, C-Index = {ci_val}, AUC grade= {auc_val}")
        results_ci.append(ci_val)
        results_auc.append(auc_val)
        if args.gpu_ids:
            model_state_dict = model.module.cpu().state_dict()
        else:
            model_state_dict = model.cpu().state_dict()

        torch.save({
            'split': split,
            'args': args,
            'epoch': args.n_epochs,
            'data': data,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optim.state_dict(),
            'metrics': metric_logger},
            os.path.join(args.checkpoints_dir, args.exp_name, model_name, f'{model_name}_{split}.pt'))

        pickle.dump(preds_train, open(os.path.join(args.checkpoints_dir+'_cv', args.exp_name,
                    model_name, f'{model_name}_{split}_pred_train.pkl'), 'wb'))
        pickle.dump(preds_val, open(os.path.join(args.checkpoints_dir+'_cv', args.exp_name,
                    model_name, f'{model_name}_{split}_pred_test.pkl'), 'wb'))

    print('Split Results C-Index:', results_ci)
    print("Average:", np.array(results_ci).mean())
    pickle.dump(results_ci, open(os.path.join(args.checkpoints_dir+'_cv', args.exp_name,
                model_name, f'{model_name}_results_ci.pkl'), 'wb'))
    print('Split Results AUC grade:', results_auc)
    print("Average:", np.array(results_auc).mean())
    pickle.dump(results_auc, open(os.path.join(args.checkpoints_dir+'_cv', args.exp_name,
                model_name, f'{model_name}_results_auc.pkl'), 'wb'))

    return

def test(args, device):
    model_name = args.fusion_type+'_'+args.task+'_'+str(args.n_epochs)+'_'+str(args.lr)+'_'+str(args.feature_type)
    criterion = MultiTaskLoss()
    data = extract_csv(os.path.join(
        args.dataroot, "data_table_output_test.csv"))
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(2023)
    torch.manual_seed(2023)
    np.random.seed(2023)
    checkpoint = torch.load(os.path.join(args.checkpoints_dir, args.exp_name, model_name, f'{model_name}_best_loss.pt'))
    # print(checkpoint)
    
    # Create an instance of the model
    if args.feature_type == 'radiology':
        model = RModel(args)
    elif args.feature_type == 'pathology':
        model = PModel(args)
    else:
        model = Model(args)

    # Extract the 'epoch' from the loaded checkpoint
    saved_epoch = checkpoint['epoch']

    # Print or use the extracted epoch
    print(f"The model is saved on epoch: {saved_epoch}")

    # Load the model state from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)

    if args.feature_type == 'radiology':
        test_set = RadDataset(data, args.dataroot, index=None)
        # print(test_set[104][2].shape)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    elif args.feature_type == 'pathology':
        test_set = PathDataset(data, args.dataroot, index=None)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False,collate_fn=custom_collate_pathology)

    else:
        test_set = RadPathDataset(data, args.dataroot, index=None)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False,collate_fn=custom_collate)

    # print(test_set[104][2].shape)
    train_loss = checkpoint['train_loss']
    train_preds = checkpoint['train_pred']
    val_loss = checkpoint['val_loss']
    val_preds = checkpoint['val_pred']

    test_loss, test_preds = one_epoch(
            args, "test", model, None, test_loader, criterion)
    
    ci_train, auc_train = compute_metrics(args, train_preds)
    ci_val, auc_val = compute_metrics(args, val_preds)
    ci_test, auc_test = compute_metrics(args, test_preds)
    # print(test_preds)

    print(
        f"[Final] Apply model to training set: Loss = {train_loss}, C-Index = {ci_train}, AUC grade= {auc_train}")
    print(
        f"[Final] Apply model to validation set: Loss = {val_loss}, C-Index = {ci_val}, AUC grade= {auc_val}")
    print(
        f"[Final] Apply model to test set: Loss = {test_loss}, C-Index = {ci_test}, AUC grade= {auc_test}")

    pickle.dump(train_preds, open(os.path.join(args.checkpoints_dir, args.exp_name,model_name, f'{model_name}_pred_train.pkl'), 'wb'))
    pickle.dump(val_preds, open(os.path.join(args.checkpoints_dir, args.exp_name,model_name, f'{model_name}_pred_val.pkl'), 'wb'))
    pickle.dump(test_preds, open(os.path.join(args.checkpoints_dir, args.exp_name, model_name, f'{model_name}_pred_test.pkl'), 'wb'))
    


def train_val(args, device):
    criterion = MultiTaskLoss()
    if args.mmo_loss:
        criterion = [MultiTaskLoss(), MMOLoss()]
    data = extract_csv(os.path.join(
        args.dataroot, "data_table_output.csv"))
    labels = data[['grade', "DFS", "DFS_censor"]]

    indices = np.arange(labels.shape[0])

    cudnn.deterministic = True
    torch.cuda.manual_seed_all(2023)
    torch.manual_seed(2023)
    np.random.seed(2023)

    if args.feature_type == 'radiology':
        model = RModel(args)
    elif args.feature_type == 'pathology':
        model = PModel(args)
    else:
        model = Model(args)
    
    device = 'cuda:0'
    model.to(device)
    
    optim = define_optimizer(args, model)
    scheduler = define_scheduler(args, optim)
    # print(model)
    print("Number of Trainable Parameters: %d" %
          count_parameters(model))
    print("Optimizer Type:", args.optimizer_type)
    print("Activation Type:", args.act_type)

    _, _, _, _, train_index, test_index = train_test_split(
        data, labels, indices, test_size=0.3, random_state=2023, shuffle=False, stratify=None)
    
    model, optim, metric_logger = train_model(
        args, data, model, criterion, optim, scheduler, train_index, test_index, device)

    return metric_logger


def save_results_to_mat(split, args, model_name):
    file_path = os.path.join(args.checkpoints_dir, args.exp_name, model_name, f'{model_name}_pred_{split}.pkl')
    data = pickle.load(open(file_path, "rb"))
    
    flattened_list = [item for sublist in data[5] for item in sublist]
    IDs = np.asarray(flattened_list)

    matlab_dict = {
        f'{split}_ID': IDs,
        f'{split}_score': data[1].cpu().detach().numpy(),
        f'{split}_grade_score': data[0].cpu().detach().numpy()
    }

    mat_file_path = f"C:\\Users\\bsong47\\OneDrive - Emory University\\Documents\\MATLAB\\raptomic_{split}_data_{args.feature_type}.mat"
    scipy.io.savemat(mat_file_path, matlab_dict)


if __name__ == '__main__':
    args = parse_args()
    root = args.dataroot
    device = torch.device('cuda:{}'.format(
        args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    print("Using device:", device)
    model_name = args.fusion_type+'_'+args.task+'_'+str(args.n_epochs)+'_'+str(args.lr)+'_'+str(args.feature_type)
    print(model_name)
    metric_logger = train_val(args, device)
    test(args, device)

    
    # Save results for train, validation, and test sets
    save_results_to_mat("train", args, model_name)
    save_results_to_mat("val", args, model_name)
    save_results_to_mat("test", args, model_name)
    



    # # # # # Plotting loss and C-index
    plt.figure(figsize=(12, 12))

    # Plotting the training loss
    plt.subplot(2, 2, 1)
    plt.plot(range(args.epoch_count, args.niter + args.n_epochs + 1),
            metric_logger['train']['loss'], label='Train')
    plt.plot(range(args.epoch_count, args.niter + args.n_epochs + 1),
            metric_logger['val']['loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plotting the training C-index
    plt.subplot(2, 2, 2)
    plt.plot(range(args.epoch_count, args.niter + args.n_epochs + 1),
            metric_logger['train']['cindex'], label='Train')
    plt.plot(range(args.epoch_count, args.niter + args.n_epochs + 1),
            metric_logger['val']['cindex'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('C-index')
    plt.title('Training and Validation C-index: DFS')
    plt.legend()

    # Plotting the training C-index
    plt.subplot(2, 2, 3)
    plt.plot(range(args.epoch_count, args.niter + args.n_epochs + 1),
            metric_logger['train']['grad_auc'], label='Train')
    plt.plot(range(args.epoch_count, args.niter + args.n_epochs + 1),
            metric_logger['val']['grad_auc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC: Grade')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

    



    




    





    