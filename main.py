import numpy as np
import torch
import pickle
from model import LightGCL
from utils import *
import pandas as pd
from parser import args
from tqdm import tqdm
import time
import torch.utils.data as data
from utils import TrnData

device = 'cuda:' + args.cuda

# hyperparameters
d = args.d
l = args.gnn_layer
temp = args.temp
batch_user = args.batch
epoch_no = args.epoch
max_samp = 40
lambda_1 = args.lambda1
lambda_2 = args.lambda2
dropout = args.dropout
lr = args.lr
decay = args.decay
svd_q = args.q

log_save_id = create_log_id(args.log_path)
logging_config(folder=args.log_path, name='log{:d}'.format(log_save_id), no_console=False)
logging.info(args)
# load data

path = args.data_dir + '/' + args.data_name + '/'
f = open(path+'trnMat.pkl','rb')
train = pickle.load(f)
train_csr = (train!=0).astype(np.float32)
f = open(path+'tstMat.pkl','rb')
test = pickle.load(f)
print('Data loaded.')

print('user_num:',train.shape[0],'item_num:',train.shape[1],'lambda_1:',lambda_1,'lambda_2:',lambda_2,'temp:',temp,'q:',svd_q)

epoch_user = min(train.shape[0], 30000)

# normalizing the adj matrix
rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)
    
logging.info('user_num: {}, item_num: {}, lambda_1: {}, lambda_2: {}, temp: {}, q: {}'.format(
    train.shape[0], train.shape[1], lambda_1, lambda_2, temp, svd_q))

# construct data loader
train = train.tocoo()
train_data = TrnData(train)
train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().cuda(torch.device(device))
print('Adj matrix normalized.')

# perform svd reconstruction
adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda(torch.device(device))
print('Performing SVD...')
svd_u,s,svd_v = torch.svd_lowrank(adj, q=svd_q)
u_mul_s = svd_u @ (torch.diag(s))
v_mul_s = svd_v @ (torch.diag(s))
del s
print('SVD done.')

# process val set and test set
val_labels = process_labels(path + 'valMat.pkl')
test_labels = process_labels(path + 'tstMat.pkl')

loss_list = []
loss_r_list = []
loss_s_list = []
recall_20_x = []
recall_20_y = []
ndcg_20_y = []
recall_40_y = []
ndcg_40_y = []

model = LightGCL(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device)
model.cuda(torch.device(device))
optimizer = torch.optim.Adam(model.parameters(),weight_decay=0,lr=lr)

current_lr = lr
loss_list = []
loss_r_list = []
loss_s_list = []
recall_1_x = []
recall_1_y = []
recall_5_y = []
ndcg_5_y = []
recall_10_y = []
ndcg_10_y = []
for epoch in range(1, epoch_no + 1):
    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    train_loader.dataset.neg_sampling()
    for i, batch in enumerate(tqdm(train_loader)):
        uids, pos, neg = batch
        uids = uids.long().to(torch.device(device))
        pos = pos.long().to(torch.device(device))
        neg = neg.long().to(torch.device(device))
        iids = torch.concat([pos, neg], dim=0)

        # feed
        optimizer.zero_grad()
        loss, loss_r, loss_s= model(uids, iids, pos, neg)

        # print(uids, iids, pos, neg)
        loss.backward()


        optimizer.step()
        # print('batch',batch)

        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()

        if torch.cuda.is_available():
          torch.cuda.empty_cache()
        # print(i, len(train_loader), end='\r')

    batch_no = len(train_loader)
    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    epoch_loss_s = epoch_loss_s/batch_no
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)
    logging.info('Epoch: {}, Loss: {}, Loss_r: {}, Loss_s: {}'.format(epoch, epoch_loss, epoch_loss_r, epoch_loss_s))


    if (epoch % 1 == 0):  # test every 5 epochs
        test_uids = np.array([i for i in range(adj_norm.shape[0])])
        batch_no = int(np.ceil(len(test_uids)/batch_user))

        all_recall_1 = 0
        all_recall_5 = 0
        all_ndcg_5 = 0
        all_recall_10 = 0
        all_ndcg_10 = 0
        for batch in tqdm(range(batch_no)):
            start = batch*batch_user
            end = min((batch+1)*batch_user,len(test_uids))

            test_uids_input = torch.LongTensor(test_uids[start:end]).to(torch.device(device))
            predictions = model(test_uids_input,None,None,None,test=True, test_items=val_labels)
            predictions = np.array(predictions.cpu())

            #top@1
            recall_1, ndcg_1 = metrics(test_uids[start:end],predictions,1,val_labels)
            #top@5
            recall_5, ndcg_5 = metrics(test_uids[start:end],predictions,5,val_labels)
            #top@10
            recall_10, ndcg_10 = metrics(test_uids[start:end],predictions,10,val_labels)

            all_recall_1+=recall_1
            all_recall_5+=recall_5
            all_ndcg_5+=ndcg_5
            all_recall_10+=recall_10
            all_ndcg_10+=ndcg_10

        print('-------------------------------------------')
        
        logging.info('Test of epoch {}: Recall@1: {}, Recall@5: {}, Ndcg@5: {}, Recall@10: {}, Ndcg@10: {}'.format(epoch, all_recall_1 / batch_no, all_recall_5 / batch_no, all_ndcg_5 / batch_no, all_recall_10 / batch_no, all_ndcg_10 / batch_no))
        recall_1_x.append(epoch)
        recall_1_y.append(all_recall_1/batch_no)
        recall_5_y.append(all_recall_5/batch_no)
        ndcg_5_y.append(all_ndcg_5/batch_no)
        recall_10_y.append(all_recall_10/batch_no)
        ndcg_10_y.append(all_ndcg_10/batch_no)

        val_score = all_recall_10/batch_no
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), args.saved_model_path + 'best_model.pth')
            logging.info('Save model on epoch {:04d}!'.format(epoch))

        best_recall, should_stop = early_stopping(recall_10_y, args.stopping_steps)

        if should_stop:
            break

# final test
model.load_state_dict(torch.load(args.saved_model_path))

test_uids = np.array([i for i in range(adj_norm.shape[0])])
batch_no = int(np.ceil(len(test_uids)/batch_user))

all_recall_1 = 0
all_recall_5 = 0
all_ndcg_5 = 0
all_recall_10 = 0
all_ndcg_10 = 0
for batch in range(batch_no):
    start = batch*batch_user
    end = min((batch+1)*batch_user,len(test_uids))

    test_uids_input = torch.LongTensor(test_uids[start:end]).to(torch.device(device))
    predictions = model(test_uids_input,None,None,None,test=True, test_items=test_labels)
    predictions = np.array(predictions.cpu())

    #top@1
    recall_1, ndcg_1 = metrics(test_uids[start:end],predictions,1,test_labels)
    #top@5
    recall_5, ndcg_5 = metrics(test_uids[start:end],predictions,5,test_labels)
    #top@10
    recall_10, ndcg_10 = metrics(test_uids[start:end],predictions,10,test_labels)

    all_recall_1+=recall_1
    all_recall_5+=recall_5
    all_ndcg_5+=ndcg_5
    all_recall_10+=recall_10
    all_ndcg_10+=ndcg_10
print('-------------------------------------------')
logging.info('Final test: Recall@1: {}, Recall@5: {}, Ndcg@5: {}, Recall@10: {}, Ndcg@10: {}'.format(
    all_recall_1 / batch_no,
    all_recall_5 / batch_no,
    all_ndcg_5 / batch_no,
    all_recall_10 / batch_no,
    all_ndcg_10 / batch_no
))

recall_1_x.append('Final')
recall_1_y.append(all_recall_1/batch_no)
recall_5_y.append(all_recall_5/batch_no)
ndcg_5_y.append(all_ndcg_5/batch_no)
recall_10_y.append(all_recall_10/batch_no)
ndcg_10_y.append(all_ndcg_10/batch_no)

metric = pd.DataFrame({
    'epoch':recall_1_x,
    'recall@1':recall_1_y,
    'recall@5':recall_5_y,
    'ndcg@5':ndcg_5_y,
    'recall@10':recall_10_y,
    'ndcg@10':ndcg_10_y
})
current_t = time.gmtime()
metric.to_csv(args.log_path + '/result_'+args.data+'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.csv')

torch.save(model.state_dict(), args.saved_model_path + 'saved_model_'+args.data+'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.pth')
torch.save(optimizer.state_dict(), args.saved_model_path +'saved_optim_'+args.data+'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.pth')