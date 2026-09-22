import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import create_var, index_select_ND,mol2graph,atom_if_sca
from scipy.stats import multivariate_normal  # 生成多维概率分布的方法
import numpy as np

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn',
             'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / query.size(-1) ** 0.5
        attention_weights = self.softmax(scores)
        output = torch.matmul(attention_weights, value)
        return output
#Node-central Encode
class NMPN(nn.Module):

    def __init__(self, hidden_size, depth):
        super(NMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.W_nin = nn.Linear(ATOM_FDIM , hidden_size, bias=False)
        self.W_node = nn.Linear(hidden_size+BOND_FDIM, hidden_size, bias=False)
        #self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        #self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        #self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, mol_graph):
        fatoms, fbonds, aoutgraph, bgraph, aingraph, scope, all_bonds= mol_graph
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        aoutgraph = create_var(aoutgraph)
        #bgraph = create_var(bgraph)

        h_0 = self.W_nin(fatoms)
        h_0 = nn.ReLU()(h_0)
        h_0 = h_0.t()
        H_n = h_0
        #message = nn.ReLU()(binput)

        for i in range(self.depth):
            #Message function
            message = self.messagefunction(H_n,fbonds,all_bonds)
            nei_message = index_select_ND(message, 0, aoutgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_node(nei_message).t()
            #nei_message = nei_message.t()
            #update function
            H_n = nn.ReLU()(h_0 + nei_message)
        return H_n

    def messagefunction(self,H_n,fbonds,all_bonds):
        total_bonds = len(fbonds)
        in_n = []
        for b1 in range(1, total_bonds):
            x, y = all_bonds[b1]
            in_n.append(y)
        in_n = create_var(torch.tensor(in_n))
        message = H_n.index_select(1,in_n).t()
        #message = message.t()
        zero = create_var(torch.unsqueeze(torch.zeros(message.size()[1:]),0))
        message = torch.cat([zero,message],0)
        message = torch.cat([message , fbonds], 1)
        return message

#Edge-central Encoder
class EMPN(nn.Module):

    def __init__(self, hidden_size, depth, out):
        super(EMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.out = out
        self.W_ein = nn.Linear(BOND_FDIM , hidden_size, bias=False)
        self.W_edge = nn.Linear(hidden_size+ATOM_FDIM, hidden_size, bias=False)
        self.W_eout = nn.Linear(hidden_size + ATOM_FDIM, out, bias=False)
        #self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        #self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        #self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, mol_graph):
        fatoms, fbonds, aoutgraph, bgraph, aingraph, scope, all_bonds = mol_graph
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        #aoutgraph = create_var(aoutgraph)
        bgraph = create_var(bgraph)
        aingraph = create_var(aingraph)

        h_0 = self.W_ein(fbonds)
        h_0 = nn.ReLU()(h_0)
        H_e = h_0
        #message = nn.ReLU()(binput)

        for i in range(self.depth):
            # Message function
            message = self.messagefunction(H_e, fatoms, all_bonds)
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_edge(nei_message)
            H_e = nn.ReLU()(h_0 + nei_message)

        message = self.messagefunction(H_e, fatoms, all_bonds)
        nei_message = index_select_ND(message, 0, aingraph)
        nei_message = nei_message.sum(dim=1)
        nei_message = self.W_eout(nei_message)
        H_e = nn.ReLU()(nei_message).t()
        #H_e = H_e.t()
        return H_e

    def messagefunction(self, H_e, fatoms, all_bonds):
        total_bonds = len(all_bonds)
        out_n = []
        for b1 in range(1, total_bonds):
            x, y = all_bonds[b1]
            out_n.append(x)
        out_n = create_var(torch.tensor(out_n))
        message = fatoms.index_select(0, out_n)
        zero = create_var(torch.unsqueeze(torch.zeros(message.size()[1:]), 0))
        message = torch.cat([zero, message], 0)
        message = torch.cat([H_e,message], 1)
        return message

#gru
class MultiGRU(nn.Module):
    """ Implements a three layer GRU cell including an embedding layer
       and an output linear layer back to the size of the vocabulary"""
    def __init__(self, voc_size, h_size):
        super(MultiGRU, self).__init__()
        self.embedding = nn.Embedding(voc_size, 128)
        self.gru_1 = nn.GRUCell(128, h_size)
        self.gru_2 = nn.GRUCell(h_size, h_size)
        self.gru_3 = nn.GRUCell(h_size, h_size)
        self.linear = nn.Linear(h_size, voc_size)

    def forward(self, x, h):
        x = self.embedding(x)
        h_out = create_var(torch.zeros(h.size()))
        x = h_out[0] = self.gru_1(x, h[0])
        x = h_out[1] = self.gru_2(x, h[1])
        x = h_out[2] = self.gru_3(x, h[2])
        x = self.linear(x)
        return x, h_out

# double mpn and gru
class DMPN(nn.Module):
    def __init__(self, hidden_size, depth, out, atten_size, r, d_hid , d_z, voc, ver = False):
        super(DMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.out = out
        self.atten_size = atten_size
        self.r = r
        self.d_hid = d_hid
        self.d_z = d_z
        self.voc =voc
        self.W_1 = nn.Linear(self.hidden_size + self.out, self.atten_size, bias=False)
        self.W_2 = nn.Linear(self.atten_size, self.r, bias=False)
        self.W_3 = nn.Linear(self.r * (self.hidden_size + self.out), self.d_hid)

        self.NMPN = NMPN(self.hidden_size,self.depth)
        self.EMPN = EMPN(self.hidden_size,self.depth,self.out)
        self.attention = SelfAttention(self.d_hid*2)
        
        self.W_4 = nn.Linear(self.d_hid*2+self.d_hid, self.d_hid*2)
        if ver == False:
            self.rnn = MultiGRU(voc.vocab_size, (self.d_hid * 2 ))
            self.q_mu = nn.Linear(self.d_hid, self.d_z)
            self.q_logvar = nn.Linear(self.d_hid, self.d_z)
            self.decoder_lat = nn.Linear(self.d_z, self.d_hid)
        else :
            self.rnn = MultiGRU(voc.vocab_size, (self.d_hid * 2))
            self.q_mu = nn.Linear(self.d_hid*2, self.d_z)
            self.q_logvar = nn.Linear(self.d_hid*2, self.d_z)
            self.decoder_lat = nn.Linear(self.d_z, self.d_hid*2)

 
    def forward(self, mol_batch, sca_batch, target):
        # 获取编码器的输出
        space_side, space_sca = self.forward_encoder(mol_batch, sca_batch)
        # 计算潜在变量 z 和 KL 损失
        mu, logvar = self.q_mu(space_sca), self.q_logvar(space_sca)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        # 解码器
        sca_h = self.decoder_lat(z)
        gru_h0 = torch.cat([sca_h, space_side], 1)
        # gru_h0 = torch.unsqueeze(gru_h0, 0).repeat([3, 1, 1]).type(torch.float32)
        # 应用自注意力机制
        attention_output = self.attention(gru_h0)
        # 可以对自注意力输出进行处理（如进一步融合特征）
        combined_features = torch.cat([attention_output, sca_h],1)
        combined_features = self.W_4(combined_features)
        combined_features=torch.unsqueeze(combined_features, 0).repeat([3, 1, 1]).type(torch.float32)
        # 计算重构损失
        recon_loss = self.forward_decoder(combined_features, target)
        return kl_loss, recon_loss
# add Experimental verification
    def forward_ver(self, mol_batch, sca_batch, target):
        space_side, space_sca = self.forward_encoder( mol_batch,sca_batch)
        mol_embeding = torch.cat([space_side,space_sca],1)
        mu, logvar = self.q_mu(mol_embeding), self.q_logvar(mol_embeding)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()

        mol_h = self.decoder_lat(z)
        gru_h0 = mol_h
        gru_h0 = torch.unsqueeze(gru_h0,0).repeat([3,1,1]).type(torch.float32)
        recon_loss, pro_log = self.forward_decoder(gru_h0,target)
        return kl_loss ,recon_loss,pro_log

    def sample_ver(self,n_batch, max_length=140):
        with torch.no_grad():
            # space_side, space_sca = self.forward_encoder(mol, sca)
            z = self.sample_z_prior(n_batch)
            mol_h = self.decoder_lat(z)
            # space_side = space_side.repeat(n_batch,1)
            gru_h0 = mol_h

            gru_h0 = torch.unsqueeze(gru_h0, 0).repeat([3, 1, 1]).type(torch.float32)
            h = gru_h0

            start_token = create_var(torch.zeros(n_batch).long())
            start_token[:] = self.voc.vocab['GO']
            x = start_token
            sequences = []
            log_probs = create_var(torch.zeros(n_batch))
            finished = torch.zeros(n_batch).byte()
            if torch.cuda.is_available():
                finished = finished.cuda()

            for step in range(max_length):
                logits, h = self.rnn(x, h)
                prob = F.softmax(logits, dim=1)
                log_prob = F.log_softmax(logits, dim=1)
                x = torch.multinomial(prob, num_samples=1).view(-1)
                log_probs += self.NLLLoss(log_prob, x)
                sequences.append(x.view(-1, 1))

                x = create_var(x.data)
                EOS_sampled = (x == self.voc.vocab['EOS']).data
                finished = torch.ge(finished + EOS_sampled, 1)
                if torch.prod(finished) == 1: break
            sequences = torch.cat(sequences, 1)
            return sequences.data,-log_probs

    def read_out(self, h_node, s_sca):
        sca_index = []
        side_index = []
        for i in range(len(s_sca)):
            if s_sca[i] == 1:
                sca_index.append(i)
            else:
                side_index.append(i)

        sca_index = create_var(torch.tensor(sca_index))
        side_index = create_var(torch.tensor(side_index))
        sca_node = h_node.index_select(0,sca_index)
        side_node = h_node.index_select(0,side_index)
        sca_s = F.softmax(self.W_2(nn.Tanh()(self.W_1(sca_node))),1)
        sca_s = sca_s.t()
        side_s = F.softmax(self.W_2(nn.Tanh()(self.W_1(side_node))),1)
        side_s =side_s.t()
        sca_embeding = self.W_3(torch.flatten(torch.mm(sca_s,sca_node)))
        side_embeding = self.W_3(torch.flatten(torch.mm(side_s,side_node)))

        return sca_embeding,side_embeding

    def forward_encoder(self, mol_batch, sca_batch):
        mol_graph = mol2graph(mol_batch)
        S_sca,scope = atom_if_sca(mol_batch, sca_batch)
        H_n = self.NMPN(mol_graph)
        H_e = self.EMPN(mol_graph)
        H_node = torch.cat([H_n, H_e], 0).t()
        # H_node = H_node.t()

        hidden_space_sca = []
        hidden_space_side = []
        for st, le in mol_graph[5]:
            # readout function
            s_sca = S_sca[st: st + le]
            cur_vecs_sca, cur_vecs_side = self.read_out(H_node[st: st + le], s_sca)
            hidden_space_sca.append(cur_vecs_sca)
            hidden_space_side.append(cur_vecs_side)

        space_sca = torch.stack(hidden_space_sca, 0)
        space_side = torch.stack(hidden_space_side, 0)
        return space_side, space_sca

    def forward_decoder(self,h_origin,target):
        batch_size, seq_length = target.size()
        start_token = create_var(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']
        x = torch.cat((start_token, target[:, :-1]), 1)
        h = h_origin

        log_probs = create_var(torch.zeros(batch_size))
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], h)
            log_prob = F.log_softmax(logits, dim=1)
            log_probs += self.NLLLoss(log_prob, target[:, step])

        return log_probs.mean()

    def NLLLoss(self,inputs, targets):
        if torch.cuda.is_available():
            target_expanded = torch.zeros(inputs.size()).cuda()
        else:
            target_expanded = torch.zeros(inputs.size())
        target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
        loss = create_var(target_expanded) * inputs
        loss = -torch.sum(loss, 1)
        return loss

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)
        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return create_var(torch.randn(n_batch, self.d_z))

    def sample(self,n_batch,mol,sca, max_length=140):
        with torch.no_grad():
            space_side, space_sca = self.forward_encoder(mol, sca)
            z = self.sample_z_prior(n_batch)
            sca_h = self.decoder_lat(z)
            space_side = space_side.repeat(n_batch,1)
            gru_h0 = torch.cat([sca_h,space_side], 1)

            attention_output = self.attention(gru_h0)
        # 可以对自注意力输出进行处理（如进一步融合特征）
            combined_features = torch.cat([attention_output, sca_h],1)
            combined_features = self.W_4(combined_features)
            combined_features=torch.unsqueeze(combined_features, 0).repeat([3, 1, 1]).type(torch.float32)
            # gru_h0 = torch.unsqueeze(gru_h0, 0).repeat([3, 1, 1]).type(torch.float32)
            # h = gru_h0
            h=combined_features

            start_token = create_var(torch.zeros(n_batch).long())
            start_token[:] = self.voc.vocab['GO']
            x = start_token
            sequences = []
            log_probs = create_var(torch.zeros(n_batch))
            finished = torch.zeros(n_batch).byte()
            if torch.cuda.is_available():
                finished = finished.cuda()

            for step in range(max_length):
                logits, h = self.rnn(x, h)
                prob = F.softmax(logits, dim=1)
                log_prob = F.log_softmax(logits, dim=1)
                x = torch.multinomial(prob, num_samples=1).view(-1)
                log_probs += self.NLLLoss(log_prob, x)
                sequences.append(x.view(-1, 1))

                x = create_var(x.data)
                EOS_sampled = (x == self.voc.vocab['EOS']).data
                finished = torch.ge(finished + EOS_sampled, 1)
                if torch.prod(finished) == 1: break

            sequences = torch.cat(sequences, 1)
            return sequences.data


if __name__ == "__main__":
    import os
    new_directory = "/ScaffoldInvent/"
    os.chdir(new_directory)
    print("")

