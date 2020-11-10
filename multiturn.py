import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
import numpy as np

def cnn_output_size(input_size,filter_size,stride,padding):
    '''
    Calculate the size of a conv2d_layer output or pooling2d_layer output tensor based on its hyperparameters. 
    All hyperparameters should be 2-element tuple or list of length 2.
    '''
    return [int((input_size[i]-filter_size[i]+padding[i])/stride[i] +1 )for i in range(2)]

class GRUEncoder(nn.Module):
    """
    A GRU recurrent neural network encoder.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 embedder=None,
                 rnn_hidden_size=None,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0):
        super(GRUEncoder, self).__init__()

        self.num_directions = 2 if bidirectional else 1
        if not rnn_hidden_size:
            assert hidden_size % self.num_directions == 0
        else:
            assert rnn_hidden_size == hidden_size
        self.rnn_hidden_size = rnn_hidden_size or hidden_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.rnn_hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          bidirectional=self.bidirectional)

    def forward(self, inputs, hidden=None):
        if isinstance(inputs, tuple):
            inputs, lengths = inputs
        else:
            inputs, lengths = inputs, None

        if self.embedder is not None:
            rnn_inputs = self.embedder(inputs)
        else:
            rnn_inputs = inputs

        batch_size = rnn_inputs.size(0)
        max_seq_length = rnn_inputs.size(1)
       
        if lengths is not None:
            
            num_valid = lengths.gt(0).int().sum().item()  # 当batch不足batch_size
            sorted_lengths, indices = lengths.sort(descending=True)
            rnn_inputs = rnn_inputs.index_select(0, indices)
            
            rnn_inputs = pack_padded_sequence(
                rnn_inputs[:num_valid],
                sorted_lengths[:num_valid].tolist(),
                batch_first=True)

            if hidden is not None:
                hidden = hidden.index_select(1, indices)[:, :num_valid]

        self.rnn.flatten_parameters()

        
        outputs, last_hidden = self.rnn(rnn_inputs,hidden)
        
        if self.bidirectional:
            last_hidden = self._bridge_bidirectional_hidden(last_hidden)

        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
            

            if num_valid < batch_size:
                zeros = outputs.new_zeros(
                    batch_size - num_valid, outputs.size(1), self.rnn_hidden_size * self.num_directions)
                outputs = torch.cat([outputs, zeros], dim=0)

                zeros = last_hidden.new_zeros(
                    self.num_layers, batch_size - num_valid, self.rnn_hidden_size * self.num_directions)
                last_hidden = torch.cat([last_hidden, zeros], dim=1)
            
            if sorted_lengths[0] < max_seq_length: # 最大长度padding
                zero = zeros = outputs.new_zeros(
                    outputs.size(0), max_seq_length-sorted_lengths[0], self.rnn_hidden_size * self.num_directions)
                outputs = torch.cat([outputs, zeros], dim=1)

            _, inv_indices = indices.sort()
            outputs = outputs.index_select(0, inv_indices)
            last_hidden = last_hidden.index_select(1, inv_indices)

        return outputs, last_hidden

    def _bridge_bidirectional_hidden(self, hidden):
        """
        hidden is the last-hide
        the bidirectional hidden is (num_layers * num_directions, batch_size, rnn_hidden_size)
        we need to convert it to (num_layers, batch_size, num_directions * rnn_hidden_size)
        """
        num_layers = hidden.size(0) // 2
        _, batch_size, hidden_size = hidden.size()
        return hidden.view(num_layers, 2, batch_size, hidden_size) \
            .transpose(1, 2).contiguous().view(num_layers, batch_size, hidden_size * 2)

class URMatching(nn.Module):
    '''
    Utterance-Response Matching Module as a sub-module in SMN model, 
    modeling pair-wise matching information between utterances and response text.
    '''
    def __init__(self,args) -> None:
        super(URMatching,self).__init__()
        self.args=args
        self.embedding=nn.Embedding(
            num_embeddings=self.args.num_embeddings,
            embedding_dim=self.args.embedding_size,
            padding_idx=0
        )
        pretrained_weight=pickle.load(open(self.args.data_dir+'embedding.pkl','rb'),encoding="bytes")
        pretrained_weight = np.array(pretrained_weight)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.gru_utterance=GRUEncoder(
            input_size=self.args.embedding_size,
            hidden_size=self.args.hidden_size,
            rnn_hidden_size=self.args.hidden_size
            )
        ih_u = (param.data for name, param in self.gru_utterance.named_parameters() if 'weight_ih' in name)
        hh_u = (param.data for name, param in self.gru_utterance.named_parameters() if 'weight_hh' in name)
        for k in ih_u:
            nn.init.orthogonal_(k)
        for k in hh_u:
            nn.init.orthogonal_(k)

        self.gru_response=GRUEncoder(
            input_size=self.args.embedding_size,
            hidden_size=self.args.hidden_size,
            rnn_hidden_size=self.args.hidden_size
            )
        ih_r = (param.data for name, param in self.gru_response.named_parameters() if 'weight_ih' in name)
        hh_r = (param.data for name, param in self.gru_response.named_parameters() if 'weight_hh' in name)
        for k in ih_r:
            nn.init.orthogonal_(k)
        for k in hh_r:
            nn.init.orthogonal_(k)

        self.gru_hidden_size=self.args.hidden_size*2

        self.A=nn.Parameter(
            torch.randn(size=(self.gru_hidden_size,self.gru_hidden_size),requires_grad=True)
            )
        nn.init.xavier_uniform_(self.A)
        
        self.conv=nn.Conv2d(
            in_channels=2,
            out_channels=self.args.out_channels,
            kernel_size=self.args.kernel_size
            )
        conv2d_weight = (param.data for name, param in self.conv.named_parameters() if "weight" in name)
        for w in conv2d_weight:
            nn.init.kaiming_normal_(w)

        self.act=nn.ReLU()
        self.pooling=nn.MaxPool2d(kernel_size=self.args.kernel_size,stride=self.args.stride)
        self.output_size=cnn_output_size(
            input_size=cnn_output_size(
                input_size=(self.args.input_size,self.args.input_size),
                filter_size=self.args.kernel_size,stride=(1,1),padding=(0,0)),
                filter_size=self.args.kernel_size,stride=self.args.stride,padding=(0,0)
                )
        
        self.linear=nn.Linear(
            self.args.out_channels*self.output_size[0]*self.output_size[1],
            self.args.inter_size
            )
        linear_weight = (param.data for name, param in self.linear.named_parameters() if "weight" in name)
        for w in linear_weight:
            nn.init.xavier_uniform_(w)

    def forward(self,utterance,len_utterance,response,len_response):
        utterance_embed=self.embedding(utterance) 
        
        all_utterance_embeddings = utterance_embed.permute(1, 0, 2, 3)
       
        all_utterance_length=len_utterance.permute(1,0)
       

        response_embed=self.embedding(response)
        response_embed_gru,_=self.gru_response((response_embed,len_response))
        response_embeddings = response_embed.permute(0, 2, 1)
        response_embed_gru = response_embed_gru.permute(0, 2, 1)

        matching_vectors ,h_u= [],[]
        for utterance_embeddings,utterance_length in zip(all_utterance_embeddings,all_utterance_length):
            m1 = torch.matmul(utterance_embeddings, response_embeddings)
            
            # utterance_embed_gru, last_hidden_u = self.gru_utterance((utterance_embeddings,utterance_length))
            utterance_embed_gru, last_hidden_u = self.gru_utterance(utterance_embeddings)
            
            m2 = torch.einsum('aij,jk->aik', utterance_embed_gru, self.A)
            m2 = torch.matmul(m2, response_embed_gru)
            
            m=torch.stack([m1,m2],dim=1)
            
            conv_layer = self.conv(m)
            conv_layer = F.relu(conv_layer)
            pooling_layer = self.pooling(conv_layer)
            pooling_layer = pooling_layer.view(pooling_layer.size(0), -1)

            matching_vector = self.linear(pooling_layer)
            matching_vector = torch.tanh(matching_vector)
            matching_vectors.append(matching_vector)

            h_u.append(last_hidden_u.squeeze(0))

        match=torch.stack(matching_vectors, dim=1)
        #for dynamic fusion of h'
        h_u=torch.stack(h_u,dim=1)
        return match,h_u


class MatchingAccumulation(nn.Module):
    '''
    Matching Accumulation Module as a sub-module in SMN model, modeling sequential characteristic of utterances fused with matching information from Utterance-Response Matching Module.
    '''
    def __init__(self,args) -> None:
        super(MatchingAccumulation,self).__init__()
        self.args=args
        self.gru=GRUEncoder(
            input_size=self.args.inter_size,
            hidden_size=self.args.hidden_size_ma,
            rnn_hidden_size=self.args.hidden_size_ma
            )
        ih_u = (param.data for name, param in self.gru.named_parameters() if 'weight_ih' in name)
        hh_u = (param.data for name, param in self.gru.named_parameters() if 'weight_hh' in name)
        for k in ih_u:
            nn.init.orthogonal_(k)
        for k in hh_u:
            nn.init.orthogonal_(k)
        
        #no extra parameter for SMN-last
        if self.args.fusion_type=='last':
            pass

        #parameters for SMN-static (w)
        if self.args.fusion_type=='static':
            self.weight_static=nn.Parameter(torch.randn(size=(1,self.args.max_utter_num),requires_grad=True))
            nn.init.xavier_uniform_(self.weight_static)

        #parameters for SMN-dynamic (W1,W2,ts)
        if self.args.fusion_type=='dynamic':
            self.linear1=nn.Linear(self.args.hidden_size*2,self.args.q,bias=True)
            linear1_weight = (param.data for name, param in self.linear1.named_parameters() if "weight" in name)
            for w in linear1_weight:
                nn.init.xavier_uniform_(w)

            self.linear2=nn.Linear(self.args.hidden_size_ma*2,self.args.q,bias=False)
            linear2_weight = (param.data for name, param in self.linear2.named_parameters() if "weight" in name)
            for w in linear2_weight:
                nn.init.xavier_uniform_(w)

            self.ts=nn.Parameter(torch.randn(size=(self.args.q,1)),requires_grad=True)

        self.lin_output=nn.Linear(self.args.hidden_size_ma*2,2,bias=True)
        final_linear_weight = (param.data for name, param in self.lin_output.named_parameters() if "weight" in name)
        for w in final_linear_weight:
            nn.init.xavier_uniform_(w)
        
    def forward(self,num_utterance,match,h_u):
        
        output,last_hidden=self.gru((match,num_utterance))
        #last:the last hidden state 
        if self.args.fusion_type=='last':
            L=last_hidden.squeeze(0)
            
        #static:weighted sum of all hidden states
        elif self.args.fusion_type=='static':
            L=torch.matmul(self.weight_static,output).squeeze(1)

        #dynamic:attention-weighted hidden states
        elif self.args.fusion_type=='dynamic':
            t=torch.tanh(self.linear1(h_u)+self.linear2(output))
            alpha=F.softmax(torch.matmul(t,self.ts),dim=1).squeeze(-1)
            L=torch.matmul(alpha.unsqueeze(1),output).squeeze(1)

        #deal with unbound variable
        else:
            L=0
        g=self.lin_output(L)

        return g

class SMN(nn.Module):
    '''
    Refer to paper for details.
    Title:Sequential Matching Network:A New Architecture for Multi-turn
    Response Selection in Retrieval-Based Chatbots 
    Authors:Yu Wu†, Wei Wu‡, Chen Xing♦, Zhoujun Li†∗, Ming Zhou
    Tensorflow version : https://github.com/MarkWuNLP/MultiTurnResponseSelection
    '''
    def __init__(self,args) -> None:
        super().__init__()
        self.urmatch=URMatching(args)
        self.matchacc=MatchingAccumulation(args)

    def forward(self,
                utterance,len_utterance,
                num_utterance,
                response,len_response):
        match,h_u=self.urmatch(utterance,len_utterance,response,len_response)
        score=self.matchacc(num_utterance,match,h_u)
        return score


       