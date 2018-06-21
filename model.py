import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)

        print('__init__(embed_size, hidden_size, vocab_size, num_layers):', embed_size, hidden_size, vocab_size, num_layers)
        
    def init_hidden(self, n_batches):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.hidden = (torch.zeros(self.num_layers, 
                                   n_batches, 
                                   self.hidden_size).to(device),
                        torch.zeros(self.num_layers, 
                                    n_batches, 
                                    self.hidden_size).to(device))
        return self.hidden
    
    def forward(self, features, captions):
        # features shape: (n_batches, embed_size)
        # captions shape: (n_batches, n_words_in_caption)
        n_batches = features.shape[0]
        
        # embedding output shape: (n_batches, n_words_in_caption, embed_size)
        embed = self.embeddings(captions)
        
        # change input features shape to (n_batches, 1, embed_size)
        features_3d = features.view(features.shape[0], 1, -1)
        
        # stack embedded image features and captions together 
        stacked = torch.cat([features_3d, embed], dim=1)

        # for lstm input, drop the last '<end>' for each caption
        lstm_in = stacked[:, :-1, :]
        
        self.init_hidden(n_batches)        
        lstm_out, self.hidden = self.lstm(lstm_in, self.hidden)
        
        # wk_debug
        print('lstm_out:', lstm_out.size())
        
        tag_space = self.hidden2tag(lstm_out)
        
        # wk_debug
        print('tag_space:', tag_space.size())
        #tag_scores = F.softmax(tag_space, dim=2)
        return tag_space

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        
        # wk_debug
        #print('inputs:', inputs.size())
        
        self.hidden = states
        if self.hidden is None:
            self.hidden = self.init_hidden(inputs.shape[0])
            
        lstm_in = inputs
        for i in range(max_len):
            lstm_out, self.hidden = self.lstm(lstm_in, self.hidden)
            tag_space = self.hidden2tag(lstm_out)
            tag_scores = F.softmax(tag_space, dim=2)
            
            # wk_debug
            #print('lstm_out.size():', lstm_out.size())
            #print('tag_scores:', tag_scores.size())
            
            _, word_idx = torch.max(tag_scores, dim=2)
            lstm_in = self.embeddings(word_idx)
            sentence.append(word_idx.squeeze().item())
            
            # wk_debug
            #print('word_idx:', word_idx)
            #print('lstm_in.size():', lstm_in.size())            
            
        return sentence