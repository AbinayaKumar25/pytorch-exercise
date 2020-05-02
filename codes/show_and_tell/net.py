import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from drnn import DRNN

class Net(nn.Module):
    def __init__(self, TEXT, model_type,
                 hidden_dim=512, num_layers=2):
        super().__init__()

        vocab_size = TEXT.vocab.vectors.size(0)
        embed_dim = TEXT.vocab.vectors.size(1)
        
        self.encoder = Encoder(embed_dim, model_type)
        #self.decoder = Decoder(TEXT,
        #                       vocab_size, embed_dim,
        #                       hidden_dim, num_layers)

        self.decoder = Decoder_DRNN(TEXT,
                               vocab_size, embed_dim,
                               hidden_dim, num_layers)

    def forward(self, image, caption, lengths):
        feature = self.encoder(image)
        out = self.decoder(feature, caption, lengths)

        return out

    def sample(self, image):
        feature = self.encoder(image)
        out = self.decoder.sample(feature)
        
        return out


class Encoder(nn.Module):
    def __init__(self, embed_dim, model_type):
        super().__init__()

        if model_type == "resnet-50":
            self.body = models.resnet50(pretrained=True)
        else:
            self.body = models.resnet18(pretrained=True)

        for param in self.body.parameters():
            param.requires_grad_(False)

        # modify last fc layer
        self.body.fc = nn.Linear(self.body.fc.in_features, embed_dim)

    def forward(self, x):
        return self.body(x)


class Decoder(nn.Module):
    def __init__(self, TEXT,
                 vocab_size, embed_dim,
                 hidden_dim, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, 
                           num_layers=num_layers,
                           batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        self.embedding.weight.requires_grad_(False)

    def forward(self, feature, caption, lengths):
        embed = self.embedding(caption)
        embed = torch.cat((feature.unsqueeze(1), embed), 1)
        
        embed = pack_padded_sequence(embed, lengths, batch_first=True)        
        out, _ = self.rnn(embed)
        out, _ = pad_packed_sequence(out, batch_first=True)
   
        out = self.linear(out)
        out = out.view(-1, out.size(2))

        return out

    def sample(self, feature):
        batch_size = feature.size(0)
      
        hidden = None
        embed = feature.unsqueeze(1)
                
        indices = list()
        for t in range(50):
            out, hidden = self.rnn(embed, hidden)
            out = self.linear(out.squeeze(1))

            _, argmax = torch.max(out, 1)
            indices.append(argmax)
                        
            # previous output is current input
            embed = self.embedding(argmax).unsqueeze(1)
                                          
        return torch.stack(indices, 1).cpu().numpy()

class Decoder_DRNN(nn.Module):
    def __init__(self, TEXT,
                 vocab_size, embed_dim,
                 hidden_dim, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = DRNN(embed_dim, hidden_dim, 
                           n_layers=num_layers,
                           cell_type='LSTM',
                           batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        self.embedding.weight.requires_grad_(False)

    def forward(self, feature, caption, lengths):
        embed = self.embedding(caption)
        embed = torch.cat((feature.unsqueeze(1), embed), 1)
                
        out, _ = self.rnn(embed, lengths)
   
        out = self.linear(out)
        out = out.view(-1, out.size(2))

        return out

    def sample(self, feature):
        batch_size = feature.size(0)
      
        hidden = None
        
        #embed = feature.unsqueeze(1)
        embed = feature
        result = torch.zeros(batch_size, 51, embed.size(1)).cuda()
        result[:,0,:] = embed
        #result = torch.cat([embed, result])
                
        indices = list()
        for t in range(50):
            #out, hidden = self.rnn(embed,hidden=hidden)
            out, hidden = self.rnn(result)
            out = self.linear(out.squeeze(1))

            #take current timestep
            out = out[:, t, :]

            _, argmax = torch.max(out, 1)
            indices.append(argmax)
            
            
            # previous output is current input
            #embed = self.embedding(argmax).unsqueeze(1)

            embed = self.embedding(argmax)
            result[:, (t+1), :] = embed
                                          
        return torch.stack(indices, 1).cpu().numpy()