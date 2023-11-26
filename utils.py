import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as tt

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] # Seguimiento del learning rate
    
def r2(y_pred, y_true):
    # Calculate the mean of the true values
    mean = torch.mean(y_true)

    # Calculate the total sum of squares
    ss_total = torch.sum((y_true - mean) ** 2)

    # Calculate the residual sum of squares
    ss_residual = torch.sum((y_true - y_pred) ** 2)

    # Calculate R2 score
    r2 = 1 - (ss_residual / ss_total)
    
    return r2

def multiclass_accuracy(predicted, target):
    _, preds = torch.max(predicted, dim=1)
    return torch.tensor(torch.sum(preds == target).item() / len(target))

def multiclass_precision(predicted, target):
    _, preds = torch.max(predicted, dim=1)
    correct = (preds == target).float()
    true_positive = torch.sum(correct).item()
    false_positive = torch.sum(preds != target).item()
    precision = true_positive / (true_positive + false_positive + 1e-7)
    return torch.tensor(precision)

def multiclass_recall(predicted, target):
    _, preds = torch.max(predicted, dim=1)
    correct = (preds == target).float()
    true_positive = torch.sum(correct).item()
    false_negative = torch.sum(preds != target).item()
    recall = true_positive / (true_positive + false_negative + 1e-7)
    return torch.tensor(recall)

def compute_all(predictions, targets):
    accuracy = multiclass_accuracy(predictions, targets)
    precision = multiclass_precision(predictions, targets)
    recall = multiclass_recall(predictions, targets)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    return accuracy, precision, recall, f1_score

class PostProcessing(nn.Module):
    def __init__(self, task: str, model):
        super(PostProcessing, self).__init__()
        self.task_type = task
        self.model = model
    def forward(self, x):
        out = self.model(x)
        return out
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        
        if self.task_type == 'regression':
            batch_metric = [x['r2'] for x in outputs]
            epoch_r2 = torch.stack(batch_metric).mean()
            return {'val_loss': epoch_loss.item(), 'r2': epoch_r2.item()}
        else:
            batch_accuracy = [x['accuracy'] for x in outputs]
            epoch_accuracy = torch.stack(batch_accuracy).mean()
            batch_precision = [x['precision'] for x in outputs]
            epoch_precision = torch.stack(batch_precision).mean()
            batch_recall = [x['recall'] for x in outputs]
            epoch_recall = torch.stack(batch_recall).mean()
            batch_f1 = [x['f1'] for x in outputs]
            epoch_f1 = torch.stack(batch_f1).mean()
            return {'val_loss': epoch_loss.item(), 'accuracy': epoch_accuracy.item(), 'precision': epoch_precision.item(), 'recall': epoch_recall.item(), 'f1': epoch_f1.item()}

    def epoch_end(self, epoch, result):
        if self.task_type == 'regression':
            print("Epoch [{}]:\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}\n\tr2_score: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['r2']))
        else:
            print("Epoch [{}]:\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}\n\taccuracy: {:.4f}\n\tprecision: {:.4f}\n\trecall: {:.4f}\n\tf1_score: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['accuracy'], result['precision'], result['recall'], result['f1']))

    def epoch_end_one_cycle(self, epoch, result):
        if self.task_type == 'regression':
            print("Epoch [{}]:\n\tlast_lr: {:.5f}\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}\n\tr2_score: {:.4f}".format(
                epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['r2']))
        else:
            print("Epoch [{}]:\n\tlast_lr: {:.5f}\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}\n\taccuracy: {:.4f}\n\tprecision: {:.4f}\n\trecall: {:.4f}\n\tf1_score: {:.4f}".format(
                epoch, result['lrs'][-1], result['train_loss'], result['val_loss'],result['accuracy'], result['precision'], result['recall'], result['f1']))
        
    
    def evaluate(self, val_loader):
        self.eval()
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)
    
    def fit(self, epochs, lr, train_loader, val_loader,
                      weight_decay=0, grad_clip=False, opt_func=torch.optim.Adam, lr_sched=None, start_factor:float = 1.0, end_factor:float = 1e-4, steps: int = 4, gamma: float = 0.99999, weights: list = [0.1,0.1,0.3, 1], encoder_forcing: bool = True):
        torch.cuda.empty_cache()
        history = [] # Seguimiento de entrenamiento
        onecycle = False
        linear = False
        # Poner el método de minimización personalizado
        optimizer = opt_func(self.parameters(), lr, weight_decay=weight_decay)
        #Learning rate scheduler
        if lr_sched is not None:    
            try:
                sched = lr_sched(optimizer, lr, epochs=epochs,steps_per_epoch=len(train_loader))
                onecycle = True
            except TypeError:
                try:
                    sched = lr_sched(optimizer, start_factor = start_factor, end_factor=end_factor, total_iters = epochs)
                    linear = True
                except TypeError:
                    sched = lr_sched(optimizer, step_size = round(epochs/steps), gamma = gamma)
                    linear = True
        for epoch in range(epochs):
            # Training Phase
            self.train()  #Activa calcular los vectores gradiente
            train_losses = []
            if lr_sched is not None:
                lrs = []
            for batch in train_loader:
                # Calcular el costo
                loss = self.training_step(batch, weights, encoder_forcing)
                #Seguimiento
                train_losses.append(loss)
                #Calcular las derivadas parciales
                loss.backward()

                # Gradient clipping, para que no ocurra el exploding gradient
                if grad_clip:
                    nn.utils.clip_grad_value_(self.parameters(), grad_clip)

                #Efectuar el descensod e gradiente y borrar el historial
                optimizer.step()
                optimizer.zero_grad()
                #sched step
                if onecycle:
                    lrs.append(get_lr(optimizer))
                    sched.step()
            if linear:
                lrs.append(get_lr(optimizer))
                sched.step()
            # Fase de validación
            result = self.evaluate(val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item() #Stackea todos los costos de las iteraciones sobre los batches y los guarda como la pérdida general de la época
            if lr_sched is not None:
                result['lrs'] = lrs
                self.epoch_end_one_cycle(epoch, result) #imprimir en pantalla el seguimiento
            else:
                self.epoch_end(epoch, result)

            history.append(result) # añadir a la lista el diccionario de resultados
        return history
# Base Deep Neural Network
def  SingularLayer(input_size, output):
    out = nn.Sequential(
        nn.Linear(input_size, output),
        nn.ReLU(True)
    )
    return out
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__('regression')
        self.model = nn.Linear(input_size, 1)
    def forward(self, x):
        out = self.model(x)
        return out
    
class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, *args, activation=None):
        super(DeepNeuralNetwork, self).__init__()
        
        self.overall_structure = nn.Sequential()
        #Model input and hidden layer
        for num, output in enumerate(args):
            self.overall_structure.add_module(name = f'layer_{num+1}', module = SingularLayer(input_size, output))
            input_size = output

        #Model output layer
        self.output_layer = nn.Sequential(nn.Linear(input_size, output_size))
        if activation is not None:
            self.output_layer.add_module(activation)
    def forward(self, xb):
        out = self.overall_structure(xb)
        out = self.output_layer(out)
        return out
    
# Attention based RNNs
class Attention(nn.Module):
    def __init__(self, input_size, num_heads = 1):
        super(Attention, self).__init__()
        self.attn = nn.MultiheadAttention(input_size, num_heads, batch_first = True)
        self.layer_norm = nn.LayerNorm(input_size)
    
    def forward(self, x):
        attn, _ = self.attn(x,x,x)
        context = self.layer_norm(attn)
        return context
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0, bidirectional = True, attention = True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)
        self.att = False
        if attention:
            self.attention = Attention(input_size)
            self.att = True
    def forward(self, x, hn = None, cn = None):
        batch_size,_,_ = x.size()
        if hn is None:
            hn = torch.zeros(self.num_layers * (2 if self.lstm.bidirectional else 1), batch_size, self.hidden_size, requires_grad=True, device= 'cuda')
        if cn is None:
            cn = torch.zeros(self.num_layers * (2 if self.lstm.bidirectional else 1), batch_size, self.hidden_size, requires_grad=True, device = 'cuda')
        if self.att:
            x = self.attention(x)
        out, (hn,cn) = self.lstm(x, (hn,cn))
        return out, (hn,cn)
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0, bidirectional = True, attention = True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)
        self.att = False
        if attention:
            self.attention = Attention(input_size)
            self.att = True
    def forward(self, x, hn = None):
        batch_size,_,_ = x.size()
        if hn is None:
            hn = torch.zeros(self.num_layers * (2 if self.gru.bidirectional else 1), batch_size, self.hidden_size, requires_grad=True)
        if self.att:
            x = self.attention(x)
        out, hn = self.gru(x, hn)
        return out, hn

class MC3_18(nn.Module):
    def __init__(self, hidden_state_size, architecture, dropout: float = 0.2):
        from torchvision.models.video import mc3_18, MC3_18_Weights
        super(MC3_18, self).__init__()
        
        self.model = mc3_18(weights = MC3_18_Weights.KINETICS400_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            DeepNeuralNetwork(512, hidden_state_size, *architecture),
            nn.Dropout(dropout, inplace = True)
            )

        self.transform = tt.Compose([tt.ToTensor(), MC3_18_Weights.KINETICS400_V1.transforms(antialias=True)])
    def forward(self, x):
        hn = self.model(x)
        return hn
    

class MVIT_V2_S(nn.Module):
    def __init__(self, hidden_state_size, architecture, dropout: float = 0.2):
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
        super(MVIT_V2_S, self).__init__()
        
        self.model = mvit_v2_s(weights = MViT_V2_S_Weights.KINETICS400_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.head = nn.Sequential(
            DeepNeuralNetwork(768, hidden_state_size, *architecture),
            nn.Dropout(dropout, inplace = True)
            )

        self.transform = tt.Compose([tt.ToTensor(), MViT_V2_S_Weights.KINETICS400_V1.transforms(antialias=True)])
    def forward(self, x):
        hn = self.model(x)
        return hn

class SWIN3D_B(nn.Module):
    def __init__(self, hidden_state_size, architecture, dropout: float = 0.2):
        from torchvision.models.video import swin3d_b, Swin3D_B_Weights
        super(SWIN3D_B, self).__init__()
        
        self.model = swin3d_b(weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.head = nn.Sequential(
            DeepNeuralNetwork(1024, hidden_state_size, *architecture),
            nn.Dropout(dropout, inplace = True)
            )

        self.transform = tt.Compose([tt.ToTensor(), Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1.transforms(antialias=True)])
    def forward(self, x):
        hn = self.model(x)
        return hn

class FeatureExtractorResnet50(nn.Module):
    def __init__(self, architecture):
        super(FeatureExtractorResnet50, self).__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.fc = DeepNeuralNetwork(1280, 100, *architecture)
        
        self.transform = tt.Compose([tt.ToTensor(), ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=True)])
    def forward(self, x):
        out = self.model(x)
        return out

class FeatureExtractorVGG19(nn.Module):
    def __init__(self, architecture):
        super(FeatureExtractorVGG19, self).__init__()
        from torchvision.models import vgg19, VGG19_Weights

        self.model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.classifier = DeepNeuralNetwork(1280, 100, *architecture)
        
        self.transform = tt.Compose([tt.ToTensor(), VGG19_Weights.IMAGENET1K_V1.transforms(antialias=True)])
    def forward(self, x):
        out = self.model(x)
        return out