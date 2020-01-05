import torch
from torch import nn
import numpy as np
from gspen.fastmp import fastmp
try:
    from gspen.fastmp import ilpinf
    has_ilpinf = True
except:
    has_ilpinf = False



class GSPENCRFModel(nn.Module):
    def __init__(self, unary_model, pair_model, t_model, num_nodes, pairs, num_vals, params):
        super(GSPENCRFModel, self).__init__()
        self.unary_model = unary_model
        self.pair_model = pair_model
        self.combined_pots = self.pair_model is None
        self.t_model = t_model
        self.num_nodes = num_nodes
        self.pairs = pairs
        self.num_vals = num_vals
        self.num_unary = self.num_nodes*self.num_vals
        self.num_pair = 0#len(pairs)*num_vals*num_vals
        self.num_inf_itrs = params.get('num_inf_itrs', 100)
        self.inf_lr = params.get('inf_lr', 0.05)
        self.use_sqrt_decay = params.get('use_sqrt_decay', False)
        self.use_linear_decay = params.get('use_linear_decay', False)
        self.ignore_unary_dropout = params.get('ignore_unary_dropout', False)
        self.use_relu = params.get('use_relu', False)
        self.inf_method = params.get('inf_method')
        self.gpu = params.get('gpu', False)
        self.use_loss_aug = params.get('use_loss_aug', False)
        self.use_recall_loss_aug = params.get('use_recall_loss_aug', False)
        self.mp_eps = params.get('mp_eps', 0.)
        self.mp_itrs = params.get('mp_itrs')
        self.gt_belief_interpolate = params.get('gt_belief_interpolate', -1)
        self.only_h_pair = params.get('only_h_pair', False)
        self.only_h_pair_unary = params.get('only_h_pair_unary', False)
        self.use_t_input = params.get('use_t_input', False)
        self.inf_eps = params.get('inf_eps', None)
        self.inf_region_eps = params.get('inf_region_eps', None)
        self.use_entropy = params.get('use_entropy', False)
        self.inf_loss = params.get('inf_loss', None)
        self.use_log_inf_loss = params.get('use_log_inf_loss', False)
        self.inf_loss_coef = params.get('inf_loss_coef', 1.)
        self.label_sq_coef = params.get('label_sq_coef', None)
        self.entropy_coef = params.get('entropy_coef', 1.)
        self.return_all_vals = params.get('return_all_vals', False)
        self.pot_scaling_factor = params.get('pot_scaling_factor', 1.)
        if self.gpu:
            self.tensor_mod = torch.cuda
            self.unary_model.cuda()
            if not self.combined_pots:
                self.pair_model.cuda()
            self.t_model.cuda()
        else:
            self.tensor_mod = torch
        batch_size = params['batch_size']
        self.num_potentials = num_nodes*num_vals #+ num_vals*num_vals*len(pairs)
        self.soft = nn.Softmax(dim=2)

    def get_optimizer(self, params):
        unary_lr = params['unary_lr']
        pair_lr = params['pair_lr']
        combined_lr = params.get('combined_lr', None)
        t_lr = params.get('t_lr', None)
        t_unary_lr = params.get('t_unary_lr', None)
        t_pair_lr = params.get('t_pair_lr', None)
        t_wd = params.get('t_wd', 0.)
        t_unary_wd = params.get('t_unary_wd', 0.)
        t_pair_wd = params.get('t_pair_wd', 0.)
        unary_wd = params.get('unary_wd', 0.)
        pair_wd = params.get('pair_wd', 0.)
        combined_wd = params.get('combined_wd', 0.)
        unary_mom = params.get('unary_mom', 0)
        pair_mom = params.get('pair_mom', 0)
        unary_t_mom = params.get('t_unary_mom', 0)
        t_mom = params.get('t_mom', 0)
        if self.combined_pots:
            param_groups = [{'params':self.unary_model.parameters(), 'lr':combined_lr, 'weight_decay':combined_wd}]
        else:
            param_groups = [
                    {'params':self.unary_model.parameters(), 'lr':unary_lr, 'weight_decay':unary_wd, 'momentum':unary_mom},
                    {'params':self.pair_model.parameters(), 'lr':pair_lr, 'weight_decay':pair_wd, 'momentum':pair_mom}]
        if t_lr is not None:
            param_groups.append({'params':self.t_model.parameters(), 'lr':t_lr, 'weight_decay':t_wd, 'momentum':t_mom})
        else:
            if t_unary_lr is not None:
                param_groups.append({'params':self.t_model.unary_t_model.parameters(), 'lr':t_unary_lr, 'weight_decay':t_unary_wd, 'momentum':unary_t_mom})
            if t_pair_lr is not None:
                param_groups.append({'params':self.t_model.pair_t_model.parameters(), 'lr':t_pair_lr, 'weight_decay':t_pair_wd})
        if params.get('use_adam', False):
            opt = torch.optim.Adam(param_groups)
        else:
            opt = torch.optim.SGD(param_groups)
        return opt

    def get_random_beliefs(self, num_samples, nodes, pairs):
        all_probs = []
        if nodes is None:
            beliefs = torch.FloatTensor(self.num_potentials)
            for _ in range(self.num_nodes):
                vals = [np.random.rand() for i in range(self.num_vals-1)]
                vals.sort()
                vals.append(1)
                vals.insert(0, 0)
                probs = [vals[i+1] - vals[i] for i in range(self.num_vals)]
                probabilities = torch.FloatTensor(probs)
                all_probs.append(probabilities)
            beliefs[:self.num_nodes*self.num_vals] = torch.cat(all_probs)
            offset = self.num_nodes*self.num_vals
            for pair in self.pairs:
                n1, n2 = pair
                bels1 = beliefs[n1*self.num_vals:(n1+1)*self.num_vals]
                bels2 = beliefs[n2*self.num_vals:(n2+1)*self.num_vals]
                beliefs[offset:offset+self.num_vals*self.num_vals] = torch.ger(bels2, bels1).view(-1) #This is the outer product function
                offset += self.num_vals*self.num_vals
            beliefs = beliefs.expand(num_samples, self.num_potentials).contiguous()
        else:
            #TODO: need to do similar to above, but for each individual data point
            pass
        if self.gpu:
            beliefs = beliefs.cuda(async=True)
        return beliefs

    def _find_predictions(self, is_test, epoch, inputs, pots_una, pots_pair, lossaug, labels, belief_masks=None, nodes=None, pairs=None, init_predictions=None, msgs=None, log_callback=None):
        
        pots_pair_t = torch.transpose(pots_pair, 1, 2)

        const_lambda = 0.1

        #A = torch.matmul(pots_pair, pots_pair_t) + const_lambda * torch.eye(26*5).cuda()
        A = torch.eye(26*5).cuda()

        # solve linear system
        prediction, _ = torch.gesv(pots_una.unsqueeze(-1), A) 
        prediction = prediction.view(-1, 5, 26)
        prediction = self.soft(prediction).view(-1, 5*26)

        return prediction, 0 


    

    def calculate_pots_(self, inp, belief_labels):
        una = self.unary_model(inp)
        pair = self.pair_model(inp).view(-1,5*26,32)
        
        return una, pair

    def _call_t(self, predictions, pots, inputs):
        if self.use_t_input:
            return self.t_model(predictions, pots, inputs)
        else:
            return self.t_model(predictions, pots)


    def _get_loss_aug(self, belief_labels): 
        if self.gpu:
            loss_aug = torch.cuda.FloatTensor(belief_labels.size(0), belief_labels.size(1)).fill_(0.)
        else:
            loss_aug = torch.FloatTensor(belief_labels.size(0), belief_labels.size(1)).fill_(0.)
        num_unary = self.num_nodes*self.num_vals
        loss_aug[:, :num_unary] = 1-belief_labels[:, :num_unary]
        return loss_aug

    #@profile
    def calculate_obj(self, epoch, inputs, belief_labels, belief_masks=None, nodes=None, pairs=None, init_predictions=None, messages=None):
        if self.ignore_unary_dropout:     
            self.unary_model.eval()
        else:
            self.unary_model.train()
        if not self.combined_pots:
            self.pair_model.train()
        self.t_model.eval()
        if self.use_loss_aug:
            lossaug = self._get_loss_aug(belief_labels)
        elif self.use_recall_loss_aug:
            lossaug = self._get_recall_loss_aug(belief_labels)
        else:
            lossaug = None
        
        #import pdb; pdb.set_trace()
        #pots_una, pots_pair = self.calculate_pots(inputs, belief_labels)
        pots_una, pots_pair_emb = self.calculate_pots_(inputs, belief_labels)
        
        
        #pots = torch.cat([pots_una, pots_pair], dim=1)

        predictions, num_iters = self._find_predictions(False, epoch, inputs, pots_una, pots_pair_emb, lossaug, belief_labels, belief_masks, nodes, pairs, init_predictions=init_predictions, msgs=messages)
        self.t_model.zero_grad()
        self.unary_model.zero_grad()
        self.unary_model.train()
        if not self.combined_pots:
            self.pair_model.zero_grad()
            self.pair_model.train()
        self.t_model.train()
        inf_obj = self._call_t(predictions, pots_una, inputs)
        if lossaug is not None:
            inf_obj = inf_obj + (predictions.squeeze()*lossaug).sum(dim=1)
        if self.use_entropy:
            inf_obj = inf_obj - self.entropy_coef*(predictions*torch.log(predictions+1e-6)).sum(dim=1)
        label_term = self._call_t(belief_labels, pots_una, inputs)
        obj = inf_obj - label_term
        if self.use_relu:
            obj = nn.ReLU()(obj)
        obj = obj.mean()

        #import pdb; pdb.set_trace() 
            
        if self.return_all_vals:
            return obj, inf_obj.mean(), label_term.mean(), num_iters
        else:
            return obj, inf_obj.sum()/pots_una.size(0)

    def calculate_beliefs(self, inputs, labels=None):
        self.unary_model.eval()
        if not self.combined_pots:
            self.pair_model.eval()
        self.t_model.eval()
        pots_una, pots_pair = self.calculate_pots(inputs, labels)
        result, _ = self._find_predictions(True, None, inputs, pots_una, pots_pair, None, labels)
        return result

    def predict(self, inputs, labels=None, log_callback=None):
        self.unary_model.eval()
        if not self.combined_pots:
            self.pair_model.eval()
        self.t_model.eval()
        pots_una, pots_pair = self.calculate_pots_(inputs, labels)
        
        predictions = self._find_predictions(True, None, inputs, pots_una, pots_pair, None, labels, log_callback=log_callback)[0][:,:self.num_nodes*self.num_vals].contiguous().view(-1, self.num_vals).argmax(dim=1).view(-1, self.num_nodes)
        return predictions

    def save_unary(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.unary_model.state_dict()]
            torch.save(result, fout)

    def load_unary(self, file_path):
        with open(file_path, "rb") as fin:
            if not self.gpu:
                params = torch.load(fin, map_location='cpu')
            else:
                params = torch.load(fin)
        self.unary_model.load_state_dict(params[0])

    def save_pair(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.pair_model.state_dict()]
            torch.save(result, fout)

    def load_pair(self, file_path):
        #import pdb; pdb.set_trace()
        with open(file_path, "rb") as fin:
            if not self.gpu:
                params = torch.load(fin, map_location='cpu')
            else:
                params = torch.load(fin)
        self.pair_model.load_state_dict(params[0])

    def save_t(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.t_model.state_dict()]
            torch.save(result, fout)

    def load_t(self, file_path):
        with open(file_path, "rb") as fin:
            if not self.gpu:
                params = torch.load(fin, map_location='cpu')
            else:
                params = torch.load(fin)
        self.t_model.load_state_dict(params[0])

    def save_unary_t(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.t_model.unary_t_model.state_dict()]
            torch.save(result, fout)

    def load_unary_t(self, file_path):
        with open(file_path, "rb") as fin:
            params = torch.load(fin)
        self.t_model.unary_t_model.load_state_dict(params[0])

    def save_pair_t(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.t_model.pair_t_model.state_dict()]
            torch.save(result, fout)

    def load_pair_t(self, file_path):
        with open(file_path, "rb") as fin:
            if not self.gpu:
                params = torch.load(fin, map_location='cpu')
            else:
                params = torch.load(fin)
        self.t_model.pair_t_model.load_state_dict(params[0])