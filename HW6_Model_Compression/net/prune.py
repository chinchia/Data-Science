import numpy as np
import torch
from torch.nn.modules.module import Module


class PruningModule(Module):
    def prune_by_percentile(self, q={'conv1': 16, 'conv2': 62, 'conv3': 65, 'conv4': 63, 'conv5': 63, 'fc1': 91, 'fc2': 91, 'fc3': 75}):
        ########################
        # TODO
        # 	For each layer of weights W (including fc and conv layers) in the model, obtain the qth percentile of W as
        # 	the threshold, and then set the nodes with weight W less than threshold to 0, and the rest remain unchanged.
        ########################
        
        # Calculate percentile value
        params = []
        for name, p in self.named_parameters():
            if 'bias' in name:
                continue
            tensor = p.data.cpu().numpy()
            alive_params = tensor[np.nonzero(tensor)]
            params.append(alive_params)
    
        all_params = np.concatenate(params)
        
        # Prune the weights and mask
        for name, module in self.named_modules():
            if name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
                percentile_value = np.percentile(abs(all_params), q[name])
                print(f'Pruning with threshold : {percentile_value} for layer {name}')
                self.prune(module, percentile_value)
                
            if name in ['fc1', 'fc2', 'fc3']:
                percentile_value = np.percentile(abs(all_params), q[name])
                print(f'Pruning with threshold : {percentile_value} for layer {name}')
                self.prune(module, percentile_value)

    def prune_by_std(self, s=0.25):
        for name, module in self.named_modules():

            #################################
            # TODO:
            #    Only fully connected layers were considered, but convolution layers also needed
            #################################
            
            if name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s * 0.243531
                print(f'Pruning with threshold : {threshold} for layer {name}')
                self.prune(module, threshold)
                
            if name in ['fc1', 'fc2', 'fc3']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold} for layer {name}')
                self.prune(module, threshold)

    def prune(self, module, threshold):

        #################################
        # TODO:
        #    1. Use "module.weight.data" to get the weights of a certain layer of the model
        #    2. Set weights whose absolute value is less than threshold to 0, and keep the rest unchanged
        #    3. Save the results of the step 2 back to "module.weight.data"
        #    --------------------------------------------------------
        #    In addition, there is no need to return in this function ("module" can be considered as call by
        #    reference)
        #################################
        
        weight_device = module.weight.device        
        tensor = module.weight.data.cpu().numpy()
        mask = torch.nn.Parameter(torch.from_numpy(np.ones_like(tensor)), requires_grad=False)
        new_mask = np.where(abs(tensor)<threshold, 0, mask)        
        module.weight.data = torch.from_numpy(tensor * new_mask).to(weight_device)

