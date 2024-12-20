from utils_spot import *
from slot_attn import SlotAttentionEncoder
from transformer import TransformerDecoder
from mlp import MlpDecoder
import torch
import random
import math
from torch_pca import PCA

class SPOT(nn.Module):
    def __init__(self, encoder, args, second_encoder=None):
        super().__init__()

        self.which_encoder = args.which_encoder
        self.encoder = encoder
        self.second_encoder = second_encoder
        self.encoder_final_norm = args.encoder_final_norm
        
        for param_name, param in self.encoder.named_parameters():
            if ('blocks' in param_name):
                block_id = int(param_name.split('.')[1])
                if block_id >= args.finetune_blocks_after:
                    param.requires_grad = True  # update by gradient
                else:
                    param.requires_grad = False  # not update by gradient
            else:
                param.requires_grad = False  # not update by gradient
            
        if self.second_encoder is not None:
            for param in self.second_encoder.parameters():
                param.requires_grad = False  # not update by gradient

        # Estimate number of tokens for images of size args.image_size and
        # embedding size (d_model)
        with torch.no_grad():
            x = torch.rand(1, args.img_channels, args.image_size, args.image_size)
            x = self.forward_encoder(x, self.encoder)
            _, num_tokens, d_model = x.shape

        args.d_model = d_model

        self.num_slots = args.num_slots
        self.d_model = args.d_model

        self.slot_attn = SlotAttentionEncoder(
            args.num_iterations, args.num_slots,
            args.d_model, args.slot_size, args.mlp_hidden_size, args.pos_channels,
            args.truncate, args.init_method)

        self.input_proj = nn.Sequential(
            linear(args.d_model, args.d_model, bias=False),
            nn.LayerNorm(args.d_model),
        )
        
        size = int(math.sqrt(num_tokens))
        standard_order = torch.arange(size**2) # This is the default "left_top"
        
        self.cappa = args.cappa
        self.train_permutations = args.train_permutations
        
        if self.train_permutations == 'standard':
            self.permutations = [standard_order]
            self.eval_permutations = 'standard'
        
        else:
            standard_order_2d = standard_order.reshape(size,size)
            
            perm_top_left = torch.tensor([standard_order_2d[row,col] for col in range(0, size, 1) for row in range(0, size, 1)])
            
            perm_top_right = torch.tensor([standard_order_2d[row,col] for col in range(size-1, -1, -1) for row in range(0, size, 1)])
            perm_right_top = torch.tensor([standard_order_2d[row,col] for row in range(0, size, 1) for col in range(size-1, -1, -1)])
            
            perm_bottom_right = torch.tensor([standard_order_2d[row,col] for col in range(size-1, -1, -1) for row in range(size-1, -1, -1)])
            perm_right_bottom = torch.tensor([standard_order_2d[row,col] for row in range(size-1, -1, -1) for col in range(size-1, -1, -1)])
            
            perm_bottom_left = torch.tensor([standard_order_2d[row,col] for col in range(0, size, 1) for row in range(size-1, -1, -1)])
            perm_left_bottom = torch.tensor([standard_order_2d[row,col] for row in range(size-1, -1, -1) for col in range(0, size, 1)])
            
            perm_spiral = spiral_pattern(standard_order_2d, how = 'top_right')
            perm_spiral = torch.tensor((perm_spiral[::-1]).copy())
    
            self.permutations = [standard_order, # left_top
                                 perm_top_left, 
                                 perm_top_right, 
                                 perm_right_top, 
                                 perm_bottom_right, 
                                 perm_right_bottom,
                                 perm_bottom_left,
                                 perm_left_bottom,
                                 perm_spiral
                                 ]
            self.eval_permutations = args.eval_permutations

        self.perm_ind = list(range(len(self.permutations)))

        self.bos_tokens = nn.Parameter(torch.zeros(len(self.permutations), 1, 1, args.d_model))
        torch.nn.init.normal_(self.bos_tokens, std=.02)
        
        self.dec_type = args.dec_type
        self.use_slot_proj = args.use_slot_proj
        
        if self.dec_type=='mlp' and not self.use_slot_proj:
            self.slot_proj = nn.Identity()
            self.dec_input_dim = args.slot_size
        else:
            self.slot_proj = nn.Sequential(
                linear(args.slot_size, args.d_model, bias=False),
                nn.LayerNorm(args.d_model),
            )
            self.dec_input_dim = args.d_model
        
        if self.dec_type=='transformer':
            self.dec = TransformerDecoder(
                args.num_dec_blocks, args.max_tokens, args.d_model, args.num_heads, args.dropout, args.num_cross_heads)
            if self.cappa > 0:
                assert (self.train_permutations == 'standard') and (self.eval_permutations == 'standard')   
                self.mask_token = nn.Parameter(torch.zeros(1, 1, args.d_model))
                self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, args.d_model))
                torch.nn.init.normal_(self.pos_embed, std=.02)
                torch.nn.init.normal_(self.mask_token, std=.02)
                  
        elif self.dec_type=='mlp':
            self.dec = MlpDecoder(self.dec_input_dim, args.d_model, args.max_tokens, args.mlp_dec_hidden)

            assert (self.train_permutations == 'standard') and (self.eval_permutations == 'standard')  
        else:
            raise

        if self.dec_type=='transformer':
            # Register hook for capturing the cross-attention (of the query patch
            # tokens over the key/value slot tokens) from the last decoder
            # transformer block of the decoder.
            self.dec_slots_attns = []
            def hook_fn_forward_attn(module, input):
                self.dec_slots_attns.append(input[0])
            self.remove_handle = self.dec._modules["blocks"][-1]._modules["encoder_decoder_attn"]._modules["attn_dropout"].register_forward_pre_hook(hook_fn_forward_attn)


    def forward_encoder(self, x, encoder):
        encoder.eval()

        if self.which_encoder in ['dinov2_vitb14', 'dinov2_vits14', 'dinov2_vitb14_reg', 'dinov2_vits14_reg']:
            x = encoder.prepare_tokens_with_masks(x, None)
        else:
            x = encoder.prepare_tokens(x)

        for blk in encoder.blocks:
            x = blk(x)
        if self.encoder_final_norm: # The DINOSAUR paper does not use the final norm layer according to the supplementary material.
            x = encoder.norm(x)
        
        offset = 1
        if self.which_encoder in ['dinov2_vitb14_reg', 'dinov2_vits14_reg']:
            offset += encoder.num_register_tokens
        elif self.which_encoder in ['simpool_vits16']:
            offset += -1
        x = x[:, offset :] # remove the [CLS] and (if they exist) registers tokens 

        return x

    def forward_decoder(self, slots, emb_target):
        # Prepate the input tokens for the decoder transformer:
        # (1) insert a learnable beggining-of-sequence ([BOS]) token at the beggining of each target embedding sequence.
        # (2) remove the last token of the target embedding sequence
        # (3) no need to add positional embeddings since positional information already exists at the DINO's outptu.
        

        if self.training:
            if self.train_permutations == 'standard':
                which_permutations = [0] # USE [0] FOR THE STANDARD ORDER
            elif self.train_permutations == 'random':
                which_permutations = [random.choice(self.perm_ind)]
            elif self.train_permutations == 'all':
                which_permutations = self.perm_ind
            else:
                raise
        else:
            if self.eval_permutations == 'standard':
                which_permutations = [0] # USE [0] FOR THE STANDARD ORDER
            elif self.eval_permutations == 'random':
                which_permutations = [random.choice(self.perm_ind)]
            elif self.eval_permutations == 'all':
                which_permutations = self.perm_ind
            else:
                raise
        
        
        all_dec_slots_attns = []
        all_dec_output = []
        
        for perm_id in which_permutations:
            current_perm = self.permutations[perm_id]

            bos_token = self.bos_tokens[perm_id]
            bos_token = bos_token.expand(emb_target.shape[0], -1, -1)
            
            use_pos_emb = self.cappa > 0
            parallel_dec = self.cappa > 0 and ((self.cappa >= 1.0) or (self.training and random.random() < self.cappa))
            #print(f"Paralled Decoder (CAPPA) {parallel_dec}")
            # Input to the decoder
            if parallel_dec: # Use parallel decoder
                dec_input = self.mask_token.to(emb_target.dtype).expand(emb_target.shape[0], -1, -1)
            else: # Use autoregressive decoder
                dec_input = torch.cat((bos_token, emb_target[:,current_perm,:][:, :-1, :]), dim=1)
      
            if use_pos_emb:
                # Add position embedding if they exist.
                dec_input = dec_input + self.pos_embed.to(emb_target.dtype)

            # dec_input has the same shape as emb_target, which is [B, N, D]
            dec_input = self.input_proj(dec_input)
    
            # Apply the decoder
            dec_input_slots = self.slot_proj(slots) # shape: [B, num_slots, D]
            if self.dec_type=='transformer':
                dec_output = self.dec(dec_input, dec_input_slots, causal_mask=(not parallel_dec))
                # decoder_output shape [B, N, D]

                dec_slots_attns = self.dec_slots_attns[0]
                self.dec_slots_attns = []

                # sum over the heads and 
                dec_slots_attns = dec_slots_attns.sum(dim=1) # [B, N, num_slots]
                # dec_slots_attns shape [B, num_heads, N, num_slots]
                # L1-normalize over the slots so as to sum to 1.
                dec_slots_attns = dec_slots_attns / dec_slots_attns.sum(dim=2, keepdim=True)

                inv_current_perm = torch.argsort(current_perm)
                dec_slots_attns = dec_slots_attns[:,inv_current_perm,:]
                dec_output = dec_output[:,inv_current_perm,:]

            elif self.dec_type=='mlp':
                dec_output, dec_slots_attns = self.dec(dec_input_slots)
                dec_slots_attns = dec_slots_attns.transpose(1,2)

            else:
                raise
            
            all_dec_slots_attns.append(dec_slots_attns)
            all_dec_output.append(dec_output)

        mean_dec_slots_attns = torch.stack(all_dec_slots_attns).mean(0)
        mean_dec_output = torch.stack(all_dec_output).mean(0)

        return mean_dec_output, mean_dec_slots_attns

    def get_embeddings_n_slots(self, image):
        """
        image: batch_size x img_channels x H x W
        """

        B, _, H, W = image.size()
        with torch.no_grad():
            emb_target = self.forward_encoder(image, self.encoder)
        # emb_target shape: B, N, D

        # Apply the slot attention
        slots, slots_attns, _ = self.slot_attn(emb_target)
        return emb_target, slots, slots_attns

    def forward(self, image):
        """
        image: batch_size x img_channels x H x W
        """

        B, _, H, W = image.size()
        emb_input = self.forward_encoder(image, self.encoder)
        with torch.no_grad():
            if self.second_encoder is not None:
                emb_target = self.forward_encoder(image, self.second_encoder)
            else:
                emb_target = emb_input.clone().detach()
        # emb_target shape: B, N, D

        # Apply the slot attention
        slots, slots_attns, init_slots, attn_logits = self.slot_attn(emb_input)
        attn_logits = attn_logits.squeeze()
        # slots shape: [B, num_slots, Ds]
        # slots_attns shape: [B, N, num_slots]

        # Apply the decoder.
        dec_recon, dec_slots_attns = self.forward_decoder(slots, emb_target)

        # Mean-Square-Error loss
        H_enc, W_enc = int(math.sqrt(emb_target.shape[1])), int(math.sqrt(emb_target.shape[1]))
        loss_mse = ((emb_target - dec_recon) ** 2).sum()/(B*H_enc*W_enc*self.d_model)

        # Reshape the slot and decoder-slot attentions.
        slots_attns = slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)
        dec_slots_attns = dec_slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)

        return loss_mse, slots_attns, dec_slots_attns, slots, dec_recon, attn_logits

class DINOSlot(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()

        self.which_encoder = args.which_encoder
        self.encoder = encoder
        self.encoder_final_norm = args.encoder_final_norm
        
        for param in self.encoder.parameters():
            param.requires_grad = False  # not update by gradient
            
        # Estimate number of tokens for images of size args.image_size and
        # embedding size (d_model)
        with torch.no_grad():
            x = torch.rand(1, args.img_channels, args.image_size, args.image_size)
            x = self.forward_encoder(x, self.encoder)
            _, num_tokens, d_model = x.shape

        args.d_model = d_model

        self.num_slots = args.num_slots
        self.d_model = args.d_model

        self.slot_attn = SlotAttentionEncoder(
            args.num_iterations, 
            args.num_slots,
            args.d_model, 
            args.slot_size, 
            args.mlp_hidden_size, 
            args.pos_channels,
            args.truncate, 
            args.init_method
        )

        self.input_proj = nn.Sequential(
            linear(args.d_model, args.d_model, bias=False),
            nn.LayerNorm(args.d_model),
        )
        
        self.bos_tokens = nn.Parameter(torch.zeros(1, 1, args.d_model))
        torch.nn.init.normal_(self.bos_tokens, std=.02)
        
        self.dec_type = args.dec_type
        self.use_slot_proj = args.use_slot_proj
        
        self.dec = TransformerDecoder(
            args.num_dec_blocks, 
            args.max_tokens, 
            args.d_model, 
            args.num_heads, 
            args.dropout, 
            args.num_cross_heads
        )
                
        self.dec_slots_attns = []
        def hook_fn_forward_attn(module, input):
            self.dec_slots_attns.append(input[0])
        self.remove_handle = self.dec._modules["blocks"][-1]._modules["encoder_decoder_attn"]._modules["attn_dropout"].register_forward_pre_hook(hook_fn_forward_attn)


    def forward_encoder(self, x, encoder):
        encoder.eval()

        if self.which_encoder in ['dinov2_vitb14', 'dinov2_vits14', 'dinov2_vitb14_reg', 'dinov2_vits14_reg']:
            x = encoder.prepare_tokens_with_masks(x, None)
        else:
            x = encoder.prepare_tokens(x)

        for blk in encoder.blocks:
            x = blk(x)
        if self.encoder_final_norm: # The DINOSAUR paper does not use the final norm layer according to the supplementary material.
            x = encoder.norm(x)
        
        offset = 1
        if self.which_encoder in ['dinov2_vitb14_reg', 'dinov2_vits14_reg']:
            offset += encoder.num_register_tokens
        elif self.which_encoder in ['simpool_vits16']:
            offset += -1
        x = x[:, offset :] # remove the [CLS] and (if they exist) registers tokens 

        return x

    def forward_decoder(self, slots, emb_target):
        # Prepate the input tokens for the decoder transformer:
        # (1) insert a learnable beggining-of-sequence ([BOS]) token at the beggining of each target embedding sequence.
        # (2) remove the last token of the target embedding sequence
        # (3) no need to add positional embeddings since positional information already exists at the DINO's outptu.

        bos_token = self.bos_tokens
        bos_token = bos_token.expand(emb_target.shape[0], -1, -1)
       
        dec_input = torch.cat((bos_token, emb_target[:, :-1, :]), dim=1)
    
        # dec_input has the same shape as emb_target, which is [B, N, D]
        dec_input = self.input_proj(dec_input)

        # Apply the decoder
        dec_input_slots = self.slot_proj(slots) # shape: [B, num_slots, D]
        dec_output = self.dec(dec_input, dec_input_slots, causal_mask=True)
        # decoder_output shape [B, N, D]

        dec_slots_attns = self.dec_slots_attns[0]
        self.dec_slots_attns = []

        # sum over the heads and 
        dec_slots_attns = dec_slots_attns.sum(dim=1) # [B, N, num_slots]
        # dec_slots_attns shape [B, num_heads, N, num_slots]
        # L1-normalize over the slots so as to sum to 1.
        dec_slots_attns = dec_slots_attns / dec_slots_attns.sum(dim=2, keepdim=True)

        return dec_output, dec_slots_attns

    def get_embeddings_n_slots(self, image):
        """
        image: batch_size x img_channels x H x W
        """

        B, _, H, W = image.size()
        with torch.no_grad():
            emb_target = self.forward_encoder(image, self.encoder)
        # emb_target shape: B, N, D

        # Apply the slot attention
        slots, slots_attns, _ = self.slot_attn(emb_target)
        return emb_target, slots, slots_attns

    def forward(self, image):
        """
        image: batch_size x img_channels x H x W
        """

        B, _, H, W = image.size()
        emb_input = self.forward_encoder(image, self.encoder)
        with torch.no_grad():
            emb_target = emb_input.clone().detach()
        # emb_target shape: B, N, D

        # Apply the slot attention
        slots, slots_attns, init_slots, attn_logits = self.slot_attn(emb_input)
        attn_logits = attn_logits.squeeze()
        # slots shape: [B, num_slots, Ds]
        # slots_attns shape: [B, N, num_slots]

        # Apply the decoder.
        dec_recon, dec_slots_attns = self.forward_decoder(slots, emb_target)

        # Mean-Square-Error loss
        H_enc, W_enc = int(math.sqrt(emb_target.shape[1])), int(math.sqrt(emb_target.shape[1]))
        loss_mse = ((emb_target - dec_recon) ** 2).sum()/(B*H_enc*W_enc*self.d_model)

        # Reshape the slot and decoder-slot attentions.
        slots_attns = slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)
        dec_slots_attns = dec_slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)

        return loss_mse, slots_attns, dec_slots_attns, slots, dec_recon, attn_logits



def get_normalizer(norm, channels, groups=16, dim='2d'):
    """Get normalization layer."""
    if norm == '':
        return nn.Identity()
    elif norm == 'bn':
        return eval(f'nn.BatchNorm{dim}')(channels)
    elif norm == 'gn':
        # 16 is taken from Table 3 of the GN paper
        return nn.GroupNorm(groups, channels)
    elif norm == 'in':
        return eval(f'nn.InstanceNorm{dim}')(channels)
    elif norm == 'ln':
        return nn.LayerNorm(channels)
    else:
        raise ValueError(f'Normalizer {norm} not supported!')


def get_act_func(act):
    """Get activation function."""
    if act == '':
        return nn.Identity()
    if act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'leakyrelu':
        return nn.LeakyReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'swish':
        return nn.SiLU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'softplus':
        return nn.Softplus()
    elif act == 'mish':
        return nn.Mish()
    elif act == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Activation function {act} not supported!')
    
def deconv_norm_act(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    norm='bn',
    act='relu',
    dim='2d',
):
    """ConvTranspose - Norm - Act."""
    deconv = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        output_padding=stride - 1,
        dilation=dilation,
        groups=groups,
        bias=norm not in ['bn', 'in'],
    )
    normalizer = get_normalizer(norm, out_channels, dim=dim)
    act_func = get_act_func(act)
    return nn.Sequential(deconv, normalizer, act_func)

def deconv_out_shape(
    in_size,
    stride,
    padding,
    kernel_size,
    out_padding,
    dilation=1,
):
    """Calculate the output shape of a ConvTranspose layer."""
    if isinstance(in_size, int):
        return (in_size - 1) * stride - 2 * padding + dilation * (
            kernel_size - 1) + out_padding + 1
    elif isinstance(in_size, (tuple, list)):
        return type(in_size)((deconv_out_shape(s, stride, padding, kernel_size,
                                               out_padding, dilation)
                              for s in in_size))
    else:
        raise TypeError(f'Got invalid type {type(in_size)} for `in_size`')

def build_grid(resolution):
    """return grid with shape [1, H, W, 4]."""
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)

class SoftPositionEmbed(nn.Module):
    """Soft PE mapping normalized coords to feature maps."""

    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.dense = nn.Linear(in_features=4, out_features=hidden_size)
        self.register_buffer('grid', build_grid(resolution))  # [1, H, W, 4]

    def forward(self, inputs):
        """inputs: [B, C, H, W]."""
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2).contiguous()
        return inputs + emb_proj
    
class DINOUp(nn.Module):
    def __init__(self, upsampler, args):
        super().__init__()
        self.which_encoder = args.which_encoder
        self.upsampler = upsampler
        self.encoder_final_norm = args.encoder_final_norm
        self.image_size = args.image_size
        
        for param in self.upsampler.parameters():
            param.requires_grad = False  # not update by gradient
            
        # Estimate number of tokens for images of size args.image_size and
        # embedding size (d_model)
        with torch.no_grad():
            x = torch.rand(1, args.img_channels, args.image_size, args.image_size)
            self.upsampler.eval()
            x = self.upsampler.model(x)
            _, d_model, _, _ = x.shape

        args.d_model = d_model

        self.num_slots = args.num_slots
        self.d_model = args.d_model

        self.slot_attn = SlotAttentionEncoder(
            args.num_iterations, 
            args.num_slots,
            args.d_model, 
            args.slot_size, 
            args.mlp_hidden_size, 
            args.pos_channels,
            args.truncate, 
            args.init_method
        )
        
        """proj_size = int(args.slot_size//2)
        self.slot_proj = nn.Sequential(
            linear(args.slot_size, proj_size, bias=False),
            nn.LayerNorm(proj_size),
        )"""
        """self.dec = MlpDecoder(
            object_dim=proj_size, 
            output_dim=30, 
            num_patches=args.image_size**2, 
            hidden_features=args.mlp_dec_hidden
        )"""

        self.dec_channels = [args.slot_size, 64, 64, 64, 64, 64]
        self.dec_resolution = (7, 7) # broadcast size
        self.dec_ks = 5  # kernel size
        self.dec_norm = ''  # norm in CNN
        self.decoder_pos_embedding  = None
        modules = []
        in_size = self.dec_resolution[0]
        out_size = in_size
        stride = 2
        for i in range(len(self.dec_channels) - 1):
            if out_size == self.image_size:
                stride = 1
            modules.append(
                deconv_norm_act(
                    self.dec_channels[i],
                    self.dec_channels[i + 1],
                    kernel_size=self.dec_ks,
                    stride=stride,
                    norm=self.dec_norm,
                    act='relu'))
            out_size = deconv_out_shape(out_size, stride, self.dec_ks // 2,
                                        self.dec_ks, stride - 1)

        # out Conv for RGB and seg mask
        modules.append(
            nn.Conv2d(
                self.dec_channels[-1], args.d_model+1, kernel_size=1, stride=1, padding=0))

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(args.slot_size,
                                                    self.dec_resolution)

        # self.pca = PCA(n_components=30,) # TODO : choose the right num of components

    def forward(self, image):
        """
        image: batch_size x img_channels x H x W
        """

        B, _, H, W = image.size()
        self.upsampler.eval()
        emb_input = self.upsampler.model(image)
        H_enc, W_enc = emb_input.shape[-1], emb_input.shape[-1]
        with torch.no_grad():
            emb_target = self.upsampler.upsampler(emb_input, image).clone().detach().flatten(-2, -1).permute(0, 2, 1)
            B, N, C = emb_target.shape
            # emb_target = self.pca.fit_transform(emb_target.flatten(0, 1)) # Reduce the dimensionality of the embeddings
            emb_target = emb_target.reshape(B, N, -1)
        # emb_target shape: B, N, D ==> here high res
        emb_input = emb_input.flatten(2).transpose(1, 2)
        
        # Apply the slot attention
        slots, slots_attns, _, attn_logits = self.slot_attn(emb_input)
        B, S, D = slots.shape
        attn_logits = attn_logits.squeeze()

        # Apply the decoder.
        # dec_input_slots = self.slot_proj(slots) # shape: [B, num_slots, D]
        # recons, dec_masks = self.dec(dec_input_slots)
        decoder_in = slots.view(B * S, D, 1, 1)
        decoder_in = decoder_in.repeat(1, 1, self.dec_resolution[0], self.dec_resolution[1])
        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        out = out.view(B, S, C + 1, self.image_size, self.image_size)
        recons = out[:, :, :C, :, :]  # [B, num_slots, 3, H, W]
        dec_masks = out[:, :, -1:, :, :]
        dec_masks = F.softmax(dec_masks, dim=1)  # [B, num_slots, 1, H, W]
        recons = torch.sum(recons * dec_masks, dim=1)  # [B, C, H, W]
        recons = recons.flatten(2, 3).permute(0, 2, 1)

        # Mean-Square-Error loss
        loss_mse = ((emb_target - recons) ** 2).sum()/(B*self.image_size*self.image_size*self.d_model)

        # Reshape the slot and decoder-slot attentions.
        slots_attns = slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)
        dec_masks = dec_masks.transpose(-1, -2).reshape(B, self.num_slots, self.image_size, self.image_size)

        return loss_mse, slots_attns, dec_masks, slots, attn_logits, None


class SPOTEval(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()

        self.encoder = encoder

        # Estimate number of tokens for images of size args.image_size and
        # embedding size (d_model)
        with torch.no_grad():
            x = torch.rand(1, 3, args.image_size, args.image_size)
            x = self.forward_encoder(x, self.encoder)
            _, num_tokens, embed_dim = x.shape


        self.num_slots = args.num_obj
        self.embed_dim = embed_dim

        self.slot_attn = SlotAttentionEncoder(
            3, args.num_obj,
            embed_dim, args.dim_slot, args.dim_hidden, 4,
            "none", "shared_gaussian")


        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        self.bos_tokens = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        torch.nn.init.normal_(self.bos_tokens, std=.02)
        
        self.slot_proj = nn.Sequential(
            nn.Linear(args.dim_slot, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.dec_input_dim = embed_dim

        self.dec = TransformerDecoder(4, num_tokens, embed_dim, 6, 0.0, None)

        self.dec_slots_attns = []
        def hook_fn_forward_attn(module, input):
            self.dec_slots_attns.append(input[0])
        self.remove_handle = self.dec._modules["blocks"][-1]._modules["encoder_decoder_attn"]._modules["attn_dropout"].register_forward_pre_hook(hook_fn_forward_attn)


    def forward_encoder(self, x, encoder):
        encoder.eval()

        x = encoder.prepare_tokens(x)

        for blk in encoder.blocks:
            x = blk(x)
        
        x = x[:, 1:] # remove the [CLS] and (if they exist) registers tokens 

        return x

    def forward_decoder(self, slots, emb_target):
        # Prepate the input tokens for the decoder transformer:
        # (1) insert a learnable beggining-of-sequence ([BOS]) token at the beggining of each target embedding sequence.
        # (2) remove the last token of the target embedding sequence
        # (3) no need to add positional embeddings since positional information already exists at the DINO's outptu.
        
        bos_token = self.bos_tokens[0].expand(emb_target.shape[0], -1, -1)
        dec_input = torch.cat((bos_token, emb_target[:, :-1, :]), dim=1)
    
        # Proj
        dec_input = self.input_proj(dec_input)
        dec_input_slots = self.slot_proj(slots) # shape: [B, num_slots, D]

        dec_output = self.dec(dec_input, dec_input_slots, causal_mask=True)

        dec_slots_attns = self.dec_slots_attns[0]
        self.dec_slots_attns = []

        # sum over the heads and 
        dec_slots_attns = dec_slots_attns.sum(dim=1) # [B, N, num_slots]
        # dec_slots_attns shape [B, num_heads, N, num_slots]
        # L1-normalize over the slots so as to sum to 1.
        dec_slots_attns = dec_slots_attns / dec_slots_attns.sum(dim=2, keepdim=True)

        return dec_output, dec_slots_attns

    def forward(self, image):
        """
        image: batch_size x img_channels x H x W
        """

        B, _, H, W = image.size()
        emb_input = self.forward_encoder(image, self.encoder)
        emb_target = emb_input.clone().detach()
        # emb_target shape: B, N, D

        # Apply the slot attention
        slots, slots_attns, init_slots, attn_logits = self.slot_attn(emb_input)
        # slots shape: [B, num_slots, Ds]
        # slots_attns shape: [B, N, num_slots]

        # Apply the decoder.
        dec_recon, dec_slots_attns = self.forward_decoder(slots, emb_target)

        # Mean-Square-Error loss
        H_enc, W_enc = int(math.sqrt(emb_target.shape[1])), int(math.sqrt(emb_target.shape[1]))
        loss_mse = ((emb_target - dec_recon) ** 2).sum()/(B*H_enc*W_enc*self.embed_dim)

        # Reshape the slot and decoder-slot attentions.
        slots_attns = slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)
        dec_slots_attns = dec_slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)

        return loss_mse, slots_attns, dec_slots_attns, slots, dec_recon, attn_logits
