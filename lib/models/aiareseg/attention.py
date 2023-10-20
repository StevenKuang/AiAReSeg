import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn.functional import *
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.init import xavier_uniform_
from torch.nn.modules.linear import Linear
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class InnerAttention(Module):
    __constants__ = ['batch_first']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(InnerAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))

        if bias: # This is usually true, so you are defining the bias
            self.q_proj_bias = Parameter(torch.empty(embed_dim, **factory_kwargs))
            self.k_proj_bias = Parameter(torch.empty(embed_dim, **factory_kwargs))
        else:
            self.register_parameter('q_proj_bias', None)
            self.register_parameter('k_proj_bias', None)
        self.out_proj = Linear(self.vdim, self.vdim, bias=bias)

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier uniform is an initialization distribution, where the tensor's weights are samples from a uniform distribution
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)

        # If the outward projection is not none, so you add bias to the outward projection
        if self.out_proj.bias is not None:
            constant_(self.q_proj_bias, 0.)
            constant_(self.k_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def _in_projection(
            self,
            q: Tensor,
            k: Tensor,
            w_q: Tensor,
            w_k: Tensor,
            b_q: Optional[Tensor] = None,
            b_k: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return linear(q, w_q, b_q), linear(k, w_k, b_k)

    def inner_scaled_dot_product_attention(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            attn_mask: Optional[Tensor] = None,
            dropout_p: float = 0.0
    ) -> Tuple[Tensor, Tensor]:
        """
        This is the scaled inner dot product that is needed for attention
        """
        B, Nt, E = q.shape # Batch size,
        q = q / math.sqrt(E) # The query will be scaled
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1)) # Batch matrix multiplication between the queries and the keys
        if attn_mask is not None:
            attn += attn_mask # Add the mask if it is provided
        attn = softmax(attn, dim=-1) # pass through a softmax
        if dropout_p > 0.0:
            attn = dropout(attn, p=dropout_p) # Apply dropout if it is provided
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v) # Multiply the result with the value
        return output, attn

    def inner_attention_forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            embed_dim_to_check: int,
            num_heads: int,
            q_proj_bias: Optional[Tensor],
            k_proj_bias: Optional[Tensor],
            add_zero_attn: bool,
            dropout_p: float,
            out_proj_weight: Tensor,
            out_proj_bias: Optional[Tensor],
            training: bool = True,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            q_proj_weight: Optional[Tensor] = None,
            k_proj_weight: Optional[Tensor] = None,
            static_k: Optional[Tensor] = None,
            static_v: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        # Check the head dimension and store it
        if isinstance(embed_dim, torch.Tensor):
            # Embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads

        # Compute in-projection
        if q_proj_bias is None:
            b_q = b_k = None
        else:
            b_q, b_k = q_proj_bias, k_proj_bias
        # Compute the projection of the queries and the keys inputs. This is simply applying a linear transform to the q and k's, with the weights and biases as defined before
        q, k = self._in_projection(query, key, q_proj_weight, k_proj_weight, b_q, b_k)
        v = value

        # Prep attention mask. Remember, attention mask is simply a mask that removes the model's ability to apply attention to parts of the query array.
        # Masking simply prevents the attention network from cheating in the decoder during training. Basically this is done by masking the future information, and shielding it
        # from the previous tokens
        # Masks are recommended for decoder, as you dont want to provide information of the future to make the decision at this instance
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)
            # Ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)

        # Prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            key_padding_mask = key_padding_mask.to(torch.bool)

        # Reshape q, k, v for multihead attention and make em batch first
        # Returns the neighbouring tensor in memory that contains the same data as the self tensoor. If the tensor is already in memory then simply return the tensor.
        # Now, we try to tweak the dimensions of the query to be target length, batch size * heads, head dimensions, and then transpose the first and second dimensions
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        # The keys, we dont care about its target length, but since we are multiplying by the query, we want to make sure the dimensiosn match up for multiplication

        # Note that we have the options to keep the keys and the values constant/static if you want
        if static_k is None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        else:
            k = static_k
        if static_v is None:
            v = v.contiguous().view(-1, bsz * num_heads, self.vdim // self.num_heads).transpose(0, 1)
        else:
            v = static_v

        # Add zero attention along batch dimension (what is zero attention?)
        if add_zero_attn:
            # The zero attention is probably for padding?
            # First ensure that the sizes match up so that the rest (batchsize * heads, target length, head dimension)
            zero_attn_shape = (bsz * num_heads, 1, head_dim)
            # For the keys, we concatenate the zeros with the shape as specified before. There is always one zero in the array.
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            #  You can input an attention mask if you want to
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            # You can pad the keys with masks if you want to
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        # Update source sequence length after adjustments
        src_len = k.size(1)

        # Merge key padding and attention masks
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float('-inf'))

        # Convert mask to float
        # If there is an attention mask, convert it to float, and then now we create the -infinity terms that we need to add to the terms on the output, such that
        # When it is passed through the softmax function, it masks out the future term, or the term that we dont want
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float('-inf'))
            attn_mask = new_attn_mask

        # Adjust dropout probability
        if not training:
            dropout_p = 0.0

        # Calculate attention and out projection
        # Now finally we calculate attention
        attn_output, attn_output_weights = self.inner_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        # Reshape the attention output
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.vdim)
        # Skip connections
        attn_output = attn_output + linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # Average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        # This is usually off
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        # Here we call the inner attention forward function
        attn_output, attn_output_weights = self.inner_attention_forward(
            query, key, value, self.embed_dim, self.num_heads, self.q_proj_bias, self.k_proj_bias,
            self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight)

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        # And finally just output the attention weights
        else:
            return attn_output, attn_output_weights


class CorrAttention(Module):

    def __init__(self, num_heads, dropout, match_dim, feat_size):
        super(CorrAttention, self).__init__()
        self.match_dim = match_dim
        self.feat_size = feat_size
        self.corr_proj = nn.Linear(self.feat_size, self.match_dim)
        self.corr_attn = InnerAttention(self.match_dim, 1, dropout=dropout, vdim=self.feat_size)
        self.feat_norm1 = nn.LayerNorm(self.match_dim)
        self.feat_norm2 = nn.LayerNorm(self.feat_size)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads

    def forward(self, corr_map, pos_emb):
        """ This unit is the one that truely implements the double attention in attention that the paper introduces"""
        # Grabs the batch size
        batch_size = pos_emb.shape[1]
        # Grabs the positional embedding, repeats it across the number of heads, and then swaps it so we have the number of heads first, and then reshapes it to that we got the matching dimension, and the feat dimensions, and then we swap the match dimension and the feature size
        pos_emb = torch.repeat_interleave(pos_emb, self.num_heads, dim=1).transpose(0, -1).reshape(self.match_dim, -1,
                                                                                                   self.feat_size).transpose(
            0, -1)
        # Now we grab the correlation map, reshape it into feature size
        corr_map = corr_map.transpose(0, 1).reshape(self.feat_size, -1, self.feat_size)
        corr_map = corr_map.transpose(0, -1)  # From the perspective of keys
        # Apply a FC linear layer to the correlation map, then normalize it, and add the positional encoding
        q = k = self.feat_norm1(self.corr_proj(corr_map)) + pos_emb
        # Make the queries and the keys the same, and both of which are simply the correlation map that is applied as input to this method
        # Now, we apply attention
        corr_map1 = self.corr_attn(q, k, value=self.feat_norm2(corr_map))[0]
        # Then we apply drop out
        corr_map = self.dropout(corr_map1)
        # Transpose and reshpae the correlation map such that it becomes the number of heads * batch size, feature size, -1
        corr_map = corr_map.transpose(0, -1)
        corr_map = corr_map.reshape(self.feat_size, self.num_heads * batch_size, -1).transpose(0, 1)
        return corr_map


class AiAModule(Module):
    """
    This class builds the attention in attention framework central to this publication
    """

    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None, use_AiA=True, match_dim=64,
                 feat_size=400) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AiAModule, self).__init__()
        # Define the number of embedding dimensions
        self.embed_dim = embed_dim
        # Define the dimensions of the keys, if not same as embedding size
        self.kdim = kdim if kdim is not None else embed_dim
        # Define the dimensions of the value, if not same as embedding size
        self.vdim = vdim if vdim is not None else embed_dim
        # Check if the dimensions of queries, keys and values are the same
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        # Define the number of attention heads
        self.num_heads = num_heads
        # Define the dropout probability
        self.dropout = dropout
        # Define the batch
        self.batch_first = batch_first
        # Define how much of the embedding is going into each attention head
        self.head_dim = embed_dim // num_heads

        self.use_AiA = use_AiA
        if self.use_AiA:
            # Apply the attention in attention module
            self.inner_attn = CorrAttention(num_heads, dropout, match_dim, feat_size)

        # Define the weights of the q, k and v projections, if the dimensions are not the same as the embedding dimensions.
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)

        # We register the parameters onto the model, effectively adding it to the module
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        # Add the bias terms, if they are defined
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        # If there are no bias, then we will need to create the bias parameter and once again register it to the module
        else:
            self.register_parameter('in_proj_bias', None)
        # The output projection layer is defined as a linear layer
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        # This is the bias layer added to the keys and values embeddings themselves
        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        # If no bias then simply dont put bias
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(AiAModule, self).__setstate__(state)

    def _in_projection_packed(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            w: Tensor,
            b: Optional[Tensor] = None,
    ) -> List[Tensor]:
        E = q.size(-1)
        if k is v:
            if q is k:
                # Self-attention
                return linear(q, w, b).chunk(3, dim=-1)
            else:
                # Encoder-decoder attention
                w_q, w_kv = w.split([E, E * 2])
                if b is None:
                    b_q = b_kv = None
                else:
                    b_q, b_kv = b.split([E, E * 2])
                return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
        else:
            w_q, w_k, w_v = w.chunk(3)
            if b is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = b.chunk(3)
            return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

    def _in_projection(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            w_q: Tensor,
            w_k: Tensor,
            w_v: Tensor,
            b_q: Optional[Tensor] = None,
            b_k: Optional[Tensor] = None,
            b_v: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

    def aia_scaled_dot_product_attention(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            attn_mask: Optional[Tensor] = None,
            dropout_p: float = 0.0,
            pos_emb=None
    ) -> Tuple[Tensor, Tensor]:

        """
        This is the magic function that does two layers of attention
        """
        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        # This is the first attention we are applying, simply an attention between the input queries and keys
        attn = torch.bmm(q, k.transpose(-2, -1))

        # This is the second attention that we are applying, the inner attention that simply grabs the attention map and then applies self attention
        if self.use_AiA:
            corr_map = attn
            corr_map = self.inner_attn(corr_map, pos_emb)
            attn = attn + corr_map

        # We comment out the following two lines since applying mask to the padding regions doesn't have obvious influence on the performance
        # You can use it if you like (by removing the comment), but the model should be retrained with the padding mask
        # if attn_mask is not None:
        #     attn += attn_mask

        attn = softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = dropout(attn, p=dropout_p)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn

    def aia_attention_forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            embed_dim_to_check: int,
            num_heads: int,
            in_proj_weight: Tensor,
            in_proj_bias: Optional[Tensor],
            bias_k: Optional[Tensor],
            bias_v: Optional[Tensor],
            add_zero_attn: bool,
            dropout_p: float,
            out_proj_weight: Tensor,
            out_proj_bias: Optional[Tensor],
            training: bool = True,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            use_separate_proj_weight: bool = False,
            q_proj_weight: Optional[Tensor] = None,
            k_proj_weight: Optional[Tensor] = None,
            v_proj_weight: Optional[Tensor] = None,
            static_k: Optional[Tensor] = None,
            static_v: Optional[Tensor] = None,
            pos_emb=None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        if isinstance(embed_dim, torch.Tensor):
            # Embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads

        # Compute in-projection
        if not use_separate_proj_weight:
            q, k, v = self._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            q, k, v = self._in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

        # Prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)
            # Ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)

        # Prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            key_padding_mask = key_padding_mask.to(torch.bool)

        # Add bias along batch dimension
        if bias_k is not None and bias_v is not None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None
            assert bias_v is None

        # Reshape q, k, v for multihead attention and make em batch first
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if static_k is None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        else:
            k = static_k
        if static_v is None:
            v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        else:
            v = static_v

        # Add zero attention along batch dimension
        if add_zero_attn:
            zero_attn_shape = (bsz * num_heads, 1, head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        # Update source sequence length after adjustments
        src_len = k.size(1)

        # Merge key padding and attention masks
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float('-inf'))

        # Convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float('-inf'))
            attn_mask = new_attn_mask

        # Adjust dropout probability
        if not training:
            dropout_p = 0.0

        # Calculate attention and out projection
        # The AiA sclaed dot product attention is a funciton that
        attn_output, attn_output_weights = self.aia_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p,
                                                                                 pos_emb)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # Pass the output through a linear layer
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # Average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, pos_emb=None) -> Tuple[
        Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self.aia_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, pos_emb=pos_emb)
        else:
            attn_output, attn_output_weights = self.aia_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, pos_emb=pos_emb)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
