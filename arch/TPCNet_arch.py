import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from arch.Color_transform import *
# ==================================
# ======== Gaussian filter =========
# ==================================
def pair_downsampler(img):
    # img has shape B C H W
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = torch.nn.functional.conv2d(img, filter1, stride=2, groups=c)
    output2 = torch.nn.functional.conv2d(img, filter2, stride=2, groups=c)
    return output1,output2

def gaussian_basis_filters(scale, k=3):
    device = scale.device
    std = torch.pow(2,scale).to(device)
    k = torch.tensor(k).to(device)


    # Define the basis vector for the current scale
    filtersize = torch.ceil(k*std+0.5)
    x = torch.arange(start=-filtersize.item(), end=filtersize.item()+1).to(device)

    x = torch.meshgrid([x,x])

    # Calculate Gaussian filter base
    # Only exponent part of Gaussian function since it is normalized anyway
    g = torch.exp(-(x[0]/std)**2/2)*torch.exp(-(x[1]/std)**2/2)
    g = g / torch.sum(g)  # Normalize

    # Gaussian derivative dg/dx filter base
    dgdx = -x[0]/(std**3*2*math.pi)*torch.exp(-(x[0]/std)**2/2)*torch.exp(-(x[1]/std)**2/2)
    dgdx = dgdx / torch.sum(torch.abs(dgdx))  # Normalize

    # Gaussian derivative dg/dy filter base
    dgdy = -x[1]/(std**3*2*math.pi)*torch.exp(-(x[1]/std)**2/2)*torch.exp(-(x[0]/std)**2/2)
    dgdy = dgdy / torch.sum(torch.abs(dgdy))  # Normalize

    # Stack and expand dim
    basis_filter = torch.stack([g,dgdx,dgdy], dim=0)[:,None,:,:]

    return basis_filter


# =================================
# == Color invariant definitions ==
# =================================

eps = 1e-5

# =================================
# == Color invariant convolution ==
# =================================
#CIConv2d adapted from:https://github.com/Attila94/CIConv/blob/main/method/ciconv2d.py
#
class CIConv2d(nn.Module):
    def __init__(self,k=3, scale=0.0,learnability = True):

        super(CIConv2d, self).__init__()
        # Constants
        self.gcm = torch.tensor([[0.06,0.63,0.27],[0.3,0.04,-0.35],[0.34,-0.6,0.17]])
        self.k = k

        # Learnable parameters
        self.scale = torch.nn.Parameter(torch.tensor([scale]), requires_grad=learnability)

    def W_inv(self,E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
        Wx = Ex / (E + eps)
        Wlx = Elx / (E + eps)
        Wllx = Ellx / (E + eps)
        Wy = Ey / (E + eps)
        Wly = Ely / (E + eps)
        Wlly = Elly / (E + eps)

        W = Wx ** 2 + Wy ** 2 + Wlx ** 2 + Wly ** 2 + Wllx ** 2 + Wlly ** 2
        return W


    def forward(self, batch):
        device = batch.device
        # Make sure scale does not explode: clamp to max abs value of 2.5
        self.scale.data = torch.clamp(self.scale.data, min=-2.5, max=2.5).to(device)
        self.gcm = self.gcm.to(device)

        # Measure E, El, Ell by Gaussian color model
        in_shape = batch.shape  # bchw
        batch = batch.view((in_shape[:2]+(-1,)))  # flatten image
        batch = torch.matmul(self.gcm,batch)  # estimate E,El,Ell
        batch = batch.view((in_shape[0],)+(3,)+in_shape[2:])  # reshape to original image size
        E, El, Ell = torch.split(batch, 1, dim=1)
        # Convolve with Gaussian filters
        w = gaussian_basis_filters(scale=self.scale)  # KCHW

        # the padding here works as "same" for odd kernel sizes
        E_out = F.conv2d(input=E, weight=w, padding=int(w.shape[2]/2))
        El_out = F.conv2d(input=El, weight=w, padding=int(w.shape[2]/2))
        Ell_out = F.conv2d(input=Ell, weight=w, padding=int(w.shape[2]/2))

        E, Ex, Ey = torch.split(E_out,1,dim=1)
        El, Elx, Ely = torch.split(El_out,1,dim=1)
        Ell, Ellx, Elly = torch.split(Ell_out,1,dim=1)

        inv_out = self.W_inv(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly)
        inv_out = F.instance_norm(torch.log(inv_out+eps))

        return inv_out


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class NormDownsample(nn.Module):
    def __init__(self,in_ch,out_ch,scale=0.5,use_norm=False):
        super(NormDownsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))

    def forward(self, x):
        x = self.down(x)
        x = self.prelu(x)
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x


class NormUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2, use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
        self.up = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, y):
        x = self.up_scale(x)
        x = torch.cat([x, y], dim=1)
        x = self.up(x)
        x = self.prelu(x)
        if self.use_norm:
            return self.norm(x)
        else:
            return x
class CG_MSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CG_MSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qv_ = nn.Conv2d(2 *dim, 2*dim, kernel_size=1, bias=bias)
        self.qv__dwconv = nn.Conv2d(2*dim, 2*dim, kernel_size=3, stride=1, padding=1, groups=2*dim, bias=bias)
        self.kv = nn.Conv2d(2 * dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Sequential(nn.Conv2d(dim, 2*dim * 4, kernel_size=1, bias=bias),
                                         nn.PixelShuffle(2)
                                         )
        self.v_fuse = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.pos = nn.Conv2d(2 * dim,2 * dim,kernel_size=1,padding=0,bias=False)



    def forward(self, x, y):
        b, c, h, w = x.shape
        xy_pos = self.pos(torch.cat([x,y],dim=1))
        x1,x2 = pair_downsampler(x)
        y1,y2 = pair_downsampler(y)

        qv_ = self.qv__dwconv(self.qv_(torch.cat([x1,y2],dim=1)))
        kv = self.kv_dwconv(self.kv(torch.cat(([x2,y1]),dim=1)))
        q,v_ = qv_.chunk(2, dim=1)
        k, v = kv.chunk(2, dim=1)
        v = self.v_fuse(torch.cat([v,v_],dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h//2, w=w//2)

        out = self.project_out(out)+xy_pos
        return out
# Cross Attention Block
class CG_MSA_M(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CG_MSA_M, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qv_ = nn.Conv2d(dim, 2*dim, kernel_size=1, bias=bias)
        self.qv__dwconv = nn.Conv2d(2*dim, 2*dim, kernel_size=3, stride=1, padding=1, groups=2*dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, 2*dim, kernel_size=1, bias=bias)
        self.v_fuse = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.pos = nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, padding=0, bias=False)

    def forward(self, x, y):
        b, c, h, w = x.shape

        xy_pos = self.pos(torch.cat([x,y],dim=1))

        qv_ = self.qv__dwconv(self.qv_(x))
        kv = self.kv_dwconv(self.kv(y))
        q,v_ = qv_.chunk(2, dim=1)
        k, v = kv.chunk(2, dim=1)

        v = self.v_fuse(torch.cat([v,v_],dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)+xy_pos
        return out

class CG_MSA_V(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CG_MSA_V, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qv_ = nn.Conv2d(2*dim, 2*dim, kernel_size=1, bias=bias)
        self.qv__dwconv = nn.Conv2d(2*dim, 2*dim, kernel_size=3, stride=1, padding=1, groups=2*dim, bias=bias)
        self.kv = nn.Conv2d(2*dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Sequential(nn.Conv2d(dim, 1* dim * 4, kernel_size=1, bias=bias),
                                         nn.PixelShuffle(2)
                                         )
        self.v_fuse = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.pos = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)

    def forward(self, x, y):
        b, c, h, w = x.shape

        x_pos = self.pos(x)
        x1, x2 = pair_downsampler(x)
        y1, y2 = pair_downsampler(y)

        qv_ = self.qv__dwconv(self.qv_(torch.cat([x1, y2], dim=1)))
        kv = self.kv_dwconv(self.kv(torch.cat(([x2, y1]), dim=1)))
        q,v_ = qv_.chunk(2, dim=1)
        k, v = kv.chunk(2, dim=1)

        v = self.v_fuse(torch.cat([v,v_],dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h//2, w=w//2)

        out = self.project_out(out)+x_pos
        return out

class CG_MSA_V_M(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CG_MSA_V_M, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qv_ = nn.Conv2d(1*dim, 2*dim, kernel_size=1, bias=bias)
        self.qv__dwconv = nn.Conv2d(2*dim, 2*dim, kernel_size=3, stride=1, padding=1, groups=2*dim, bias=bias)
        self.kv = nn.Conv2d(1*dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Sequential(nn.Conv2d(dim, 1* dim * 1, kernel_size=1, bias=bias)

                                         )
        self.v_fuse = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.pos = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)

    def forward(self, x, y):
        b, c, h, w = x.shape

        x_pos = self.pos(x)

        qv_ = self.qv__dwconv(self.qv_(x))
        kv = self.kv_dwconv(self.kv(y))
        q,v_ = qv_.chunk(2, dim=1)
        k, v = kv.chunk(2, dim=1)

        v = self.v_fuse(torch.cat([v,v_],dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)+x_pos
        return out


class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x

class Light_feature_estimator(nn.Module):
    def __init__(
            self, n_fea_middle):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Light_feature_estimator, self).__init__()
        self.Con_embedding = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(4, n_fea_middle, 3, stride=1, padding=0,bias=False)
            )

        self.W_inv = CIConv2d()

        self.middle = nn.Conv2d(n_fea_middle,n_fea_middle,kernel_size=3,padding=1,bias=True)

        self.last = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(n_fea_middle, 1, 3, stride=1, padding=0,bias=False)
            )

        self.act = nn.Sigmoid()

    def forward(self, img):


        w_inv = self.W_inv(img)
        im_w = torch.cat([img,w_inv],dim=1)
        fea_con = self.Con_embedding(im_w)

        e_hat = self.middle(fea_con)
        alpha = self.act(self.last(e_hat))

        return e_hat,alpha

class Reflectivity_feature_estimator(nn.Module):
    def __init__(
            self, n_fea_middle):  #
        super(Reflectivity_feature_estimator, self).__init__()

        self.in_embedding = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, n_fea_middle, 3, stride=1, padding=0,bias=False)
            )


        self.middle = CGAB_V(dim=n_fea_middle,num_heads=1)

        self.act = nn.PReLU()

        self.last = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(n_fea_middle, n_fea_middle, 3, stride=1, padding=0,bias=False)
            )


    def forward(self, img,L_comp):

        E_hat = self.in_embedding(img)

        E_hat = self.middle(E_hat, E_hat)

        E_sub = E_hat - L_comp

        L_cj = self.last(E_sub)

        R_hat = self.act(E_sub * L_cj)

        return R_hat


class CGAB(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(CGAB, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim*2,ffn_expansion_factor=1, bias=False)
        self.attn = CG_MSA(dim, num_heads, bias=bias)

    def forward(self, x, y):
        xy = torch.cat([x,y],dim=1)
        xy = xy + self.attn(self.norm(x), self.norm(y))
        xy = xy + self.gdfn(xy)

        return xy
class CGAB_M(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(CGAB_M, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim*2,ffn_expansion_factor=1, bias=False)
        self.attn = CG_MSA_M(dim, num_heads, bias=bias)

    def forward(self, x, y):
        xy = torch.cat([x,y],dim=1)
        xy = xy + self.attn(self.norm(x), self.norm(y))
        xy = xy + self.gdfn(xy)

        return xy



class CGAB_V(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(CGAB_V, self).__init__()
        self.gdfn = IEL(dim)  # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.attn = CG_MSA_V(dim, num_heads, bias)

    def forward(self, x, y):
        x = x + self.attn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x

class CGAB_V_M(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(CGAB_V_M, self).__init__()
        self.gdfn = IEL(dim)  # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.attn = CG_MSA_V_M(dim, num_heads, bias)

    def forward(self, x, y):
        x = x + self.attn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x

class TPCNet(nn.Module):
    def __init__(self,out_channels=3,channels=[12,24,64,108], heads=[1, 2, 8,12],CAM_type='HVI'):  #__init__部分是内部属性，而forward的输入才是外部输入
        '''

        :param out_channels:
        :param channels:
        :param heads:
        :param CAM_type: Color-Association Mechanism support color space (HVI; YCBCR; HSV)
        '''
        super(TPCNet, self).__init__()
        self.level = len(channels)
        n_feat = channels[0]
        self.light_f = Light_feature_estimator(n_fea_middle=n_feat)
        self.ref_est = Reflectivity_feature_estimator(n_fea_middle=n_feat)
        self.re_encoding = nn.ModuleList([])
        self.trans = Color_transform(trans_type = CAM_type)
        self.CAM_type = CAM_type

        self.color_encoding = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, n_feat, 3, stride=1, padding=0,bias=False)
            )


        for i in range(self.level-1):
            self.re_encoding.append(
                nn.ModuleList([
                    CGAB(dim = channels[i],num_heads=heads[i]),
                    NormDownsample(in_ch=2 * channels[i],out_ch=2 * channels[i+1],use_norm=False),
                    CGAB_V(dim = channels[i],num_heads=heads[i]),
                    NormDownsample(in_ch=1 * channels[i], out_ch=1 * channels[i + 1], use_norm=False)
                ])
            )


        self.re_middle = CGAB_M(dim=channels[-1],num_heads=heads[-1])
        self.color_middle = CGAB_V_M(dim = channels[-1],num_heads=heads[-1])

        self.re_decoding = nn.ModuleList([])

        for i in range(self.level - 1, 0, -1):
            self.re_decoding.append(
                nn.ModuleList([
                    NormUpsample(in_ch=2 * channels[i], out_ch=2 * channels[i -1], use_norm=False),
                    CGAB(dim=channels[i-1], num_heads=heads[i-1]),
                    NormUpsample(in_ch=1 * channels[i], out_ch=1 * channels[i - 1], use_norm=False),
                    CGAB_V(dim=channels[i-1], num_heads=heads[i-1])

                ])
            )

        self.I_mapping = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(n_feat, 1, 3, stride=1, padding=0,bias=False),
            )
        self.C_mapping = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(n_feat, 2, 3, stride=1, padding=0,bias=False),
            )


    def forward(self, img):


        I_color = self.trans.color_transform(img)


        F_ce = self.color_encoding(I_color)



        e_hat,alpha = self.light_f(img)

        L_comp = e_hat * (1-alpha)

        L_ = e_hat* alpha


        R_hat= self.ref_est(img,L_comp/2)


        R_e_list = []

        F_ce_list = []

        e_hat = L_

        for (re_block,re_down,color_block,color_down) in self.re_encoding:
            # Dual-Stream Cross-Guided Transformer DCGT encoding
            Re = re_block(e_hat,R_hat)
            Re = re_down(Re)

            R_e_list.append(torch.cat([e_hat,R_hat],dim=1))

            F_ce = color_block(F_ce,e_hat * R_hat)# Color-Association Mechanism

            F_ce_list.append(F_ce)
            F_ce = color_down(F_ce)

            e_hat,R_hat = Re.chunk(2,dim = 1)





        Re = self.re_middle(e_hat,R_hat)
        F_ce = self.color_middle(F_ce, e_hat * R_hat)


        for i,(re_up,re_block,color_up,color_block) in enumerate(self.re_decoding):
            # Dual-Stream Cross-Guided Transformer DCGT decoding
            Re_J = R_e_list[self.level - 2 - i]
            hvi_J = F_ce_list[self.level - 2 - i]

            Re = re_up(Re,Re_J)
            F_ce = color_up(F_ce,hvi_J)
            R_hat_st,e_hat_st = Re.chunk(2,dim=1)
            Re = re_block(R_hat_st,e_hat_st)
            F_ce = color_block(F_ce,R_hat_st * e_hat_st) # Color-Association Mechanism

        e_hat_st,R_hat_st= Re.chunk(2,dim=1)

        E = R_hat_st * e_hat_st * alpha+e_hat_st * (1-alpha) / 2

        I_out = self.I_mapping(E)

        Y_ce = self.C_mapping(F_ce)

        if self.CAM_type == 'YCBCR':
            I_color_st = torch.cat([I_out, Y_ce], dim=1) + I_color
        elif self.CAM_type =='LAB':
            I_color_st = torch.cat([I_out, Y_ce], dim=1) + I_color
        else:
            I_color_st = torch.cat([Y_ce, I_out], dim=1) + I_color


        img_en = self.trans.color_inverse_transform(I_color_st)





        return img_en


    def Color_transform(self, x):
        hvi = self.trans.color_transform(x)
        return hvi



# if __name__ == '__main__':
#     from fvcore.nn import FlopCountAnalysis
#     # from thop import profile
#     model = TPCNet(out_channels=3,channels=[12,24,64,108], heads=[1, 2, 8,12],CAM_type='HVI')
#     print(model)
#     inputs = torch.randn((1, 3, 256, 256))
#     flops = FlopCountAnalysis(model,inputs)
#     # flops2, params2 = profile(model, inputs=(inputs,))
#     n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
#     print(f'GMac:{flops.total()/(1024*1024*1024)}')
#     # print(f'GMac2:{flops2/ (1024 * 1024 * 1024)}')
#     print(f'Params:{n_param}')
#     # print(f'Params2:{params2 / 1e6}')












