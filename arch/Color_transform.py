# This file includes partial code adapted from:
# https://github.com/Fediory/HVI-CIDNet/blob/master/net/HVI_transform.py
# License: MIT License
import torch
import torch.nn as nn
import kornia
pi = 3.141592653589793
class Color_transform(nn.Module):
    def __init__(self,trans_type = 'HVI'):
        super(Color_transform, self).__init__()
        self._gated2 = False
        self._alpha = 1.0
        self.trans_type = trans_type

        if trans_type == 'HSV':
            self.color_transform = kornia.color.rgb_to_hsv
            self.color_inverse_transform = self.hsv_to_rgb

        elif trans_type == 'YCBCR':
            self.color_transform = kornia.color.rgb_to_ycbcr
            self.color_inverse_transform = self.ycbcr_to_rgb
        elif trans_type == 'LAB':
            self.color_transform = kornia.color.rgb_to_lab
            self.color_inverse_transform = self.lab_to_rgb

        elif trans_type =='HVI':
            self.trans_ini = RGB_HVI()
            self.trans_ini.gated2=self.gated2
            self.trans_ini.alpha=self.alpha
            self.color_transform = self.trans_ini.HVIT
            self.color_inverse_transform = self.trans_ini.PHVIT
        else:
            raise ValueError(f"Unsupported color space: '{trans_type}'. Supported options are: 'HSV', 'Ycbcr','HVI'")

    # ---------- property: gated2 ----------
    @property
    def gated2(self):
        return self._gated2

    @gated2.setter
    def gated2(self, value):
        self._gated2 = value
        if self.trans_type == 'HVI':
            self.trans_ini.gated2 = value

    # ---------- property: alpha ----------
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        if self.trans_type == 'HVI':
            self.trans_ini.alpha = value

    def hsv_to_rgb(self,img):
        if self.gated2:
            return self.alpha * kornia.color.hsv_to_rgb(img)
        return kornia.color.hsv_to_rgb(img)

    def ycbcr_to_rgb(self,img):
        if self.gated2:
            return self.alpha * kornia.color.ycbcr_to_rgb(img)
        return kornia.color.ycbcr_to_rgb(img)

    def lab_to_rgb(self,img):
        if self.gated2:
            return self.alpha * kornia.color.lab_to_rgb(img)
        return kornia.color.lab_to_rgb(img)

class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1],0.2)) # k is reciprocal to the paper mentioned
        self.gated = False
        self.gated2= False
        self.alpha = 1.0
        self.alpha_s = 1.3
        self.this_k = 0
        
    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:,2]==value] = 4.0 + ( (img[:,0]-img[:,1]) / (value - img_min + eps)) [img[:,2]==value]
        hue[img[:,1]==value] = 2.0 + ( (img[:,2]-img[:,0]) / (value - img_min + eps)) [img[:,1]==value]
        hue[img[:,0]==value] = (0.0 + ((img[:,1]-img[:,2]) / (value - img_min + eps)) [img[:,0]==value]) % 6

        hue[img.min(1)[0]==value] = 0.0
        hue = hue/6.0

        saturation = (value - img_min ) / (value + eps )
        saturation[value==0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        
        k = self.density_k
        self.this_k = k.item()
        
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        xyz = torch.cat([H, V, I],dim=1)
        return xyz
    
    def PHVIT(self, img):
        eps = 1e-8
        H,V,I = img[:,0,:,:],img[:,1,:,:],img[:,2,:,:]
        
        # clip
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        I = torch.clamp(I,0,1)
        
        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        h = torch.atan2(V + eps,H + eps) / (2*pi)
        h = h%1
        s = torch.sqrt(H**2 + V**2 + eps)
        
        if self.gated:
            s = s * self.alpha_s
        
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
                
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb
