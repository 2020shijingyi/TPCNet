from arch.TPCNet_arch import TPCNet

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    import torch._dynamo
    torch._dynamo.config.disable = True
    import torch
    #The model complexity reported in our manuscript is calculated by fvcore Library
    model = TPCNet(out_channels=3, channels=[12, 24, 64, 108], heads=[1, 2, 8, 12], CAM_type='HVI')
    print(model)
    inputs = torch.randn((1, 3, 256, 256))
    flops = FlopCountAnalysis(model, inputs)
    n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
    print(f'FLOPs(G):{flops.total() / (1024 * 1024 * 1024)}')
    print(f'Params(M):{n_param/1e6}')


    # The model complexity calculated by thop Library is similar to the results reported by us
    # from thop import profile
    # flops2, params2 = profile(model, inputs=(inputs,))
    # print(f'GMac2:{flops2 / (1024 * 1024 * 1024)}')
    # print(f'Params2:{params2 / 1e6}')