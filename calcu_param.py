from functools import reduce 

 
# 计算模型核的权重总参数量   
def summary(model):
    print("-"*73)
    line_new = "{:>20}  {:>25} {:>23}".format("layer kernel", "kernel Shape", "kernel Param")
    print(line_new)
    print("="*73)
    total_params = 0
    for name,param in model.named_parameters():
        params = reduce(lambda x,y: x*y, list(param.shape)) 
        total_params += params
        line_new = "{:>20}  {:>25} {:>15}".format(name, str(list(param.shape)), params)
        print(line_new)
    print("="*73)
    train_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: ',total_params)
    print('Trainable params: ',train_param)
    print('Non-trainable params: ',total_params-train_param)   
    # assume 4 bytes/number (float on cuda).
    total_params_size = abs(total_params * 4. / (1024 ** 2.))  
    print("Params size (MB): %f" % total_params_size)
    print("-"*73)