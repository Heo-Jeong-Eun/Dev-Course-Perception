# parse the YOLOv3 configuration
def parse_hyperparam_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    # 주석 처리 된 부분은 읽어오지 않는다. 
    lines = [x for x in lines if x and not x.startswith('#')]
    # 공백 부분은 읽어오지 않는다. 
    lines = [x.rstrip().lstrip() for x in lines]

    module_defs = []
    for line in lines:
        if line.startswith('['):
            type_name = line[1:-1].rstrip()
            
            if type_name != 'net':
                continue  
            
            module_defs.append({})
            module_defs[-1]['type'] = type_name            
            
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            if type_name != 'net':
                continue
    
            key, value = lines.split('=')
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    
    return module_defs

def get_hyperparam(data):
    for d in data:
        if d['type'] == 'net':
            batch = int(d['batch'])
            subdivision = int(d['subdivision'])
            momentum = float(d['momentum'])
            decay = float(d['decay'])
            saturation = float(d['saturation'])
            lr = float(d['learning rate'])
            burn_in = float(d['burn in'])
            max_batch = int(d['max batch'])
            lr_policy = d['policy']
            in_width = int(d['width'])
            in_height = int(d['height'])
            in_channels = int(d['channel'])
            classes = int(d['class'])
            ignore_class = int(d['ignore cls'])

            return {'batch' : batch,
                    'subdivision' : subdivision,
                    'momentum' : momentum,
                    'decay' : decay,
                    'saturation' : saturation,
                    'lr' : lr,
                    'burn_in' : burn_in,
                    'max_batch' : max_batch,
                    'lr_policy' : lr_policy,
                    'in_width' : in_width,
                    'in_height' : in_height,
                    'in_channels' : in_channels,
                    'classes' : classes,
                    'ignore_class' : ignore_class}
        else:
            continue