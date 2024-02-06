import torch
import torch.nn.functional as F

def get_lr(optimizer):
    """
    Get the learning rate at any given point from the optimizer.

    Parameters:
    - optimizer: The PyTorch optimizer.

    Returns:
    - lr (float): The learning rate of the optimizer at the current point.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_optim_sched(model, args):
    """
    Get the optimizer and schedulers based on the specified arguments.

    Parameters:
    - model: The PyTorch model for which the optimizer and schedulers are obtained.
    - args (dict): Dictionary containing model arguments, including optimizer and scheduler settings.
                It should have keys 'model_args', 'optimizer', 'lr', 'momentum', 'weight_decay', 'amsgrad',
                'scheduler', 'gamma', 'lr_decay_steps'.

    Returns:
    - optimizer: The PyTorch optimizer based on the specified arguments.
    - scheduler: The PyTorch learning rate scheduler based on the specified arguments.
    - scheduler2: Additional learning rate scheduler (ReduceLROnPlateau) with fixed parameters.

    Note:
    - The function creates the optimizer and schedulers based on the specified settings in the 'args' dictionary.
    - If no scheduler is specified, a default ExponentialLR scheduler with gamma=1 is used = no scheduler 
    """
    if args['model_args']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args['model_args']['lr'], momentum=args['model_args']['momentum'])
    elif args['model_args']['optimizer'] == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args['model_args']['lr'], weight_decay=args['model_args']['weight_decay'], amsgrad=args['model_args']['amsgrad'])

    if args['model_args']['scheduler'] == 'exponentiallr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args['model_args']['gamma'])  # fixed gamma
    elif args['model_args']['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['model_args']['lr_decay_steps'], gamma=args['model_args']['gamma'])
    elif args['model_args']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)  # fixed cycle
    elif args['model_args']['scheduler'] == 'plat':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)  # fixed patience
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)  # this should be equivalent to having no scheduler

    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)  # fixed gamma

    return optimizer, scheduler, scheduler2
