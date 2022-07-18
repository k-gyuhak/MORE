
best_args = \
{
    'cifar100':
    {
        'derpp':
        {
            'lr': 0.03,
            'alpha': 0.1,
            'beta': 1.0,
            'minibatch_size': 16,
        },
        'derpp_deit':
        {
            'lr': 0.03,
            'alpha': 0.1,
            'beta': 1.0,
            'minibatch_size': 16,
        },
        'joint':
        {
            'lr': 0.1,
        },
        'owm':
        {
            'lr': 0.5,
            'owm_alpha': [2.0],
        },
    },

    'cifar10':
    {
        'derpp':
        {
            'lr': 0.03,
            'alpha': 0.1,
            'beta': 1.0,
            'minibatch_size': 16,
        },
        'derpp_deit':
        {
            'lr': 0.03,
            'alpha': 0.1,
            'beta': 1.0,
            'minibatch_size': 16,
        },
    },
}