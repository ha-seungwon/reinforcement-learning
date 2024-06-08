def set_parameters(args):
    if args.model_name == 'MC':
        if args.env_name == 'cliffwalking':
            args.alpha = 0.01
            args.gamma = 0.99
            args.glie = True
            args.glie_update_time=10
            args.epsilon = 0.4 if not args.glie else 1


        elif args.env_name == 'frozenlake':
            args.alpha = 0.01
            args.gamma = 0.99
            args.glie = True
            args.glie_update_time = 100
            args.epsilon = 0.4 if not args.glie else 1

        elif args.env_name == 'taxi':
            args.alpha = 0.1 # 0.00001 glie
            args.gamma = 0.99
            args.glie = True #it is not okay to do false in taxi
            args.glie_update_time = 100
            args.epsilon = 0.1 if not args.glie else 1


    elif args.model_name == 'TD':
        if args.env_name == 'cliffwalking':
            args.alpha = 0.005
            args.gamma = 0.99
            args.epsilon = 0.4


        elif args.env_name == 'frozenlake':
            args.alpha = 0.005
            args.gamma = 0.99
            args.epsilon = 0.4

        elif args.env_name == 'taxi':
            args.alpha = 0.1
            args.gamma = 0.99
            args.epsilon = 0.1

    elif args.model_name == 'SARSA':
        if args.env_name == 'cliffwalking':
            args.alpha = 0.1
            args.gamma = 0.99
            args.epsilon = 0.2


        elif args.env_name == 'frozenlake':
            args.alpha = 0.1
            args.gamma = 0.99
            args.epsilon = 0.1

        elif args.env_name == 'taxi':
            args.alpha = 0.1
            args.gamma = 0.99
            args.epsilon = 0.1


    elif args.model_name == 'Q_learning':
        if args.env_name == 'cliffwalking':
            args.alpha = 0.1
            args.gamma = 0.99
            args.epsilon = 0.1


        elif args.env_name == 'frozenlake':
            args.alpha = 0.1
            args.gamma = 0.99
            args.epsilon = 0.2

        elif args.env_name == 'taxi':
            args.alpha = 0.1
            args.gamma = 0.99
            args.epsilon = 0.1


    elif args.model_name == 'DQN':
        if args.env_name == 'cliffwalking':
            args.alpha = 0.01
            args.gamma = 0.92
            args.epsilon = 0.99


        elif args.env_name == 'frozenlake':
            args.alpha = 0.001
            args.gamma = 0.92
            args.epsilon = 0.2

        elif args.env_name == 'taxi':
            args.alpha = 0.001
            args.gamma = 0.99
            args.epsilon = 0.1



    return args