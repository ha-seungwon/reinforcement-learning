def set_parameters(args):
    if args.model_name == 'MC':
        if args.env_name == 'cliffwalking':
            args.alpha = 0.01
            args.gamma = 0.99
            args.glie = False
            args.glie_update_time=10
            args.epsilon = 0.1 if not args.glie else 1

            args.num_episodes = 5000


        elif args.env_name == 'frozenlake':
            args.alpha = 0.01
            args.gamma = 0.99
            args.glie = False
            args.glie_update_time = 1
            args.epsilon = 0.1 if not args.glie else 1

            args.num_episodes = 3000
        elif args.env_name == 'taxi':
            args.alpha = 0.1 # 0.00001 glie
            args.gamma = 0.99
            args.glie = False #it is not okay to do false in taxi
            args.glie_update_time = 10
            args.epsilon = 0.1 if not args.glie else 1

            args.num_episodes = 5000


    elif args.model_name == 'TD':
        if args.env_name == 'cliffwalking':
            args.alpha = 0.01
            args.gamma = 0.99
            args.glie = False
            args.glie_update_time=10
            args.epsilon = 0.1 if not args.glie else 1

            args.num_episodes = 3000


        elif args.env_name == 'frozenlake':
            args.alpha = 0.01
            args.gamma = 0.99
            args.glie = False
            args.glie_update_time=10
            args.epsilon = 0.1 if not args.glie else 1

            args.num_episodes = 3000

        elif args.env_name == 'taxi':
            args.alpha = 0.01
            args.gamma = 0.99
            args.glie = False
            args.glie_update_time=10
            args.epsilon = 0.4 if not args.glie else 1

            args.num_episodes = 8000

    elif args.model_name == 'SARSA':
        if args.env_name == 'cliffwalking':
            args.alpha = 0.1
            args.gamma = 0.99
            args.epsilon = 0.1

            args.num_episodes = 1500


        elif args.env_name == 'frozenlake':
            args.alpha = 0.1
            args.gamma = 0.99
            args.epsilon = 0.1

            args.num_episodes = 3000

        elif args.env_name == 'taxi':
            args.alpha = 0.1
            args.gamma = 0.99
            args.epsilon = 0.4

            args.num_episodes = 5000


    elif args.model_name == 'Q_learning':
        if args.env_name == 'cliffwalking':
            args.alpha = 0.1
            args.gamma = 0.99
            args.epsilon = 0.1

            args.num_episodes = 1500


        elif args.env_name == 'frozenlake':
            args.alpha = 0.1
            args.gamma = 0.99
            args.epsilon = 0.1
            args.num_episodes = 3000

        elif args.env_name == 'taxi':
            args.alpha = 0.1
            args.gamma = 0.99
            args.epsilon = 0.4

            args.num_episodes = 5000


    elif args.model_name == 'DQN':
        if args.env_name == 'cliffwalking':
            args.alpha = 0.001
            args.gamma = 0.92
            args.epsilon = 0.99
            args.threshold = 40
            args.batch_size = 32
            args.hidden_dim = 32
            args.num_episodes =800


        elif args.env_name == 'frozenlake':
            args.alpha = 0.001
            args.gamma = 0.92
            args.epsilon = 0.99
            args.threshold = 40
            args.batch_size = 32
            args.hidden_dim = 32
            args.num_episodes = 800

        elif args.env_name == 'taxi':
            args.alpha = 0.001
            args.gamma = 0.92
            args.epsilon = 0.99
            args.threshold = 40
            args.batch_size = 32
            args.hidden_dim = 64
            args.num_episodes = 800




    return args