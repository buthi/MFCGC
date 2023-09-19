import argparse

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs1", type=int, default=500
    )
    parser.add_argument(
        "--epochs2", type=int, default=500
    )
    parser.add_argument(
        "--epochs3", type=int, default=500
    )
    parser.add_argument("--pt_lr", type=float, default=0.0003)
    parser.add_argument("--pt_wd", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--ft_lr", type=float, default=0.0001)
    parser.add_argument("--ft_wd", type=float, default=1e-5)
    parser.add_argument("--tau", type=float, default=1)

    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="Hidden layer dim."
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of GNN layers."
    )
    parser.add_argument(
        "--pe",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--pf1",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--pf2",
        type=float,
        default=0.5,
    )
    parser.add_argument("--normalize", type=bool, default=True)

    # args = parser.parse_args()
    # print(args)
    return parser

def Cora_config(parser):
    args = parser.parse_args()
    print(args)
    return args


def Citeseer_config(parser):
    args = parser.parse_args()
    print(args)
    return args

def ACM_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="ACM")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument(
        "--epochs1", type=int, default=500
    )
    parser.add_argument(
        "--epochs2", type=int, default=500
    )
    parser.add_argument(
        "--epochs3", type=int, default=500
    )
    parser.add_argument("--pt_lr", type=float, default=0.0005)
    parser.add_argument("--pt_wd", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--ft_lr", type=float, default=0.0005)
    parser.add_argument("--ft_wd", type=float, default=1e-5)
    parser.add_argument("--tau", type=float, default=0.2)

    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="Hidden layer dim."
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of GNN layers."
    )
    parser.add_argument(
        "--pe",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--pf1",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--pf2",
        type=float,
        default=0.3,
    )

    args = parser.parse_args()
    print(args)
    return args


def IMDB_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="IMDB")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument(
        "--epochs1", type=int, default=0
    )
    parser.add_argument(
        "--epochs2", type=int, default=500
    )
    parser.add_argument(
        "--epochs3", type=int, default=500
    )
    parser.add_argument("--pt_lr", type=float, default=0.0005)
    parser.add_argument("--pt_wd", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--ft_lr", type=float, default=0.0005)
    parser.add_argument("--ft_wd", type=float, default=1e-5)
    parser.add_argument("--tau", type=float, default=1)

    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="Hidden layer dim."
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of GNN layers."
    )
    parser.add_argument(
        "--pe",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--pf1",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--pf2",
        type=float,
        default=0.3,
    )

    args = parser.parse_args()
    print(args)
    return args


def DBLP_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="DBLP")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument(
        "--epochs1", type=int, default=0
    )
    parser.add_argument(
        "--epochs2", type=int, default=0
    )
    parser.add_argument(
        "--epochs3", type=int, default=100
    )
    parser.add_argument("--pt_lr", type=float, default=0.001)
    parser.add_argument("--pt_wd", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--ft_lr", type=float, default=0.0001)
    parser.add_argument("--ft_wd", type=float, default=1e-5)
    parser.add_argument("--tau", type=float, default=1.0)

    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden layer dim."
    )

    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of GNN layers."
    )
    parser.add_argument(
        "--pe",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--pf1",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--pf2",
        type=float,
        default=0,
    )

    args = parser.parse_args()
    print(args)
    return args