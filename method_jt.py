import os
import sys
import argparse
import warnings
from utils_jt import *
from utils_dataset_jt import *
from utils_train_inference_jt import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training and search specifications")

    parser.add_argument("--run", type=str, help="Input training file")
    parser.add_argument("--name", type=str, help="Name of the experiment folder")
    parser.add_argument("--seed", default=0, type=int, help="Choose a seed")
    parser.add_argument("--dataset", type=str, help="Choose the dataset")
    parser.add_argument("--train_only", nargs="+", type=str, help="Domains to train")
    parser.add_argument("--workers", default=6, type=int, help="Number of workers")
    parser.add_argument("--prefetch_factor", default=2, type=int, help="DataLoader prefetch factor")
    parser.add_argument("--pin_memory", action="store_true", help="Enable DataLoader pin_memory on CUDA")
    parser.add_argument("--persistent_workers", action="store_true", help="Keep worker processes alive between epochs")
    parser.add_argument("--gpu", default=0, type=int, help="Choose the GPU id")
    parser.add_argument("--backbone", default="resnet18", type=str, help="Backbone")
    parser.add_argument(
        "--from_scratch", action="store_false", dest="pretrained", help="No pre-train"
    )

    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--method_loss", type=float, help="Method loss")
    parser.add_argument("--epochs", default=300, type=int, help="Number of epochs")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
    parser.add_argument("--batch", default=64, type=int, help="Batch size")
    parser.add_argument(
        "--bt_exp_scheduler_gamma", default=0.01, type=float, help="Total LR decrease"
    )

    parser.add_argument(
        "--search_mode",
        choices=["resume", "new_test"],
        default="resume",
        type=str,
        help="What to do when the experiment already exists",
    )
    parser.add_argument(
        "--lr_search_no",
        default=33,
        type=int,
        help="Learning rate number of trainings in the search",
    )
    parser.add_argument(
        "--ml_search_no",
        default=17,
        type=int,
        help="Method loss number of trainings in the search",
    )
    parser.add_argument(
        "--lr_search_range",
        type=float,
        nargs=2,
        default=(1e-5, 1),
        help="Search range for lr tuning",
    )
    parser.add_argument(
        "--ml_search_range",
        type=float,
        nargs=2,
        default=(0, 1),
        help="Search range for method loss tuning",
    )
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (AMP)")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic behavior")
    parser.add_argument(
        "--fusion_weights",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90],
        help="Fusion weights w0 for eval-time geometric mean (w1=1-w0)",
    )
    parser.add_argument(
        "--val_only_metric",
        nargs="+",
        default=["average"],
        choices=["average", "worst", "cvar"],
        help="Metrics to summarize val_only groups",
    )
    parser.add_argument(
        "--val_only_cvar_k",
        type=int,
        default=2,
        help="Bottom-k groups for val_only CVaR",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    setup_device(gpu_id=args.gpu)
    set_all_seeds(seed=args.seed)
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    run = read_yaml(args.run)
    run["exec"] = f"python {' '.join(sys.argv)}"
    if args.dataset is not None:
        run["io_var"]["dataset"] = args.dataset
    if args.name is not None:
        run["io_var"]["run_name"] = args.name
    if "vit" in args.backbone.lower():
        vit_normalization(data=run)

    dataset_path = os.path.join(".", "data", run["io_var"]["dataset"])
    experiment_dir = os.path.join(
        "Results",
        f"{run['io_var']['dataset']}_{args.backbone}",
        run["io_var"]["run_name"],
    )
    try_make_dir(directory=experiment_dir)
    save_yaml(structure=run, direct=os.path.join(experiment_dir, "run.yaml"))

    domains = sorted(
        os.listdir(
            os.path.join(
                dataset_path,
                f"{run['io_var']['dataset']}_{run['pseudo_domains'][0]['dir'][0]}",
            )
        )
    )
    train_domains_only = args.train_only if args.train_only else domains
    domain_idx = list(range(len(domains)))
    current_dom_idx = [x for x in domain_idx]
    rest_dom_idx = [
        [x for x in domain_idx if x != current_idx] for current_idx in domain_idx
    ]

    for idx in list(range(len(domains))):
        train_idx, test_idx_list = current_dom_idx[idx], rest_dom_idx[idx]
        print(
            f"Training: {domains[train_idx].lower()} "
            f"Testing: {', '.join(domains[x].lower() for x in test_idx_list)}"
        )

        if domains[train_idx].lower() not in [x.lower() for x in train_domains_only]:
            print(
                f"{domains[train_idx].lower()} not in the specified train domains. Skipping..."
            )
            continue

        loader_info = set_dataloaders(
            args=args,
            run=run,
            dataset_path=dataset_path,
            train_domain_idx=train_idx,
            test_domain_idx=test_idx_list,
            domains=domains,
        )

        save_path = os.path.join(experiment_dir, domains[train_idx], f"Seed_{args.seed}")
        try_make_dir(directory=save_path)
        csv_file_name = os.path.join(
            experiment_dir,
            f"Results_source_{domains[train_idx].lower()}_seed_{args.seed}.csv",
        )
        # If running in new_test mode, rewrite CSV instead of appending
        if args.search_mode.lower() == "new_test" and os.path.isfile(csv_file_name):
            try:
                os.remove(csv_file_name)
            except Exception:
                pass
        initialize_csv_file(
            loader_info=loader_info,
            csv_file_name=csv_file_name,
            test_idx_list=test_idx_list,
            domains=domains,
            val_only_metric=args.val_only_metric,
        )

        if args.lr is not None and args.method_loss is not None:
            model = training_function(
                args=args,
                loader_info=loader_info,
                lr=args.lr,
                method_loss=args.method_loss,
                save_path=save_path,
                experiment_dir=experiment_dir,
            )
            testing_function(
                model=model,
                loader_info=loader_info,
                test_idx_list=test_idx_list,
                lr=args.lr,
                method_loss=args.method_loss,
                csv_file_name=csv_file_name,
                domains=domains,
                args=args,
            )
        else:
            search_hyperparameters(
                args=args,
                loader_info=loader_info,
                test_idx_list=test_idx_list,
                csv_file_name=csv_file_name,
                domains=domains,
                save_path=save_path,
                experiment_dir=experiment_dir,
            )


if __name__ == "__main__":
    main()
