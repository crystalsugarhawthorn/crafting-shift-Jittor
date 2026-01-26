import jittor as jt
import jittor.nn as nn
import jittor.optim as optim
import models_jt as my_models
from utils_jt import *
import os
import time
import numpy as np
import contextlib

try:
    import jittor.amp as amp
except Exception:  # pragma: no cover - amp may be unavailable
    amp = None


def _unwrap_singleton(x):
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]
    return x


def _to_var(x, dtype="float32"):
    x = _unwrap_singleton(x)
    if isinstance(x, jt.Var):
        return x.int32() if dtype == "int32" else x.float32()
    if isinstance(x, (list, tuple)):
        # If elements are Vars, concat them; otherwise cast to numpy then to Var
        if all(isinstance(v, jt.Var) for v in x):
            x = [v.reshape((-1,)) for v in x]
            x = jt.concat(x, dim=0)
            return x.int32() if dtype == "int32" else x.float32()
        x = np.array(x)
    if dtype == "int32":
        return jt.array(x).int32()
    return jt.array(x).float32()


def _ensure_label_var(y):
    # Normalize labels to a single Var
    if isinstance(y, (list, tuple)):
        if len(y) == 1:
            y = y[0]
        elif all(isinstance(v, jt.Var) for v in y):
            y = jt.concat([v.reshape((-1,)) for v in y], dim=0)
        else:
            y = np.array(y)
    return _to_var(y, dtype="int32")


def inference(model, loader):
    model.eval()
    acc_list = None
    set_size = len(loader.dataset) if hasattr(loader, "dataset") else None
    seen = 0  # fallback if loader lacks dataset length

    with jt.no_grad():
        for input_image, y, path in loader:
            y = _ensure_label_var(y)
            input_image = [_to_var(x, dtype="float32") for x in input_image]
            y_var = y if isinstance(y, jt.Var) else jt.array(np.array(y)).int32()
            if jt.flags.use_cuda:
                y_var = y_var.cuda()
                input_image = [img.cuda() for img in input_image]

            outputs = model(input_image)

            if acc_list is None:
                acc_list = [0] * len(outputs)

            for idx, output in enumerate(outputs):
                preds = jt.argmax(output, dim=1)
                preds_var = preds if isinstance(preds, jt.Var) else jt.array(preds)
                if jt.flags.use_cuda:
                    preds_var = preds_var.cuda()
                acc_list[idx] += int(((preds_var == y_var).sum()).item())
            # Align with Torch: count by batch dimension (y_var.shape[0])
            seen += int(y_var.shape[0])

    denom = set_size if set_size is not None else seen
    denom = max(denom, 1)
    acc_list = [round(acc / denom, 4) for acc in acc_list]
    return acc_list


def training_function(
    args, loader_info, lr, method_loss, save_path, experiment_dir
):
    model_name_path = os.path.join(
        ".", save_path, f"Method_loss_{method_loss}_lr_{lr}.pkl"
    )

    model = my_models.PseudoCombiner(
        no_classes=len(loader_info["classes"]),
        pretrained=args.pretrained,
        backbone_name=args.backbone,
    )
    model.train()

    save_architecture(network=model, direct=experiment_dir)

    if args.search_mode.lower() == "new_test" and os.path.isfile(model_name_path):
        load_model(network=model, model_location=model_name_path)
        return model
    elif os.path.isfile(model_name_path):
        print(f"{model_name_path} exists.")
        return

    # Jittor CrossEntropyLoss lacks reduction arg; keep default (mean) and scale by batch to mimic sum
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    gamma = pow(args.bt_exp_scheduler_gamma, (1.0 / float(args.epochs)))
    epochs = args.epochs

    use_amp = bool(getattr(args, "amp", False) and jt.flags.use_cuda and amp is not None)
    amp_ctx = amp.autocast() if use_amp else contextlib.nullcontext()

    for epoch in range(epochs):
        start = time.time()
        train_acc, train_count = 0, 0

        # Match Torch ExponentialLR: use current optimizer.lr during epoch, then step at end

        for input_image, y, path in loader_info["train_loader"]:
            y = _ensure_label_var(y)
            if isinstance(y, (list, tuple)):
                y = jt.array(np.array(y)).int32()
            input_image = [_to_var(x, dtype="float32") for x in input_image]
            y_var = y if isinstance(y, jt.Var) else jt.array(np.array(y)).int32()
            if jt.flags.use_cuda:
                y_var = y_var.cuda()
                input_image = [img.cuda() for img in input_image]

            with amp_ctx:
                outputs = model(input_image)
            batches = [x.shape[0] for x in outputs]

            loss = jt.zeros(1).float32()[0]
            for idx2, output in enumerate(outputs):
                current_loss = criterion(output, y_var) * output.shape[0]
                loss += current_loss if idx2 == 0 else current_loss * method_loss
            loss = loss / sum(batches)

            # Accuracy: align with Torch version — only use outputs[0]
            preds = jt.argmax(outputs[0], dim=1)
            preds_var = preds if isinstance(preds, jt.Var) else jt.array(preds)
            if jt.flags.use_cuda:
                preds_var = preds_var.cuda()
            batch_correct = int(((preds_var == y_var).sum()).item())
            batch_size = int(y_var.shape[0])
            if batch_correct > batch_size:
                print(f"Warning: batch_correct ({batch_correct}) > batch_size ({batch_size}). Clamping.")
                print(f"preds shape: {preds_var.shape}, y shape: {y_var.shape}")
                batch_correct = min(batch_correct, batch_size)
            train_acc += batch_correct
            train_count += batch_size

            optimizer.step(loss)

            if nan_in_grad(model=model):
                save_model(
                    network=model,
                    optimizer=optimizer,
                    epoch=args.epochs,
                    direct=model_name_path,
                    is_nan=True,
                )
                return model

        # ExponentialLR step: decay lr for the next epoch (match Torch behavior)
        optimizer.lr *= gamma

        print(
            f"Epoch: {epoch} "
            f"LR: {round(optimizer.lr, 8)} "
            f"Acc: {round(train_acc / train_count, 4)} "
            f"Time: {report_time(time.time() - start)}"
        )

    save_model(
        network=model,
        optimizer=optimizer,
        epoch=args.epochs,
        direct=model_name_path,
    )
    return model


def testing_function(
    model, loader_info, test_idx_list, lr, method_loss, csv_file_name, domains
):
    if model is not None:
        val_only_acc, test_acc, imgaug_average = ([] for _ in range(3))
        print("Validating Normal")
        val_list = inference(model=model, loader=loader_info["val_loader"])
        if loader_info["val_only_loader"] is not None:
            for idx, vo_loader in enumerate(loader_info["val_only_loader"]):
                print(
                    f"Validating {loader_info['pd_names_val_only'][idx].replace('_', ' ')}"
                )
                val_only_acc.append(inference(model=model, loader=vo_loader))
            imgaug_average = [
                round(sum(col) / len(col), 4) for col in zip(*val_only_acc)
            ]
            val_only_acc = [item for sublist in val_only_acc for item in sublist]
        for idx2, test_idx in enumerate(test_idx_list):
            test_acc.append(
                inference(model=model, loader=loader_info["test_loader_list"][idx2])
            )
            for idx, output in enumerate(loader_info["output_names_val"]):
                print(
                    f"Test {domains[test_idx]} domain, {output} Acc: {test_acc[-1][idx]}"
                )

        total_test = [round(sum(col) / len(col), 4) for col in zip(*test_acc)]
        test_acc_list_out = [item for sublist in test_acc for item in sublist]
        add_free_log(
            data=[[str(lr)]]
            + [[str(method_loss)]]
            + [[str(x)] for x in val_list]
            + [[str(x)] for x in imgaug_average]
            + [[str(x)] for x in total_test]
            + [[str(x)] for x in val_only_acc]
            + [[str(x)] for x in test_acc_list_out],
            save_dir=csv_file_name,
        )
        for idx, output in enumerate(loader_info["output_names_val"]):
            print(f"Total Test {output} Acc: {total_test[idx]}")
    return


def search_hyperparameters(
    args,
    loader_info,
    test_idx_list,
    csv_file_name,
    domains,
    save_path,
    experiment_dir,
):
    lr_min, lr_max = args.lr_search_range
    ml_min, ml_max = args.ml_search_range
    if lr_max < lr_min:
        lr_max, lr_min = lr_min, lr_max
    if ml_max < ml_min:
        ml_max, ml_min = ml_min, ml_max
    if args.lr is None and args.method_loss is None:
        lr_search_range = generate_points(
            range_tuple=(lr_min, lr_max),
            points=args.lr_search_no,
            log_scale=True,
        )
        ml_search_range = generate_points(
            range_tuple=(ml_min, ml_max), points=args.ml_search_no
        )
        print("Searching for the best Learning Rate and the best Method Loss...")
        print(
            f"We will train a total of {len(lr_search_range)*len(ml_search_range)} models"
        )
        for lr in lr_search_range:
            for method_loss in ml_search_range:
                print(f"Trying Learning Rate: {lr} and Method Loss: {method_loss}")
                model = training_function(
                    args,
                    loader_info,
                    lr,
                    method_loss,
                    save_path,
                    experiment_dir,
                )
                testing_function(
                    model,
                    loader_info,
                    test_idx_list,
                    lr,
                    method_loss,
                    csv_file_name,
                    domains,
                )
    elif args.lr is None:
        lr_search_range = generate_points(
            range_tuple=(lr_min, lr_max),
            points=args.lr_search_no,
            log_scale=True,
        )
        print("Searching for the best Learning Rate...")
        print(f"We will train a total of {len(lr_search_range)} models")
        for lr in lr_search_range:
            print(f"Trying Learning Rate: {lr}")
            model = training_function(
                args,
                loader_info,
                lr,
                args.method_loss,
                save_path,
                experiment_dir,
            )
            testing_function(
                model,
                loader_info,
                test_idx_list,
                lr,
                args.method_loss,
                csv_file_name,
                domains,
            )
    elif args.method_loss is None:
        ml_search_range = generate_points(
            range_tuple=(ml_min, ml_max), points=args.ml_search_no
        )
        print("Searching for the best Method Loss...")
        print(f"We will train a total of {len(ml_search_range)} models")
        for method_loss in ml_search_range:
            print(f"Trying Method Loss: {method_loss}")
            model = training_function(
                args,
                loader_info,
                args.lr,
                method_loss,
                save_path,
                experiment_dir,
            )
            testing_function(
                model,
                loader_info,
                test_idx_list,
                args.lr,
                method_loss,
                csv_file_name,
                domains,
            )
