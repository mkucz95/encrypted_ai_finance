import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import argparse
import sys
import asyncio
import numpy as np

from config import define_and_get_arguments, n_workers, base_port

FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(format=FORMAT)

import syft as sy
from syft import workers
from syft.frameworks.torch.federated import utils

sys.path.append('../')
from helpers import Model #previously created model

logger = logging.getLogger(__name__)
LOG_INTERVAL = 25

# Loss function
#read about this here: https://pytorch.org/docs/stable/jit.html#mixing-tracing-and-scripting
@torch.jit.script
def loss_fn(pred, target):
    return F.nll_loss(input=pred, target=target)

async def fit_model_on_worker(
    worker: workers.WebsocketClientWorker,
    traced_model: torch.jit.ScriptModule,
    batch_size: int,
    curr_round: int,
    max_nr_batches: int,
    lr: float,
):
    """Coroutine function that executes and awaits a return model from the async_fit() call
       Send the model to the worker and fit the model on the worker's training data.

    Args:
        worker: Remote location, where the model shall be trained.
        traced_model: Model which shall be trained.
        batch_size: Batch size of each training step.
        curr_round: Index of the current training round (for logging purposes).
        max_nr_batches: If > 0, training on worker will stop at min(max_nr_batches, nr_available_batches).
        lr: Learning rate of each training step.

    Returns:
        A tuple containing:
            * worker_id: Union[int, str], id of the worker.
            * improved model: torch.jit.ScriptModule, model after training at the worker.
            * loss: Loss on last training batch, torch.tensor.
    """
    train_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=True,
        max_nr_batches=max_nr_batches,
        epochs=1,
        optimizer="SGD",
        optimizer_args={"lr": lr},
    )
    train_config.send(worker)
    logger.info("Training round %s, calling fit on worker: %s", r, worker.id)
    loss = await worker.async_fit(dataset_key="credit", return_ids=[0])
    logger.info("Training round: %s, worker: %s, avg_loss: %s", r, worker.id, loss.mean())
    model = train_config.model_ptr.get().obj
    return worker.id, model, loss


def evaluate_model_on_worker(model_identifier, worker,
                             dataset_key: str, model: Model, nr_bins,
                             batch_size, print_target_hist=False):
    """
    Evaluate model on testing set. The testing set is
    located on the remote server worker called `testing` -- the trusted
    third party aggregartor.
    
    Loss and accuracy over the test set are returned.
    """
    model.eval()

    # Create and send train config
    train_config = sy.TrainConfig(
        batch_size=batch_size, model=model,
        loss_fn=loss_fn, optimizer_args=None,
        epochs=1
    ) #evaluate full model on 'testing' remote server worker (trusted third party)

    train_config.send(worker)

    #evaluate aggregate model on validation set
    result = worker.evaluate(
        dataset_key=dataset_key,
        nr_bins=nr_bins,
        return_loss=True,
        return_raw_accuracy=True,
    )
    
    test_loss = result["loss"]
    correct = result["nr_correct_predictions"]
    len_dataset = result["nr_predictions"]

    logger.info(
        "%s: Test set: Average loss: %s, Accuracy: %s/%s (%s)",
        model_identifier,
        "{:.4f}".format(test_loss),
        correct,
        len_dataset,
        "{:.2f}".format(100.0 * correct / len_dataset),
    )


async def main():
    # GENERAL SETUP
    args = define_and_get_arguments()
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    hook = sy.TorchHook(torch)
    kwargs_websocket = {"hook": hook, "verbose": args.verbose, "host": "0.0.0.0"}
    model = Model().to(device) #use cuda if available

    # CLIENT WORKER SETUP
    #workers on the client that interact with remote workers
    client_workers = []
    for i in range(n_workers):
        client_workers.append(workers.WebsocketClientWorker(id=f"w_{i}", port=base_port+i, **kwargs_websocket)
    
    #local client to test the models of each client as well as of aggregated model
    testing = workers.WebsocketClientWorker(id="testing", port=base_port+i, **kwargs_websocket)

    #serialize the model using jit to then send to remote server workers
    #model, example_input
    traced_model = torch.jit.trace(model, torch.zeros([1, 42], dtype=torch.float))

    # train model on `args.batch_size` batches on each remote worker.
    # after x training batches aggregate model
    # every `args.training_rounds` evaluate each worker's model, and the aggregate model
    for r in range(1, args.training_rounds + 1):
        logger.info(f"Starting training round {r}/{args.training_rounds}")

        # for each training round, distribute model across workers
        # wait until models across all remote server workers are trained and respond
        results = await asyncio.gather(
            *[
                fit_model_on_worker(worker=worker,traced_model=traced_model,
                    batch_size=args.batch_size, curr_round=r,
                    max_nr_batches=args.federate_after_n_batches,lr=args.lr)
                for worker in worker_instances
            ]
        )
        models = {}
        loss_values = {}

        # run test interation every 10 training rounds
        test_models = r % 10 == 1 or r == args.training_rounds
        if test_models: #test each remote server worker's model
            np.set_printoptions(formatter={"float": "{: .0f}".format})
            for worker_id, worker_model, _ in results:
                evaluate_model_on_worker(
                    model_identifier=worker_id,
                    worker=testing,
                    dataset_key="credit_testing",
                    model=worker_model,
                    nr_bins=10,
                    batch_size=128,
                    print_target_hist=False,
                )

        # compile losses from each individual workers
        for worker_id, worker_model, worker_loss in results:
            if worker_model is not None:
                models[worker_id] = worker_model
                loss_values[worker_id] = worker_loss

        # aggregate the model
        traced_model = utils.federated_avg(models)
        if test_models: #test aggregated model
            evaluate_model_on_worker(
                model_identifier="federated model",
                worker=testing,
                dataset_key="credit_testing",
                model=traced_model,
                nr_bins=10,
                batch_size=128
            )

        # decay learning rate
        learning_rate = max(0.98 * learning_rate, args.lr * 0.01)

    if args.save_model:
        torch.save(model.state_dict(), "credit_rating.pt")


if __name__ == "__main__":
    # Logging setup
    logger = logging.getLogger("run_websocket_client")
    logger.setLevel(level=logging.DEBUG)

    # Websockets setup
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())

    # Run main
    asyncio.get_event_loop().run_until_complete(main())
