import logging
import syft as sy
from syft.workers import WebsocketServerWorker
import torch
import argparse
from torchvision import datasets
from torchvision import transforms
import numpy as np
from syft.frameworks.torch.federated import utils
from init import n_workers

def get_dataset(worker_number:int):
    """
    returns dataset for specific worker
    
    iterate through global data based on number of workers,
    each worker (incl. testing worker) to have same number of datum
    """
    
    if worker_number > n_workers:
        raise ValueError(f"Only {n_workers} workers available, and 1 testing worker")
    
    features = np.load('../data/features.npy')
    labels = np.load('../data/labels_dim.npy')

    dataset = sy.BaseDataset(data=features[worker_number::n_workers],
                             targets=labels[worker_number::n_workers])
    
    return dataset


def start_websocket_server_worker(worker_id:str, host:str, port:str,
                                  hook:sy.TorchHook, verbose:bool,
                                  training=True):
    """
    Helper function for spinning up a websocket server
    and setting up the local datasets.
    """

    #new remoter server worker
    server = WebsocketServerWorker(id=worker_id, host=host,
                                   port=port, hook=hook,
                                   verbose=verbose)
    if training:
        #get training data
        dataset = get_dataset(int(worker_id[-1]))
        key = "credit"
    else:
        dataset = get_dataset(n_workers)
        key = "credit_testing"

    server.add_dataset(dataset, key=key)

    logger.info("datasets: %s", server.datasets)
    if training:
        logger.info("len(datasets[credit]): %s", len(server.datasets["credit"]))

    server.start()
    return server


if __name__ == "__main__":
    # Logging setup
    logger = logging.getLogger("run_websocket_server")
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d, p:%(process)d) - %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    # Parse args
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="port number of the websocket server worker, e.g. --port 8777",
    )
    parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
    parser.add_argument(
        "--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice"
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="if set, websocket server worker will load the test dataset instead of the training dataset",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket server worker will be started in verbose mode",
    )

    args = parser.parse_args()

    # Hook and start server
    hook = sy.TorchHook(torch)
    server = start_websocket_server_worker(
        worker_id=args.id,
        host=args.host,
        port=args.port,
        hook=hook,
        verbose=args.verbose,
        training=not args.testing,
    )