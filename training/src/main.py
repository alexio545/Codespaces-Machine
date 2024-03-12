import hydra
from evaluate_model import evaluate
from process import process_data
from train_model import train


@hydra.main(config_path="../../config", config_name="main")
def main(config):
    """
    Main function to execute the machine learning pipeline.

    Args:
        config: Configuration object containing parameters for data processing,
                model training, and evaluation.

    Returns:
        None
    """
    process_data(config)
    train(config)
    evaluate(config)


if __name__ == "__main__":
    main()
