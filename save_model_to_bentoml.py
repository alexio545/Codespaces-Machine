import bentoml
import hydra
import joblib
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig


def load_model(model_path: str):
    """
    Load a machine learning model from a specified path.

    Args:
        model_path (str): Path to the serialized machine learning model file.

    Returns:
        object: The loaded machine learning model.
    """
    return joblib.load(model_path)


@hydra.main(config_path="../../config", config_name="main")
def save_to_bentoml(config: DictConfig):
    """
    Save a machine learning model to BentoML format.

    This function loads a machine learning model specified in the Hydra configuration,
    then saves it to BentoML format with the specified name.

    Args:
        config (DictConfig): The Hydra configuration specifying model details.

    Returns:
        None
    """
    model = load_model(abspath(config.model.path))
    # bentoml.picklable_model.save(config.model.name, model)
    # bentoml.xgboost.save(config.model.name, model)
    bentoml.xgboost.save_model(config.model.name, model)


if __name__ == "__main__":
    save_to_bentoml()


