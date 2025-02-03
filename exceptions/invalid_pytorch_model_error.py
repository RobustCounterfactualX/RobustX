class InvalidPytorchModelError(TypeError):
    """Raised when a loaded model is not a valid PyTorch model."""
    def __init__(self,
                 message="The loaded model is not a torch model, but instead relies on a self defined class. \n"
                        "Please save your model again, ensuring to save the underlying torch model, rather than your wrapper class\n"
                            "Then try and load your model in again", model = None):
        message = f"Expected a PyTorch model (torch.nn.Module), but got {type(model).__name__} \nThe loaded model is not a torch model, but instead relies on a self defined class.\nPlease save your model again, ensuring to save the underlying torch model, rather than your wrapper class\nThen try and load your model in again"
        super().__init__(message)
