import pytest

from .test_helpers.TestingModels import TestingModels


@pytest.fixture(scope="session")
def testing_models():
    return TestingModels()