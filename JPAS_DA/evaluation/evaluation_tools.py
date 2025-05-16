import numpy as np
import logging
import torch

def assert_array_lists_equal(list1, list2, rtol=1e-5, atol=1e-8) -> bool:
    """
    Compare two lists of NumPy arrays and log results for each comparison.

    Parameters:
    ----------
    list1, list2 : list of np.ndarray
        Lists to compare.
    rtol, atol : float
        Tolerances for np.allclose comparison.

    Returns:
    --------
    all_match : bool
        True if all arrays match, False otherwise.
    """
    if len(list1) != len(list2):
        logging.error(f"❌ List lengths differ: {len(list1)} != {len(list2)}")
        return False

    all_match = True
    for i, (arr1, arr2) in enumerate(zip(list1, list2)):
        if np.allclose(arr1, arr2, rtol=rtol, atol=atol):
            logging.debug(f"✅ Arrays at index {i} match.")
        else:
            logging.warning(f"❌ Arrays at index {i} differ.")
            all_match = False

    if all_match:
        logging.info("🎉 All arrays match.")
    else:
        logging.info("⚠️ Some arrays differ.")

    return all_match


def compare_model_parameters(model1, model2, rtol=1e-5, atol=1e-8):
    """Compare parameters of two PyTorch models with optional tolerance.
    
    Returns True if all parameters match, False otherwise.
    """
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    if sd1.keys() != sd2.keys():
        print("❌ Model parameter keys do not match.")
        return False

    all_match = True

    for k in sd1.keys():
        p1 = sd1[k]
        p2 = sd2[k]
        if not torch.allclose(p1, p2, rtol=rtol, atol=atol):
            print(f"❌ Mismatch in parameter: {k}")
            all_match = False

    if all_match:
        print("✅ All parameters match.")
    else:
        print("⚠️ Some parameters differ.")

    return all_match