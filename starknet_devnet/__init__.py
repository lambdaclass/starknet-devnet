"""
Contains the server implementation and its utility classes and functions.
"""
import logging
import sys
from copy import copy

from crypto_cpp_py.cpp_bindings import cpp_hash
from starkware.starknet.services.api.contract_class import ContractClass

from starknet_devnet.cairo_rs_py_patch import cairo_rs_py_monkeypatch

logger = logging.getLogger(__name__)
__version__ = "0.4.2"


def patched_pedersen_hash(left: int, right: int) -> int:
    """
    Pedersen hash function written in c++
    """
    return cpp_hash(left, right)


# This is a monkey-patch to improve the performance of the devnet
# We are using c++ code for calculating the pedersen hashes
# instead of python implementation from cairo-lang package
setattr(
    sys.modules["starkware.crypto.signature.fast_pedersen_hash"],
    "pedersen_hash",
    patched_pedersen_hash,
)
setattr(
    sys.modules["starkware.cairo.lang.vm.crypto"],
    "pedersen_hash",
    patched_pedersen_hash,
)


# Deep copy of a ContractClass takes a lot of time, but it should never be mutated.
def simpler_copy(self, memo):  # pylint: disable=unused-argument
    """
    A dummy implementation of ContractClass.__deepcopy__
    """
    return copy(self)


setattr(ContractClass, "__deepcopy__", simpler_copy)

# Apply cairo-rs-py patch
cairo_rs_py_monkeypatch()
