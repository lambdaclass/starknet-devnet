# Patch starknet methods to use cairo_rs_py
import sys

from starknet_devnet.devnet_overrides import *


def cairo_rs_py_monkeypatch():
    setattr(ExecuteEntryPoint, "_run", cairo_rs_py_run)
    setattr(
        sys.modules["starkware.starknet.core.os.class_hash"],
        "class_hash_inner",
        cairo_rs_py_compute_class_hash_inner,
    )
    setattr(
        sys.modules["starkware.starknet.core.os.os_utils"],
        "prepare_os_context",
        cairo_rs_py_prepare_os_context,
    )
    setattr(
        sys.modules["starkware.starknet.core.os.os_utils"],
        "validate_and_process_os_context",
        cairo_rs_py_validate_and_process_os_context,
    )
    setattr(
        BusinessLogicSysCallHandler, "_allocate_segment", cairo_rs_py_allocate_segment
    )
    setattr(
        BusinessLogicSysCallHandler,
        "_read_and_validate_syscall_request",
        cairo_rs_py_read_and_validate_syscall_request,
    )
    setattr(
        BusinessLogicSysCallHandler,
        "validate_read_only_segments",
        cairo_rs_py_validate_read_only_segments,
    )
    setattr(
        sys.modules["starkware.starknet.business_logic.transaction.fee"],
        "calculate_l1_gas_by_cairo_usage)",
        cairo_rs_py_calculate_l1_gas_by_cairo_usage,
    )
    setattr(
        sys.modules["starkware.starknet.core.os.syscall_utils"],
        "get_os_segment_ptr_range",
        cairo_rs_py_get_os_segment_ptr_range,
    )
    setattr(
        sys.modules["starkware.starknet.core.os.segment_utils"],
        "validate_segment_pointers",
        cairo_rs_py_validate_segment_pointers,
    )
    setattr(
        sys.modules["starkware.starknet.business_logic.utils"],
        "get_return_values",
        cairo_rs_py_get_return_values,
    )
