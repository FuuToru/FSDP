from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)

def apply_fsdp_checkpointing(model):
    """Apply activation checkpointing to T5Block modules in the model."""
    print(f"--> Applying FSDP activation checkpointing...")
    for name, submodule in model.named_modules():
        if isinstance(submodule, T5Block):
            print(f"Applying checkpoint_wrapper to: {name}")
            wrapped_module = checkpoint_wrapper(
                submodule,
                offload_to_cpu=False,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            # Replace original submodule with the wrapped one
            setattr(model, name, wrapped_module)
