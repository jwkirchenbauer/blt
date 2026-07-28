#!/usr/bin/env python3
"""Run a bounded training rehearsal without writing a multi-GB checkpoint."""

from __future__ import annotations

import logging
import os

import bytelatent.train as train_module
from bytelatent.args import TrainArgs
from bytelatent.config_parser import parse_args_to_pydantic_model


def skip_checkpoint_save(
    self,
    model,
    optimizer,
    train_state,
    config,
    device_mesh=None,
):
    logging.getLogger("CHECKPOINT").info(
        "Skipping checkpoint save in an explicitly ephemeral rehearsal."
    )
    return True


def main() -> None:
    # The upstream trainer logs the complete process environment.  Rehearsals
    # use only local checkpoints, so hub credentials are neither needed nor
    # safe to retain in the captured run log.
    for variable in ("HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(variable, None)
    train_module.CheckpointManager.save = skip_checkpoint_save
    train_args = parse_args_to_pydantic_model(TrainArgs)
    if train_args.max_steps is None or train_args.max_steps > 10:
        raise ValueError(
            "train_rehearsal.py requires max_steps <= 10; use bytelatent.train "
            "for actual training"
        )
    train_module.train(train_args)


if __name__ == "__main__":
    main()
