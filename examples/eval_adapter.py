
import argparse
import os
import sys
from typing import List, Tuple, Optional

# Add local project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from flax.training import checkpoints
import optax
from tqdm import tqdm

try:
    import lm_eval
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
except ImportError:
    print("Error: lm_eval not installed. Run: pip install lm_eval")
    sys.exit(1)

from gemma import gm
from gemma.gm.nn import _split_brain
from examples.split_brain_fineweb import get_model_and_config, create_tokenizer

@register_model("split_brain_prophet")
class SplitBrainEvalWrapper(LM):
    def __init__(
        self,
        checkpoint_path: str,
        model_size: str = "270m",
        batch_size: int = 1,
        max_length: int = 2048,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__()
        self.max_length = max_length
        self._batch_size = int(batch_size)

        # Load components
        self.tokenizer = create_tokenizer(model_size)

        # Load config (assuming default split layers for now)
        # For a more robust solution, we might want to save config in the checkpoint
        sb_config = _split_brain.SplitBrainConfig(
            mask_ratio=0.0, # Evaluation doesn't use masking
            prophet_weight=0.0,
        )

        print(f"Loading model {model_size}...")
        self.model, self.base_config = get_model_and_config(model_size, sb_config)

        # Initialize parameters
        print(f"Restoring checkpoint from {checkpoint_path}...")
        init_rng = jax.random.PRNGKey(0)
        dummy_tokens = jnp.ones((1, 8), dtype=jnp.int32)
        initial_variables = self.model.init(init_rng, dummy_tokens, deterministic=True)

        # Restore checkpoint state
        # We need to mimic the TrainState structure used in training
        # But for eval we only need params.
        # However, flax checkpoints usually save the whole state.
        # Let's try to restore directly if it's a raw dict, or via train_state
        from flax.training import train_state

        # Dummy optimizer/state creation to match checkpoint structure
        tx = optax.adamw(1e-4) # Dummy
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=initial_variables['params'],
            tx=tx,
        )

        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_path,
            target=state,
        )

        self.params = restored_state.params
        self.jit_apply = jax.jit(self.model.apply, static_argnames=('deterministic',))

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        results = []
        for i in tqdm(range(0, len(requests), self._batch_size), desc="Evaluating"):
            batch_reqs = requests[i : i + self._batch_size]

            # Prepare batch
            inputs = []
            targets = []
            ctx_lens = []

            for context, continuation in batch_reqs:
                # Encode
                ctx_tokens = self.tokenizer.encode(context)
                cont_tokens = self.tokenizer.encode(continuation) # Note: simple concat usually works, but watch out for spaces

                # Full sequence
                full_tokens = ctx_tokens + cont_tokens
                inputs.append(full_tokens)

                # Target logic: we want logprob(continuation | context)
                # We mask out the context for loss calculation
                ctx_lens.append(len(ctx_tokens))

            # Padding
            max_len = max(len(t) for t in inputs)
            padded_inputs = []
            loss_masks = []

            for j, tokens in enumerate(inputs):
                pad_len = max_len - len(tokens)
                # Pad with 0
                padded_input = tokens + [0] * pad_len
                padded_inputs.append(padded_input)

                # Create mask: 0 for context, 1 for continuation, 0 for padding
                # Transformer predicts next token, so shift by 1?
                # lm_eval standard: input is [A, B], target is [B, C].
                # Logits at pos T predict token at T+1.
                # So to evaluate "B", we look at logits at position of "A".

                # Mask: 1 at positions corresponding to continuation tokens
                # The token at index `k` in input is setting up prediction for `k+1`
                # Continuation starts at `ctx_len`.
                # So we care about predictions from `ctx_len-1` up to `full_len-2`

                mask = [0] * (max_len - 1)
                ctx_len = ctx_lens[j]
                full_len = len(tokens)

                # Indices in `logits` that predict continuation:
                # input[ctx_len-1] predicts input[ctx_len] (first token of continuation)
                # input[full_len-2] predicts input[full_len-1] (last token of continuation)

                for k in range(ctx_len - 1, full_len - 1):
                    if k >= 0:
                        mask[k] = 1.0

                loss_masks.append(mask)

            # To numpy
            input_arr = jnp.array(padded_inputs, dtype=jnp.int32)
            # Input to model is [0..N-1], Targets are [1..N]
            model_inputs = input_arr[:, :-1]
            model_targets = input_arr[:, 1:]
            loss_masks = jnp.array(loss_masks, dtype=jnp.float32)

            # Forward pass
            output = self.jit_apply(
                {'params': self.params},
                model_inputs,
                deterministic=True
            )
            logits = output.logits # [B, L, V]

            # Gather logprobs
            log_softmax = jax.nn.log_softmax(logits, axis=-1)

            # Select target logprobs
            # model_targets: [B, L]
            # log_softmax: [B, L, V]
            # gathered: [B, L]
            gathered_logprobs = jnp.take_along_axis(
                log_softmax, model_targets[..., None], axis=-1
            ).squeeze(-1)

            # Mask and Sum
            # gathered [B, L], loss_masks [B, L]
            seq_logprobs = gathered_logprobs * loss_masks
            sum_logprobs = jnp.sum(seq_logprobs, axis=-1)

            # Check greediness (is continuation the most likely?)
            # max_indices = jnp.argmax(logits, axis=-1)
            # greedy_match = (max_indices == model_targets)
            # This logic is basically: does the model generate the continuation?
            # lm_eval uses simple `is_greedy` check usually based on exact match of all tokens
            # For simplicity here we return False for is_greedy or check properly.

            # Copy to CPU/list
            batch_res = []
            sum_logprobs_np = sum_logprobs.tolist()

            for k in range(len(batch_reqs)):
                batch_res.append((sum_logprobs_np[k], False)) # is_greedy=False for now

            results.extend(batch_res)

        return results

    def loglikelihood_rolling(self, requests):
        pass

    def generate_until(self, requests):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", default="hellaswag")
    parser.add_argument("--batch_size", default=8, type=int)
    args = parser.parse_args()

    lm = SplitBrainEvalWrapper(checkpoint_path=args.checkpoint, batch_size=args.batch_size)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks.split(","),
        batch_size=args.batch_size
    )

    print(results)
