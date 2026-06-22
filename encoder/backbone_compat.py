"""
backbone_compat.py — Compatibility mixin for non-PNA backbones.

Provides stub methods (freeze_pna, unfreeze_pna, etc.) so that all
backbones can be used interchangeably with code that expects PNA.
"""
import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class BackboneCompatMixin:
    """Mixin providing PNA compatibility stubs for non-PNA backbones."""

    _pna_frozen = False

    def freeze_pna(self):
        self._pna_frozen = True

    def unfreeze_pna(self):
        self._pna_frozen = False

    def override_degree_histogram(self, deg_hist=None):
        pass

    def load_pna_checkpoint(self, checkpoint_path):
        log.warning("%s has no PNA; ignoring %s",
                    type(self).__name__, checkpoint_path)

    def load_ssl_checkpoint(self, ckpt_path):
        """Load an SSL checkpoint with native keys (no remapping)."""
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('backbone_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        # Migrate nn.TransformerEncoder → ModuleList keys
        sd = {k.replace('temporal.transformer.layers.', 'temporal.layers.'): v
              for k, v in sd.items()}
        # Zero-pad global_proj if checkpoint has fewer input dims (e.g. 13→16)
        for suffix in ['backbone.global_proj.weight', 'global_proj.weight']:
            if suffix in sd:
                ckpt_dim = sd[suffix].shape[1]
                try:
                    own_dim = self.backbone.global_proj.in_features if hasattr(self, 'backbone') \
                        else self.global_proj.in_features
                except AttributeError:
                    own_dim = ckpt_dim
                if ckpt_dim < own_dim:
                    padded = torch.zeros(sd[suffix].shape[0], own_dim)
                    padded[:, :ckpt_dim] = sd[suffix]
                    sd[suffix] = padded
                    log.info("Zero-padded %s from %d→%d dims", suffix, ckpt_dim, own_dim)
                break
        info = self.load_state_dict(sd, strict=False)
        if info.missing_keys:
            log.warning("Missing keys when loading SSL checkpoint: %s",
                        info.missing_keys)
        if info.unexpected_keys:
            log.warning("Unexpected keys when loading SSL checkpoint: %s",
                        info.unexpected_keys)
        n_loaded = len(sd) - len(info.unexpected_keys)
        log.info("Loaded %d/%d keys from %s", n_loaded, len(sd), ckpt_path)

    @property
    def pna(self):
        return None

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        return result
