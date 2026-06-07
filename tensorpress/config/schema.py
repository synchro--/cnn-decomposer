"""User-facing configuration dataclasses for compression workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Callable, Literal

SchedulerType = Literal["cosine", "step", "none"]
MethodType = Literal["tucker", "cpd"]
LayerPredicate = Callable[[str, Any], bool]
LayerSpec = str | list[str] | LayerPredicate | None
RankSpec = str | int | float | list[int] | dict[str, Any]


@dataclass(slots=True)
class FinetuneConfig:
    """Configuration for optional post-compression finetuning.

    Parameters
    ----------
    epochs : int, default=5
        Number of training epochs.
    lr : float, default=1e-4
        Optimizer learning rate.
    scheduler : {"cosine", "step", "none"}, default="cosine"
        Learning rate schedule strategy.
    warmup_steps : int, default=0
        Number of warmup steps before scheduler updates.
    weight_decay : float, default=1e-5
        AdamW weight decay.
    use_amp : bool, default=False
        If True and CUDA is available, enable automatic mixed precision.
    """

    epochs: int = 5
    lr: float = 1e-4
    scheduler: SchedulerType = "cosine"
    warmup_steps: int = 0
    weight_decay: float = 1e-5
    use_amp: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize this configuration as JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinetuneConfig":
        """Build a configuration from a dictionary."""
        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        """Validate finetuning values.

        Raises
        ------
        ValueError
            If any field contains an invalid value.
        """
        if self.epochs <= 0:
            raise ValueError("finetune_config.epochs must be > 0")
        if self.lr <= 0:
            raise ValueError("finetune_config.lr must be > 0")
        if self.warmup_steps < 0:
            raise ValueError("finetune_config.warmup_steps must be >= 0")
        if self.weight_decay < 0:
            raise ValueError("finetune_config.weight_decay must be >= 0")
        if self.scheduler not in ("cosine", "step", "none"):
            raise ValueError("finetune_config.scheduler must be 'cosine', 'step', or 'none'")


@dataclass(slots=True)
class CompressConfig:
    """Top-level user configuration for model compression.

    Parameters
    ----------
    method : {"tucker", "cpd"}, default="tucker"
        Decomposition algorithm to use.
    layers : str | list[str] | Callable[[str, Any], bool] | None, default=None
        Layer selection strategy. Supports:
        - ``None``: same as "all" (every convolution layer)
        - "all": every convolution layer
        - ``list[str]``: exact layer names
        - ``str``: regular-expression pattern
        - predicate: ``predicate(name, module) -> bool``
    ranks : {"auto"} | int | list[int] | dict[str, Any] | float, default="auto"
        Explicit rank selection (power-user knob):
        - "auto": VBMF heuristic per layer
        - ``int``: one rank applied to every selected layer (CP rank, or Tucker
          ``[R, R]``) — convenient for iso-rank comparisons across methods
        - ``list[int]``: explicit rank tuple applied to every selected layer
        - ``dict``: explicit per-layer ranks
        - ``float`` in (0, 1]: **deprecated** alias for ``compression`` (a keep
          fraction). Prefer the ``compression`` field instead.
        Mutually exclusive with ``compression``.
    compression : float | None, default=None
        Friendly compression target: the fraction of parameters to **keep** in
        the layers that are compressed (``0.25`` ≈ keep 25%, i.e. ~4x smaller on
        those layers). Must be in (0, 1]. The library translates this into the
        per-layer rank each method needs to hit that budget, so the same value
        yields different ranks for ``tucker`` vs ``cpd``. The realized fraction
        is reported per layer and in aggregate. Mutually exclusive with an
        explicit ``ranks`` value.
    use_bn : bool, default=False
        Insert batch normalization layers between factorized layers.
    finetune : bool, default=False
        Whether to run finetuning after replacement.
    finetune_config : FinetuneConfig
        Finetuning hyperparameters.
    """

    method: MethodType = "tucker"
    layers: LayerSpec = None
    ranks: RankSpec = "auto"
    compression: float | None = None
    use_bn: bool = False
    finetune: bool = False
    finetune_config: FinetuneConfig = field(default_factory=FinetuneConfig)

    _framework: str = field(default="pytorch", init=False, repr=False)

    def effective_keep_fraction(self) -> float | None:
        """Return the resolved keep-fraction target, or ``None`` if not requested.

        Resolves the ``compression`` field, falling back to the deprecated
        ``float`` form of ``ranks``. Returns ``None`` when ranks are explicit or
        ``"auto"``.
        """
        if self.compression is not None:
            return self.compression
        if isinstance(self.ranks, float) and not isinstance(self.ranks, bool):
            return self.ranks
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation.

        Notes
        -----
        Callable layer selectors are encoded as metadata only and are not
        losslessly deserializable.
        """
        self.validate()
        serialized_layers: Any
        if callable(self.layers):
            serialized_layers = {
                "__callable__": True,
                "name": getattr(self.layers, "__name__", "<anonymous>"),
                "module": getattr(self.layers, "__module__", "<unknown>"),
            }
        else:
            serialized_layers = self.layers

        return {
            "method": self.method,
            "layers": serialized_layers,
            "ranks": self.ranks,
            "compression": self.compression,
            "use_bn": self.use_bn,
            "finetune": self.finetune,
            "finetune_config": self.finetune_config.to_dict(),
            "_framework": self._framework,
        }

    def to_json(self) -> str:
        """Serialize this configuration as JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompressConfig":
        """Build a configuration from a dictionary.

        Raises
        ------
        ValueError
            If a callable selector placeholder is provided.
        """
        payload = dict(data)
        layers = payload.get("layers", None)
        if isinstance(layers, dict) and layers.get("__callable__"):
            raise ValueError(
                "Cannot deserialize callable layer selector automatically. "
                "Provide a concrete layers value."
            )

        finetune_cfg = payload.get("finetune_config", {})
        if isinstance(finetune_cfg, FinetuneConfig):
            normalized_finetune = finetune_cfg
        elif isinstance(finetune_cfg, dict):
            normalized_finetune = FinetuneConfig.from_dict(finetune_cfg)
        else:
            raise ValueError("finetune_config must be a dict or FinetuneConfig")

        cfg = cls(
            method=payload.get("method", "tucker"),
            layers=layers,
            ranks=payload.get("ranks", "auto"),
            compression=payload.get("compression"),
            use_bn=payload.get("use_bn", False),
            finetune=payload.get("finetune", False),
            finetune_config=normalized_finetune,
        )

        framework = payload.get("_framework")
        if isinstance(framework, str) and framework:
            cfg._framework = framework

        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Raise an error for invalid configuration values.

        Raises
        ------
        ValueError
            If any field has an invalid type or unsupported value.
        """
        if self.method not in ("tucker", "cpd"):
            raise ValueError("method must be either 'tucker' or 'cpd'")

        if self.layers is None:
            pass
        elif isinstance(self.layers, str):
            if self.layers != "all":
                # Validate regex early to fail fast.
                import re

                try:
                    re.compile(self.layers)
                except re.error as exc:
                    raise ValueError(f"layers regex is invalid: {self.layers!r}") from exc
        elif isinstance(self.layers, list):
            if not self.layers:
                raise ValueError("layers list must not be empty")
            if not all(isinstance(name, str) and name for name in self.layers):
                raise ValueError("layers list entries must be non-empty strings")
        elif callable(self.layers):
            pass
        else:
            raise ValueError("layers must be None, 'all', regex str, list[str], or callable")

        self._validate_ranks_and_compression()

        if not isinstance(self.use_bn, bool):
            raise ValueError("use_bn must be a bool")
        if not isinstance(self.finetune, bool):
            raise ValueError("finetune must be a bool")
        self.finetune_config.validate()

    def _validate_ranks_and_compression(self) -> None:
        """Validate ``ranks``/``compression`` and enforce mutual exclusion."""
        ranks = self.ranks
        ranks_is_float = isinstance(ranks, float) and not isinstance(ranks, bool)

        if self.compression is not None:
            if not isinstance(self.compression, float) or isinstance(self.compression, bool):
                raise ValueError("compression must be a float in range (0, 1]")
            if not (0.0 < self.compression <= 1.0):
                raise ValueError("compression must be in range (0, 1]")
            if ranks_is_float:
                raise ValueError(
                    "set either 'compression' or a float 'ranks', not both "
                    "(they mean the same thing)"
                )
            if ranks != "auto":
                raise ValueError(
                    "'compression' and explicit 'ranks' are mutually exclusive; "
                    "leave ranks='auto' when using compression"
                )
            return

        if isinstance(ranks, str):
            if ranks != "auto":
                raise ValueError("ranks string must be 'auto'")
        elif ranks_is_float:
            if not (0.0 < ranks <= 1.0):
                raise ValueError(
                    "ranks float (deprecated keep-fraction) must be in (0, 1]; "
                    "prefer the 'compression' field"
                )
        elif isinstance(ranks, bool):
            raise ValueError("ranks must be 'auto', int, list[int], dict, or a compression float")
        elif isinstance(ranks, int):
            if ranks <= 0:
                raise ValueError("ranks int must be > 0")
        elif isinstance(ranks, (list, tuple)):
            if not ranks:
                raise ValueError("ranks list must not be empty")
            if not all(isinstance(r, int) and not isinstance(r, bool) and r > 0 for r in ranks):
                raise ValueError("ranks list entries must be positive ints")
        elif isinstance(ranks, dict):
            if not ranks:
                raise ValueError("ranks dict must not be empty")
            if not all(isinstance(name, str) and name for name in ranks):
                raise ValueError("ranks dict keys must be non-empty strings")
        else:
            raise ValueError("ranks must be 'auto', int, list[int], dict, or a compression float")
